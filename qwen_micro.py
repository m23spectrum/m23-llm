import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class QwenMicroConfig(PretrainedConfig):
    model_type = "qwen_micro"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1376,  # ~2.69 * hidden_size (для SwiGLU)
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # Используем девайс от inv_freq, чтобы корректно обрабатывать meta-device в HF from_pretrained
        current_device = self.inv_freq.device
        t = torch.arange(self.max_seq_len_cached, device=current_device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # cos, sin shape: [seq_len, dim]
    # position_ids shape: [batch_size, seq_len]
    # Убираем лишние размерности для совместимости с батчем
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QwenMicroMLP(nn.Module):
    def __init__(self, config: QwenMicroConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # SwiGLU: (gate(x) * act(up(x))) @ down
        return self.down_proj(self.gate_proj(x) * self.act_fn(self.up_proj(x)))


class QwenMicroAttention(nn.Module):
    def __init__(self, config: QwenMicroConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Вычисляем веса внимания
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class QwenMicroDecoderLayer(nn.Module):
    def __init__(self, config: QwenMicroConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QwenMicroAttention(config)
        self.mlp = QwenMicroMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self Attention + Skip Connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP + Skip Connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


class QwenMicroModel(PreTrainedModel):
    config_class = QwenMicroConfig
    base_model_prefix = "model"

    def __init__(self, config: QwenMicroConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([QwenMicroDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # Подготовка маски внимания (для каузального декодирования)
        causal_mask = torch.full(
            (seq_length, seq_length), fill_value=float("-inf"), device=input_ids.device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]

        if attention_mask is not None:
            # 0 для активных токенов, -inf для замаскированных
            expanded_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=causal_mask.dtype)) * -10000.0
            causal_mask = causal_mask + expanded_mask

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=causal_mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class QwenMicroForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = QwenMicroConfig
    base_model_prefix = "model"

    def __init__(self, config: QwenMicroConfig):
        super().__init__(config)
        self.model = QwenMicroModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Привязка весов головы к эмбеддингам
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        self.lm_head.weight = value.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Сдвигаем токены для предсказания следующего токена
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Расчет кросс-энтропии
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
