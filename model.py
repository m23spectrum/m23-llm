"""
model.py: GPT-2 с M23-Spectrum инициализацией весов.

Создаёт стандартный GPT-2 (124M параметров) но с НЕСТАНДАРТНОЙ инициализацией:
вместо обычного normal(0, 0.02) используется M23-Spectrum алгебраическая инициализация.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from m23_init import apply_m23_init, get_weight_stats


def create_gpt2_m23(
    vocab_size: int = 50257,
    n_embd: int = 768,
    n_layer: int = 12,
    n_head: int = 12,
    n_positions: int = 1024,
    init_mode: str = "m23",
    seed: int = 42,
) -> GPT2LMHeadModel:
    """
    Создаёт GPT-2 модель с выбранным режимом инициализации.

    Parameters
    ----------
    vocab_size : int
        Размер словаря токенизатора.
    n_embd : int
        Размерность эмбеддингов (768 = GPT-2 Small).
    n_layer : int
        Число трансформер-блоков.
    n_head : int
        Число голов внимания.
    n_positions : int
        Максимальная длина последовательности.
    init_mode : str
        "m23"     — M23-Spectrum инициализация (наш алгоритм)
        "default" — стандартная GPT-2 инициализация (normal 0.02)
        "xavier"  — Xavier Uniform
        "he"      — He (Kaiming) Uniform
    seed : int
        Зерно воспроизводимости.

    Returns
    -------
    GPT2LMHeadModel
        Готовая модель без предобученных весов.
    """
    torch.manual_seed(seed)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=n_positions,
        n_ctx=n_positions,
        bos_token_id=50256,
        eos_token_id=50256,
        # Без дропаута для чистоты сравнения
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )

    # Создаём модель со стандартной инициализацией
    model = GPT2LMHeadModel(config)

    # Применяем выбранную инициализацию
    if init_mode == "m23":
        print(f"[Model] Применяем M23-Spectrum инициализацию (seed={seed})...")
        apply_m23_init(model, seed=seed, verbose=True)

    elif init_mode == "orthogonal":
        print(f"[Model] Применяем стандартную Orthogonal инициализацию...")
        _apply_orthogonal_init(model)

    elif init_mode == "xavier":
        print("[Model] Применяем Xavier Uniform инициализацию...")
        _apply_xavier_init(model)

    elif init_mode == "he":
        print("[Model] Применяем He (Kaiming) инициализацию...")
        _apply_he_init(model)

    else:  # default
        print("[Model] Используем стандартную GPT-2 инициализацию (Normal 0.02)...")

    # Статистика параметров
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Параметров: {n_params:,} ({n_params/1e6:.1f}M)")

    return model


def _apply_xavier_init(model: nn.Module) -> None:
    """Xavier Uniform инициализация для сравнения."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            if isinstance(module, Conv1D):
                fan_in, fan_out = module.weight.shape
                temp = torch.empty(fan_out, fan_in)
                nn.init.xavier_uniform_(temp)
                module.weight.data = temp.t()
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)


def _apply_he_init(model: nn.Module) -> None:
    """He (Kaiming) Uniform инициализация для сравнения."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            if isinstance(module, Conv1D):
                fan_in, fan_out = module.weight.shape
                temp = torch.empty(fan_out, fan_in)
                nn.init.kaiming_uniform_(temp, mode="fan_in", nonlinearity="relu")
                module.weight.data = temp.t()
            else:
                nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


def _apply_orthogonal_init(model: nn.Module) -> None:
    """Стандартная Orthogonal инициализация с масштабированием residual-проекций."""
    n_layer = getattr(model.config, "n_layer", None) if hasattr(model, "config") else None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            scale = 1.0
            if n_layer is not None and any(proj_name in name for proj_name in ["attn.c_proj", "mlp.c_proj"]):
                scale = 1.0 / (2.0 * n_layer) ** 0.5
            
            if isinstance(module, Conv1D):
                fan_in, fan_out = module.weight.shape
                temp = torch.empty(fan_out, fan_in)
                nn.init.orthogonal_(temp)
                module.weight.data = temp.t() * scale
            else:
                nn.init.orthogonal_(module.weight)
                module.weight.data = module.weight.data * scale
                
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight)


def print_weight_summary(model: nn.Module, max_layers: int = 5) -> None:
    """Выводит сводную статистику по весам модели."""
    stats = get_weight_stats(model)
    print(f"\n{'='*60}")
    print(f"{'Слой':<40} {'Mean':>8} {'Std':>8} {'Shape'}")
    print(f"{'='*60}")
    for i, (name, s) in enumerate(stats.items()):
        if i >= max_layers:
            print(f"  ... ещё {len(stats)-max_layers} слоёв")
            break
        shape_str = str(s["shape"])
        print(f"{name:<40} {s['mean']:>8.4f} {s['std']:>8.4f} {shape_str}")
    print(f"{'='*60}\n")
