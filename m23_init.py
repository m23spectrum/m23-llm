"""
M23-Init: Адаптер для применения M23-Spectrum инициализации к PyTorch моделям.

Совместим с любой архитектурой: Linear, Embedding, Conv2d слои.
Является drop-in заменой стандартных инициализаторов (xavier, kaiming).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from m23_spectrum import build_m23_weight_matrix
from transformers.pytorch_utils import Conv1D


def m23_init_linear(
    module: nn.Module,  # nn.Linear или Conv1D
    seed: Optional[int] = None,
    bias_init: str = "zeros",
    scale: float = 1.0
) -> None:
    """
    Применяет M23-Spectrum инициализацию к nn.Linear или Conv1D слою.

    Parameters
    ----------
    module : nn.Module
        Линейный слой для инициализации.
    seed : Optional[int]
        Зерно воспроизводимости.
    bias_init : str
        Инициализация bias: "zeros" или "small_random".
    scale : float
        Коэффициент масштабирования весов (применяется для residual-проекций).
    """
    if isinstance(module, Conv1D):
        fan_in, fan_out = module.weight.shape
        W = build_m23_weight_matrix(fan_in, fan_out, seed=seed)
        # В Conv1D веса хранятся как (fan_in, fan_out), поэтому транспонируем W (fan_out, fan_in)
        module.weight.data = torch.from_numpy(W).t() * scale
    else:  # nn.Linear
        fan_out, fan_in = module.weight.shape
        W = build_m23_weight_matrix(fan_in, fan_out, seed=seed)
        module.weight.data = torch.from_numpy(W) * scale

    # Инициализация bias
    if module.bias is not None:
        if bias_init == "zeros":
            nn.init.zeros_(module.bias)
        else:
            # Маленькое случайное значение для нарушения симметрии
            nn.init.uniform_(module.bias, -0.01, 0.01)


def m23_init_embedding(
    module: nn.Embedding,
    seed: Optional[int] = None
) -> None:
    """
    Применяет M23-Spectrum инициализацию к nn.Embedding слою.

    Parameters
    ----------
    module : nn.Embedding
        Embedding слой.
    seed : Optional[int]
        Зерно воспроизводимости.
    """
    vocab_size, embed_dim = module.weight.shape

    # Для embedding инициализируем строки (каждое слово) через M23 спектр
    W = build_m23_weight_matrix(embed_dim, vocab_size, seed=seed)
    module.weight.data = torch.from_numpy(W)


def apply_m23_init(model: nn.Module, seed: Optional[int] = None, verbose: bool = True) -> None:
    """
    Применяет M23-Spectrum инициализацию ко всем Linear, Conv1D и Embedding слоям модели.

    Это главная функция — вызывать после создания модели.

    Parameters
    ----------
    model : nn.Module
        PyTorch модель.
    seed : Optional[int]
        Базовое зерно. Каждый слой получает уникальное зерно seed+i.
    verbose : bool
        Выводить статистику инициализированных слоёв.
    """
    linear_count = 0
    embedding_count = 0

    # Получаем n_layer из конфигурации модели для масштабирования residual-проекций
    # Поддерживаем как классический GPT-2 (n_layer), так и современные Qwen/LLaMA (num_hidden_layers)
    n_layer = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", None)) if hasattr(model, "config") else None

    for i, (name, module) in enumerate(model.named_modules()):
        layer_seed = (seed + i) if seed is not None else None

        if isinstance(module, (nn.Linear, Conv1D)):
            scale = 1.0
            
            # Масштабирование остаточных выходов для глубоких сетей (c_proj для GPT-2, o_proj/down_proj для Qwen/LLaMA)
            if n_layer is not None and any(proj_name in name for proj_name in ["attn.c_proj", "mlp.c_proj", "o_proj", "down_proj"]):
                scale = 1.0 / np.sqrt(2.0 * n_layer)
            
            # Адаптация под SwiGLU MLP слои (gate_proj и up_proj перемножаются поэлементно, корректируем дисперсию)
            elif any(proj_name in name for proj_name in ["gate_proj", "up_proj"]):
                scale = 1.0 / np.sqrt(2.0)
                
            m23_init_linear(module, seed=layer_seed, scale=scale)
            linear_count += 1

        elif isinstance(module, nn.Embedding):
            m23_init_embedding(module, seed=layer_seed)
            embedding_count += 1

    if verbose:
        total = linear_count + embedding_count
        print(f"[M23-Init] Инициализировано слоёв: {total}")
        print(f"  - Linear:    {linear_count}")
        print(f"  - Embedding: {embedding_count}")


def get_weight_stats(model: nn.Module) -> dict:
    """
    Возвращает статистику весов модели для диагностики.

    Parameters
    ----------
    model : nn.Module
        PyTorch модель.

    Returns
    -------
    dict
        Словарь с mean, std, min, max для каждого слоя.
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data.float()
            stats[name] = {
                "mean": data.mean().item(),
                "std": data.std().item(),
                "min": data.min().item(),
                "max": data.max().item(),
                "shape": tuple(param.shape),
            }
    return stats


def compare_condition_numbers(fan_in: int, fan_out: int) -> dict:
    """
    Сравнивает обусловленные числа матриц весов при разных инициализациях.

    Parameters
    ----------
    fan_in, fan_out : int
        Размеры матрицы весов.

    Returns
    -------
    dict
        Обусловленные числа для M23, Xavier, He инициализаций.
    """
    import torch
    results = {}

    # M23
    W_m23 = torch.from_numpy(build_m23_weight_matrix(fan_in, fan_out, seed=42))
    results["M23"] = torch.linalg.cond(W_m23.float()).item()

    # Xavier (Glorot)
    W_xavier = torch.empty(fan_out, fan_in)
    nn.init.xavier_uniform_(W_xavier)
    results["Xavier"] = torch.linalg.cond(W_xavier).item()

    # He (Kaiming)
    W_he = torch.empty(fan_out, fan_in)
    nn.init.kaiming_uniform_(W_he, mode="fan_in", nonlinearity="relu")
    results["He"] = torch.linalg.cond(W_he).item()

    return results
