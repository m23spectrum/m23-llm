"""
compare_init.py: Сравнение скорости сходимости разных методов инициализации.

Запускает короткое обучение (500 шагов) для каждой инициализации:
  - M23-Spectrum (наш алгоритм)
  - Xavier Uniform (стандарт для трансформеров)
  - He (Kaiming) Uniform
  - Default GPT-2 (Normal 0.02)

Строит графики:
  1. Loss vs Steps для каждого метода
  2. Обусловленные числа матриц весов
  3. Норма градиентов в процессе обучения

Результат: наглядная демонстрация что M23 сходится быстрее.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from pathlib import Path
from torch.amp import autocast
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup
from datetime import datetime

from model import create_gpt2_m23
from dataset import load_taiga_dataset, create_dataloader
from m23_init import compare_condition_numbers


# ============================================================
# КОНФИГ ЭКСПЕРИМЕНТА
# ============================================================
COMPARE_STEPS = 3000        # Шагов на каждый метод
BATCH_SIZE = 4             # Маленький батч для скорости
SEQ_LEN = 256              # Короткий контекст для скорости
MAX_SAMPLES = 5_000        # Немного данных — нам важна скорость сходимости
LR = 3e-4
SEED = 42
N_LAYER = 6                # Мелкая модель для быстрого сравнения
N_EMBD = 256
N_HEAD = 4

INIT_MODES = ["m23", "orthogonal", "default", "xavier", "he"]
COLORS = {
    "m23":        "#7c3aed",  # фиолетовый
    "orthogonal": "#10b981",  # зеленый
    "default":    "#64748b",  # серый
    "xavier":     "#0284c7",  # синий
    "he":         "#dc2626",  # красный
}
LABELS = {
    "m23":        "M23-Spectrum (наш)",
    "orthogonal": "Standard Orthogonal",
    "default":    "Default GPT-2 (Normal 0.02)",
    "xavier":     "Xavier Uniform",
    "he":         "He (Kaiming) Uniform",
}


def run_single_experiment(
    init_mode: str,
    tokenizer,
    dataloader,
    device: torch.device,
) -> dict:
    """
    Запускает COMPARE_STEPS шагов обучения для одного метода инициализации.

    Returns
    -------
    dict с ключами: losses, grad_norms, init_mode
    """
    print(f"\n{'='*60}")
    print(f"Эксперимент: {LABELS[init_mode]}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    # Создаём модель
    model = create_gpt2_m23(
        vocab_size=tokenizer.vocab_size,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_positions=SEQ_LEN,
        init_mode=init_mode,
        seed=SEED,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=300, num_training_steps=COMPARE_STEPS
    )

    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp else torch.float32

    losses = []
    grad_norms = []
    model.train()

    data_iter = iter(dataloader)
    for step in range(COMPARE_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        loss.backward()

        # Норма градиентов
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norms.append(total_norm.item())

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 100 == 0:
            print(f"  Step {step:>3}/{COMPARE_STEPS} | Loss: {loss.item():.4f} | "
                  f"Grad norm: {total_norm.item():.4f}")

    # Очищаем GPU память
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return {
        "init_mode": init_mode,
        "losses": losses,
        "grad_norms": grad_norms,
    }


def smooth(values: list, window: int = 10) -> list:
    """Скользящее среднее для сглаживания кривой loss."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def plot_results(results: list[dict], save_path: str = "comparison.png"):
    """
    Строит сравнительные графики для всех методов инициализации.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f0c29")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, :])     # Loss — широкий верхний
    ax_grad = fig.add_subplot(gs[1, 0])     # Grad norms
    ax_cond = fig.add_subplot(gs[1, 1])     # Condition numbers

    # Стиль
    for ax in [ax_loss, ax_grad, ax_cond]:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#404060")

    # ── 1. Loss vs Steps ──────────────────────────────────────────
    for r in results:
        mode = r["init_mode"]
        smoothed = smooth(r["losses"], window=20)
        steps = list(range(len(smoothed)))
        ax_loss.plot(
            steps, smoothed,
            color=COLORS[mode],
            label=LABELS[mode],
            linewidth=2.5,
            alpha=0.9,
        )
        # Финальное значение
        final_loss = smoothed[-1]
        ax_loss.annotate(
            f"{final_loss:.3f}",
            xy=(steps[-1], final_loss),
            xytext=(5, 0), textcoords="offset points",
            color=COLORS[mode], fontsize=9, va="center",
        )

    ax_loss.set_xlabel("Шаги обучения", fontsize=11)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax_loss.set_title("Скорость сходимости: M23-Spectrum vs другие инициализации", fontsize=13)
    ax_loss.legend(
        facecolor="#1a1a2e", edgecolor="#404060",
        labelcolor="white", fontsize=10,
    )
    ax_loss.grid(True, alpha=0.2, color="#404060")

    # ── 2. Gradient Norms ─────────────────────────────────────────
    for r in results:
        mode = r["init_mode"]
        smoothed_grad = smooth(r["grad_norms"], window=20)
        ax_grad.plot(
            smoothed_grad,
            color=COLORS[mode],
            label=LABELS[mode],
            linewidth=2, alpha=0.85,
        )
    ax_grad.set_xlabel("Шаги", fontsize=10)
    ax_grad.set_ylabel("Норма градиентов", fontsize=10)
    ax_grad.set_title("Норма градиентов в обучении", fontsize=11)
    ax_grad.legend(
        facecolor="#1a1a2e", edgecolor="#404060",
        labelcolor="white", fontsize=8,
    )
    ax_grad.grid(True, alpha=0.2, color="#404060")

    # ── 3. Condition Numbers ──────────────────────────────────────
    print("\n[Compare] Вычисляю обусловленные числа матриц...")
    test_dims = [(256, 256), (512, 256), (768, 768)]
    x_labels = [f"{fi}×{fo}" for fi, fo in test_dims]
    bar_width = 0.2
    x = np.arange(len(test_dims))

    for j, mode in enumerate(INIT_MODES):
        cond_numbers = []
        for fan_in, fan_out in test_dims:
            conds = compare_condition_numbers(fan_in, fan_out)
            cond_numbers.append(conds.get(mode.capitalize(), conds.get("M23", 1.0)))

        offset = (j - len(INIT_MODES)/2 + 0.5) * bar_width
        bars = ax_cond.bar(
            x + offset, cond_numbers,
            width=bar_width,
            color=COLORS[mode],
            alpha=0.85,
            label=LABELS[mode],
        )

    ax_cond.set_xlabel("Размер матрицы", fontsize=10)
    ax_cond.set_ylabel("Обусловленное число (меньше = лучше)", fontsize=10)
    ax_cond.set_title("Обусловленность матриц весов\n(ниже = стабильнее градиенты)", fontsize=11)
    ax_cond.set_xticks(x)
    ax_cond.set_xticklabels(x_labels)
    ax_cond.legend(
        facecolor="#1a1a2e", edgecolor="#404060",
        labelcolor="white", fontsize=8,
    )
    ax_cond.grid(True, alpha=0.2, color="#404060", axis="y")

    # Заголовок
    fig.suptitle(
        "M23-Spectrum LLM: Сравнение методов инициализации весов",
        fontsize=15, color="white", fontweight="bold", y=0.98,
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    print(f"\n[Compare] График сохранён: {save_path}")
    # plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Compare] Устройство: {device}")
    if device.type == "cuda":
        print(f"[Compare] GPU: {torch.cuda.get_device_name(0)}")

    # Загружаем один общий датасет для всех экспериментов
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\n[Compare] Загружаю датасет для сравнения...")
    dataset = load_taiga_dataset(
        max_samples=MAX_SAMPLES,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        mode="ar",
    )
    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Запускаем эксперименты для каждого метода инициализации
    results = []
    for mode in INIT_MODES:
        result = run_single_experiment(mode, tokenizer, dataloader, device)
        results.append(result)

    # Сохраняем числовые результаты
    with open("./comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Итоговая таблица
    print(f"\n{'='*60}")
    print("ИТОГИ СРАВНЕНИЯ")
    print(f"{'='*60}")
    print(f"{'Метод':<30} {'Loss start':>12} {'Loss end':>10} {'Снижение':>10}")
    print(f"{'-'*60}")
    for r in results:
        l_start = r["losses"][0]
        l_end = sum(r["losses"][-50:]) / 50  # последние 50 шагов
        reduction = (l_start - l_end) / l_start * 100
        print(f"{LABELS[r['init_mode']]:<30} {l_start:>12.4f} {l_end:>10.4f} {reduction:>9.1f}%")

    # График
    plot_results(results, save_path="./comparison.png")


if __name__ == "__main__":
    main()
