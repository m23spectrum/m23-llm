"""
train.py: Основной цикл обучения M23-LLM.

Обучает GPT-2 с M23-Spectrum инициализацией на русскоязычном датасете Taiga.
Поддерживает два режима:
  - AR (авторегрессионный): стандартный next-token prediction
  - Diffusion: восстановление замаскированных токенов (по GFusion/Сбер)

Оптимизировано для RTX 4070 Ti Super (16 GB VRAM):
  - bf16 mixed precision
  - gradient checkpointing
  - gradient clipping
"""

import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from datetime import datetime

from model import create_gpt2_m23
from dataset import load_taiga_dataset, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение M23-LLM на датасете Taiga")

    # Модель
    parser.add_argument("--init_mode", type=str, default="m23",
                        choices=["m23", "default", "xavier", "he"],
                        help="Режим инициализации весов")
    parser.add_argument("--n_layer", type=int, default=12,
                        help="Число трансформер-блоков (12 = GPT-2 Small)")
    parser.add_argument("--n_embd", type=int, default=768,
                        help="Размерность эмбеддингов")
    parser.add_argument("--n_head", type=int, default=12,
                        help="Число голов внимания")

    # Данные
    parser.add_argument("--max_samples", type=int, default=50_000,
                        help="Число документов из Taiga")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Длина контекстного окна")
    parser.add_argument("--training_mode", type=str, default="ar",
                        choices=["ar", "diffusion"],
                        help="Режим обучения: ar или diffusion (GFusion-стиль)")

    # Обучение
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Размер батча")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Шаги накопления градиента (effective batch = batch*grad_accum)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10_000,
                        help="Максимальное число шагов обучения")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Шаги разогрева lr scheduler")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Клиппинг градиентов (норма)")

    # Инфраструктура
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Директория для сохранения чекпоинтов")
    parser.add_argument("--save_every", type=int, default=1_000,
                        help="Сохранять чекпоинт каждые N шагов")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Логировать loss каждые N шагов")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Использовать bf16 (рекомендуется для 4070 Ti Super)")

    return parser.parse_args()


def train():
    args = parse_args()

    # Воспроизводимость
    torch.manual_seed(args.seed)

    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Train] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("[Train] ⚠️  GPU не найден, обучение на CPU будет ОЧЕНЬ медленным")

    # Директория сохранения
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Токенизатор
    print("[Train] Загружаю токенизатор...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Датасет
    dataset = load_taiga_dataset(
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        mode=args.training_mode,
    )
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True)

    # Модель
    print(f"\n[Train] Создаю GPT-2 ({args.init_mode} инициализация)...")
    model = create_gpt2_m23(
        vocab_size=tokenizer.vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=args.seq_len,
        init_mode=args.init_mode,
        seed=args.seed,
    )

    # Gradient Checkpointing — экономит VRAM (медленнее, но стабильно)
    model.gradient_checkpointing_enable()
    model = model.to(device)

    # Оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    # LR Scheduler — косинусный с разогревом
    total_steps = min(args.max_steps, len(dataloader) * 10)  # ~10 эпох max
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # AMP Scaler для bf16
    use_amp = args.bf16 and device.type == "cuda"
    dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"[Train] Mixed precision: {'bf16' if use_amp else 'disabled'}")

    # Метрики
    losses = []
    global_step = 0
    start_time = datetime.now()

    # Конфиг запуска
    run_config = {
        "init_mode": args.init_mode,
        "training_mode": args.training_mode,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "max_samples": args.max_samples,
        "seed": args.seed,
    }
    print(f"\n[Train] Конфигурация:\n{json.dumps(run_config, indent=2, ensure_ascii=False)}")
    print(f"\n[Train] Начинаю обучение...")

    # ============================================================
    # ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
    # ============================================================
    model.train()
    optimizer.zero_grad()

    for epoch in range(1000):  # Будем прерывать по max_steps
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            # Переносим батч на GPU
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass с AMP
            with autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                )
                loss = outputs.loss

                # Нормализуем loss для gradient accumulation
                loss = loss / args.grad_accum

            # Backward pass
            loss.backward()

            # Накопление градиентов
            if (global_step + 1) % args.grad_accum == 0:
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Логирование
            actual_loss = loss.item() * args.grad_accum
            losses.append(actual_loss)

            if global_step % args.log_every == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                lr_current = scheduler.get_last_lr()[0]
                elapsed = (datetime.now() - start_time).total_seconds()
                steps_per_sec = (global_step + 1) / elapsed if elapsed > 0 else 0

                print(
                    f"Step {global_step:>6} | "
                    f"Loss: {actual_loss:.4f} (avg100: {avg_loss:.4f}) | "
                    f"LR: {lr_current:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

                # VRAM мониторинг
                if device.type == "cuda":
                    vram_used = torch.cuda.memory_allocated(0) / 1e9
                    vram_peak = torch.cuda.max_memory_allocated(0) / 1e9
                    print(f"         VRAM: {vram_used:.1f}/{vram_peak:.1f} GB (used/peak)")

            # Сохранение чекпоинта
            if global_step > 0 and global_step % args.save_every == 0:
                ckpt_path = save_dir / f"step_{global_step}_{args.init_mode}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": actual_loss,
                    "config": run_config,
                }, ckpt_path)
                print(f"[Train] Чекпоинт сохранён: {ckpt_path}")

            global_step += 1

        if global_step >= args.max_steps:
            break

    # ============================================================
    # ФИНАЛЬНОЕ СОХРАНЕНИЕ
    # ============================================================
    final_path = save_dir / f"final_{args.init_mode}.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "config": run_config,
    }, final_path)

    # Сохраняем историю loss для графиков
    log_path = save_dir / f"loss_history_{args.init_mode}.json"
    with open(log_path, "w") as f:
        json.dump({
            "init_mode": args.init_mode,
            "losses": losses,
            "config": run_config,
        }, f, indent=2)

    elapsed_total = (datetime.now() - start_time).total_seconds()
    final_loss = sum(losses[-100:]) / min(100, len(losses))
    print(f"\n[Train] ✅ Обучение завершено!")
    print(f"  Шагов:       {global_step}")
    print(f"  Финальный loss (avg100): {final_loss:.4f}")
    print(f"  Время:       {elapsed_total/60:.1f} минут")
    print(f"  Модель:      {final_path}")
    print(f"  Loss log:    {log_path}")


if __name__ == "__main__":
    train()
