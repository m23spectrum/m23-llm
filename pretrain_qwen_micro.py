"""
pretrain_qwen_micro.py: Скрипт A/B предобучения (from scratch) модели Qwen-Micro.
Сравнивает стандартную инициализацию (Baseline) и M23-Spectrum инициализацию на TinyStories.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Добавляем пути
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen_micro import QwenMicroConfig, QwenMicroForCausalLM
from m23_init import apply_m23_init


class TinyStoriesDataset(torch.utils.data.Dataset):
    """Датасет для обучения на токенизированных текстах."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        # Для языкового моделирования метки (labels) равны входным токенам
        item["labels"] = item["input_ids"].clone()
        return item


def run_pretraining(run_name, use_m23, train_loader, config, tokenizer, max_steps=2000, lr=3e-4):
    print(f"\n{'='*20} НАЧАЛО ПРЕДОБУЧЕНИЯ: {run_name} {'='*20}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # 1. Создание модели с нуля
    model = QwenMicroForCausalLM(config)

    # 2. Инициализация весов
    if use_m23:
        print("[M23-Spectrum] Инициализируем модель через M23-Spectrum...")
        apply_m23_init(model, seed=42, verbose=True)
    else:
        print("[Baseline] Используется стандартная инициализация модели (Normal/Kaiming).")
        # В PyTorch/HuggingFace post_init() уже запускает стандартную инициализацию,
        # так что модель готова к обучению с базовыми весами.

    model.to(device)
    model.train()

    # 3. Оптимизатор и планировщик
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 4. Цикл обучения
    steps = []
    losses = []
    grad_norms = []
    
    step = 0
    start_time = time.time()
    
    # Режим смешанной точности (mixed precision)
    scaler = torch.amp.GradScaler("cuda")
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            
            # Unscale для клиппинга и расчета нормы градиентов
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            step += 1
            steps.append(step)
            losses.append(loss.item())
            grad_norms.append(grad_norm.item())

            if step % 20 == 0 or step == 1:
                elapsed = time.time() - start_time
                print(
                    f"Шаг {step}/{max_steps} | Loss: {loss.item():.4f} | "
                    f"Grad Norm: {grad_norm.item():.4f} | Время: {elapsed:.1f}с"
                )

    elapsed_total = time.time() - start_time
    print(f"[{run_name}] Общее время предобучения: {elapsed_total:.2f} сек.")

    # 5. Обязательное сохранение модели на диск
    output_dir = f"./results_qwen_micro/{run_name}_final"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{run_name}] Сохранение весов и конфигурации в {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Очистка памяти
    del model
    del optimizer
    torch.cuda.empty_cache()

    return {
        "steps": steps,
        "losses": losses,
        "grad_norms": grad_norms,
        "time_seconds": elapsed_total
    }


def main():
    max_steps = 60000  # 60 000 шагов предобучения (~30 минут на модель)
    batch_size = 8
    lr = 3e-4

    # 1. Загрузка токенизатора GPT-2
    print("Загрузка токенизатора GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Загрузка и токенизация датасета TinyStories
    print("Загрузка датасета roneneldan/TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Берем большую выборку для тренировочного лоадера
    sample_size = 150000
    shuffled_dataset = dataset.shuffle(seed=42)
    # Отфильтровываем пустые или некорректные значения (None, нестроковые типы)
    selected_texts = [str(t) for t in shuffled_dataset.select(range(sample_size))["text"] if isinstance(t, str) and len(t.strip()) > 0]
    
    print(f"Токенизация {sample_size} историй...")
    train_dataset = TinyStoriesDataset(selected_texts, tokenizer, max_length=256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # 3. Конфигурация Qwen-Micro
    config = QwenMicroConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,         # Урезанный размер эмбеддинга для скорости
        num_hidden_layers=6,     # 6 декодерных слоев
        num_attention_heads=8,
        intermediate_size=704,   # ~2.75 * hidden_size для SwiGLU
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id
    )

    # Запуск A: Baseline
    baseline_results = run_pretraining(
        run_name="qwen_micro_baseline",
        use_m23=False,
        train_loader=train_loader,
        config=config,
        tokenizer=tokenizer,
        max_steps=max_steps,
        lr=lr
    )

    # Запуск B: M23
    m23_results = run_pretraining(
        run_name="qwen_micro_m23",
        use_m23=True,
        train_loader=train_loader,
        config=config,
        tokenizer=tokenizer,
        max_steps=max_steps,
        lr=lr
    )

    # Сохраняем логи
    output_dir = "./results_qwen_micro"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "baseline": baseline_results,
        "m23": m23_results,
        "config": config.to_dict()
    }
    
    with open(f"{output_dir}/pretrain_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Строим график сравнения лосса
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_results["steps"], baseline_results["losses"], label="Baseline (Kaiming / Normal)", color="red", alpha=0.7)
    plt.plot(m23_results["steps"], m23_results["losses"], label="M23-Spectrum", color="blue", alpha=0.7)
    plt.xlabel("Шаги обучения")
    plt.ylabel("Лосс")
    plt.title("A/B Сравнение предобучения Qwen-Micro (TinyStories)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_dir}/pretrain_loss_comparison.png", dpi=300)
    
    # Строим график сравнения норм градиентов
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_results["steps"], baseline_results["grad_norms"], label="Baseline (Kaiming / Normal)", color="red", alpha=0.7)
    plt.plot(m23_results["steps"], m23_results["grad_norms"], label="M23-Spectrum", color="blue", alpha=0.7)
    plt.xlabel("Шаги обучения")
    plt.ylabel("Норма градиентов")
    plt.title("Сравнение стабильности градиентов Qwen-Micro")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(f"{output_dir}/pretrain_grad_comparison.png", dpi=300)

    print(f"\n[Успешно] Графики сохранены в {output_dir}/")


if __name__ == "__main__":
    main()
