"""
generate_stories.py: Скрипт генерации сказок для сравнения результатов A/B предобучения.
Загружает сохраненные модели и генерирует тексты на основе заданного начала.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

# Добавляем пути
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen_micro import QwenMicroForCausalLM


def generate_text(model, tokenizer, prompt, max_length=100, do_sample=False, temperature=0.7, top_k=0, top_p=0.0):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство для инференса: {device}")

    # Пути к сохраненным моделям
    baseline_dir = "./results_qwen_micro/qwen_micro_baseline_final"
    m23_dir = "./results_qwen_micro/qwen_micro_m23_final"

    if not os.path.exists(baseline_dir) or not os.path.exists(m23_dir):
        print("[Ошибка] Не найдены сохраненные модели. Сначала запусти pretrain_qwen_micro.py!")
        sys.exit(1)

    print("Загрузка токенизатора и моделей...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_dir)

    # Загружаем Baseline
    print("Загрузка Baseline модели...")
    baseline_model = QwenMicroForCausalLM.from_pretrained(baseline_dir).to(device)
    baseline_model.eval()

    # Загружаем M23
    print("Загрузка M23 модели...")
    m23_model = QwenMicroForCausalLM.from_pretrained(m23_dir).to(device)
    m23_model.eval()

    # Промпты для генерации
    prompts = [
        "Once upon a time, a little girl named Lily",
        "Tom had a small toy car. One day, he went to the park",
        "A cute dog wanted to eat a big red apple"
    ]

    print("\n" + "="*50)
    print("СРАВНЕНИЕ ГЕНЕРАЦИИ СКАЗОК")
    print("="*50)

    for i, prompt in enumerate(prompts):
        print(f"\n--- Промпт #{i+1}: '{prompt}' ---")
        
        print("\n[Baseline Model Output]:")
        baseline_out = generate_text(baseline_model, tokenizer, prompt, max_length=120)
        print(baseline_out)
        
        print("\n[M23 Model Output]:")
        m23_out = generate_text(m23_model, tokenizer, prompt, max_length=120)
        print(m23_out)
        print("-" * 50)


if __name__ == "__main__":
    main()
