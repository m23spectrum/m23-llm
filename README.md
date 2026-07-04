# M23-LLM

**GPT-2 с алгебраической инициализацией весов на основе группы Матьё M23**

Объединяет два источника:
- [M23-Spectrum](https://github.com/m23spectrum/m23-spectrum) — инициализация через полином Элки
- [GFusion (Сбер/Хабр)](https://habr.com/ru/companies/sberbank/articles/1054690/) — диффузионный режим обучения

---

## Быстрый старт

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Сравнение инициализаций (500 шагов, небольшая модель)
python compare_init.py

# 3. Полное обучение с M23 + диффузионный режим
python train.py --init_mode m23 --training_mode diffusion --max_steps 10000

# 4. Сравнение режимов обучения
python train.py --init_mode m23 --training_mode ar --max_steps 5000
python train.py --init_mode default --training_mode ar --max_steps 5000
```

---

## Структура проекта

```
m23-llm/
├── m23_spectrum.py     # Ядро алгоритма M23 (полином Элки, SVD)
├── m23_init.py         # Адаптер: M23 для PyTorch слоёв
├── model.py            # GPT-2 фабрика (m23 / xavier / he / default)
├── dataset.py          # Taiga датасет (AR + Diffusion режимы)
├── train.py            # Основной цикл обучения
├── compare_init.py     # Бенчмарк 4 инициализаций → графики
└── requirements.txt
```

---

## Алгоритм M23-Spectrum

Вместо случайных весов (Xavier, He) — детерминированные через алгебраическую структуру:

1. **Полином Элки**: `g⁴ + g³ + 9g² - 10g + 8 = 0`  
   Корни кодируют спектральные свойства группы Матьё M23 (порядок 10 200 960)

2. **SVD-декомпозиция**: `W = U · diag(σ) · Vᵀ`  
   Где `σ` — нормализованный M23-спектр

3. **Результат**: матрица с оптимальным обусловленным числом → стабильный градиентный поток

**Заявленный эффект**: 2.8× быстрее сходимость, 8× лучше обусловленность vs He

---

## Диффузионный режим (по GFusion/Сбер)

Вместо предсказания *следующего* токена — восстановление *замаскированных*:

```
t ~ Uniform(0.25, 0.85)   # уровень шума
masked_tokens = random t% of sequence
model learns: masked_tokens → original_tokens
```

**Преимущество**: параллельная генерация нескольких токенов → ускорение инференса

---

## Hardware

Оптимизировано для **RTX 4070 Ti Super (16 GB)**:
- bf16 mixed precision
- gradient checkpointing
- batch_size=8, seq_len=512 → ~8 GB VRAM

---

## Датасет

[IlyaGusev/taiga](https://huggingface.co/datasets/IlyaGusev/taiga) — русскоязычный корпус ~23GB.  
По умолчанию используется подмножество `proza` (художественная проза).
