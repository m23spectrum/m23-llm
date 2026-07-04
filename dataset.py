"""
dataset.py: Загрузка и подготовка датасета для обучения LLM.

Датасет: IlyaGusev/taiga — большой русскоязычный корпус (~23GB текста).
Поддерживает два режима:
  1. Авторегрессионный (AR) — предсказание следующего токена
  2. Диффузионный (Diffusion) — восстановление замаскированных токенов (идея GFusion/Сбер)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from typing import Optional, Literal
import random


class TextDataset(Dataset):
    """
    Датасет для обучения LLM на текстовых данных.

    Поддерживает AR и Diffusion режимы. Текст токенизируется
    и нарезается на куски фиксированного размера (seq_len).
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: GPT2Tokenizer,
        seq_len: int = 512,
        mode: Literal["ar", "diffusion"] = "ar",
        mask_rate_min: float = 0.25,
        mask_rate_max: float = 0.85,
    ):
        """
        Parameters
        ----------
        texts : list[str]
            Список строк текста.
        tokenizer : GPT2Tokenizer
            Токенизатор.
        seq_len : int
            Длина последовательности (контекстное окно).
        mode : str
            "ar"        — авторегрессионный режим (стандартный LM)
            "diffusion" — диффузионный режим (по мотивам GFusion/Сбер)
        mask_rate_min : float
            Минимальный процент маскирования (диффузионный режим).
        mask_rate_max : float
            Максимальный процент маскирования (диффузионный режим).
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mode = mode
        self.mask_rate_min = mask_rate_min
        self.mask_rate_max = mask_rate_max
        self.mask_token_id = tokenizer.eos_token_id  # используем EOS как MASK

        # Токенизируем все тексты и склеиваем в один поток токенов
        print(f"[Dataset] Токенизация {len(texts)} текстов...")
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)  # разделитель

        # Нарезаем на chunks по seq_len+1 (последний токен — таргет)
        self.chunks = []
        for i in range(0, len(all_ids) - seq_len, seq_len):
            chunk = all_ids[i: i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.chunks.append(chunk)

        print(f"[Dataset] Получено {len(self.chunks):,} последовательностей "
              f"по {seq_len} токенов (режим: {mode})")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        tokens = torch.tensor(self.chunks[idx], dtype=torch.long)
        input_ids = tokens[:-1]  # seq_len токенов
        labels = tokens[1:]      # сдвинутые на 1 (следующий токен)

        if self.mode == "ar":
            # Стандартный авторегрессионный режим
            return {"input_ids": input_ids, "labels": labels}

        else:  # diffusion режим (по GFusion)
            return self._apply_diffusion_masking(input_ids, labels)

    def _apply_diffusion_masking(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> dict:
        """
        Диффузионное маскирование по методологии GFusion (Сбер, 2026).

        Идея: вместо предсказания одного следующего токена,
        маскируем случайный процент t токенов и обучаем восстанавливать их.
        t ~ Uniform(mask_rate_min, mask_rate_max)

        Loss считается только по замаскированным позициям.

        Parameters
        ----------
        input_ids : Tensor[seq_len]
            Входные токены.
        labels : Tensor[seq_len]
            Исходные метки (для вычисления loss по маскированным).

        Returns
        -------
        dict с ключами: input_ids, labels, attention_mask, diffusion_mask
        """
        seq_len = input_ids.shape[0]

        # Семплируем уровень шума: t ~ U(0.25, 0.85)
        t = random.uniform(self.mask_rate_min, self.mask_rate_max)

        # Определяем какие позиции маскировать
        n_masked = max(1, int(seq_len * t))
        mask_positions = torch.randperm(seq_len)[:n_masked]
        diffusion_mask = torch.zeros(seq_len, dtype=torch.bool)
        diffusion_mask[mask_positions] = True

        # Создаём зашумлённый вход: заменяем маскированные позиции на MASK токен
        noisy_input = input_ids.clone()
        noisy_input[diffusion_mask] = self.mask_token_id

        # Loss только по маскированным позициям (-100 = игнор в CrossEntropy)
        diffusion_labels = torch.full_like(labels, -100)
        diffusion_labels[diffusion_mask] = labels[diffusion_mask]

        return {
            "input_ids": noisy_input,
            "labels": diffusion_labels,
            "diffusion_mask": diffusion_mask,
            "mask_rate": torch.tensor(t, dtype=torch.float32),
        }


def load_taiga_dataset(
    max_samples: int = 50_000,
    tokenizer: Optional[GPT2Tokenizer] = None,
    seq_len: int = 512,
    mode: str = "ar",
    split: str = "train",
    cache_dir: str = "./data_cache",
) -> TextDataset:
    """
    Загружает русскоязычный датасет Taiga с HuggingFace.

    Датасет: IlyaGusev/taiga
    Объём: ~23GB, мы берём max_samples документов.

    Parameters
    ----------
    max_samples : int
        Максимальное число документов (текстов) для загрузки.
    tokenizer : GPT2Tokenizer
        Токенизатор. Если None — загружается автоматически.
    seq_len : int
        Длина контекстного окна.
    mode : str
        "ar" или "diffusion".
    split : str
        "train" или "test".
    cache_dir : str
        Путь для кэширования датасета.

    Returns
    -------
    TextDataset
        Готовый датасет.
    """
    if tokenizer is None:
        print("[Dataset] Загружаю токенизатор GPT-2...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Dataset] Загружаю Taiga dataset (max_samples={max_samples:,})...")

    # Taiga имеет подмножества, берём самое большое — "proza"
    # Это произведения с прозой на русском языке
    try:
        dataset = load_dataset(
            "cointegrated/taiga_stripped_proza",
            split=split,
            streaming=True,  # Стриминг — не грузим всё в RAM
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"[Dataset] Ошибка при загрузке cointegrated/taiga_stripped_proza: {e}")
        print("[Dataset] Пробую альтернативный источник/конфигурацию...")
        dataset = load_dataset(
            "cointegrated/taiga_stripped_proza",
            split="train" if split == "train" else split,
            streaming=True,
            cache_dir=cache_dir,
        )

    # Собираем тексты
    texts = []
    for sample in dataset:
        text = sample.get("text", "") or sample.get("content", "")
        if text and len(text) > 100:  # Фильтруем слишком короткие
            texts.append(text)
        if len(texts) >= max_samples:
            break

    print(f"[Dataset] Собрано {len(texts):,} документов")

    return TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        seq_len=seq_len,
        mode=mode,
    )


def create_dataloader(
    dataset: TextDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Создаёт DataLoader с нужными настройками.

    Parameters
    ----------
    dataset : TextDataset
        Датасет.
    batch_size : int
        Размер батча. Для 4070 Ti Super с GPT-2: 8-16 безопасно.
    shuffle : bool
        Перемешивать ли данные.
    num_workers : int
        Воркеры. 0 = главный поток (стабильнее на Windows).
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Ускоряет transfer CPU → GPU
        drop_last=True,   # Не используем неполные батчи
    )
