# src/datasets/collate.py

from typing import List, Dict, Optional, Any
import math

import torch
import torch.nn.functional as F


def collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    hop_length: Optional[int] = None,
    pad_value: float = 0.0,
):
    # Проверяем что батч не пустой
    if not isinstance(batch, list) or len(batch) == 0:
        raise ValueError("collate_fn получил пустой батч")

    # Инициализируем списки для данных
    waveforms = []
    utts = []
    texts = []
    paths = []
    lengths = []

    # Обрабатываем каждый элемент батча
    for item in batch:
        w = item["waveform"]
        
        # Поддерживаем numpy arrays и torch tensors
        if not isinstance(w, torch.Tensor):
            w = torch.as_tensor(w)
            
        # Приводим к формату [каналы, время] = [C, T]
        if w.dim() == 1:
            w = w.unsqueeze(0)  # [T] -> [1, T]
        elif w.dim() == 3 and w.shape[1] == 1:
            # Если случайно получили [B, 1, T], убираем среднюю размерность
            w = w.squeeze(1)
            if w.dim() == 1:
                w = w.unsqueeze(0)
        if w.dim() != 2:
            raise ValueError(f"Неожиданная форма waveform/features: {w.shape}")
            
        if not torch.is_floating_point(w):
            w = w.float()

        # Сохраняем данные
        waveforms.append(w)
        lengths.append(w.shape[-1])
        utts.append(item.get("utt_id", ""))
        texts.append(item.get("text", None))
        paths.append(item.get("path", ""))

    # Находим максимальную длину в батче для паддинга
    max_len = max(lengths)
    
    # Дополняем все waveform'ы до максимальной длины
    padded = torch.stack([
        F.pad(w, (0, max_len - w.shape[-1]), value=pad_value) 
        for w in waveforms
    ], dim=0)  # [B, C, T_max]

    sample_lengths = torch.tensor(lengths, dtype=torch.long)

    # Формируем выходной словарь с основными данными
    out: Dict[str, Any] = {
        "inputs": padded,           # [B, C, T] - паддированные данные
        "sample_lengths": sample_lengths,  # [B] - исходные длины
        "utt_ids": utts,            # список идентификаторов
        "texts": texts,             # список текстов
        "paths": paths,             # список путей к файлам
    }

    # вычисляем длины в кадрах фичей
    if hop_length is not None:
        if hop_length <= 0:
            raise ValueError("hop_length должен быть > 0")
        input_lengths = torch.ceil(sample_lengths.float() / float(hop_length)).long()
        out["input_lengths"] = input_lengths

    # токенизируем тексты для обучения модели
    if tokenizer is not None:
        all_targets = []
        target_lens = []
        
        for txt in texts:
            if txt is None:
                enc = []
            else:
                enc = tokenizer.encode(txt)
            all_targets.extend(enc)
            target_lens.append(len(enc))
            
        if len(all_targets) == 0:
            out["targets"] = torch.empty(0, dtype=torch.long)
        else:
            out["targets"] = torch.tensor(all_targets, dtype=torch.long)
        out["target_lengths"] = torch.tensor(target_lens, dtype=torch.long)

    return out