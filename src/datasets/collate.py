# src/datasets/collate.py

from typing import List, Dict, Optional, Any
import torch
import torch.nn.functional as F
import torchaudio
import torch.nn as nn

def collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    melspec_transform: Optional[nn.Module] = None,
    spec_augmentor: Optional[nn.Module] = None,
    hop_length: Optional[int] = None,
):
    if not isinstance(batch, list) or len(batch) == 0:
        raise ValueError("collate_fn получил пустой батч")
    
    specs, utts, texts, paths = [], [], [], []
    waveforms, sample_lengths = [], []

    for item in batch:
        w = item["waveform"]
        if not isinstance(w, torch.Tensor):
            w = torch.as_tensor(w)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if not torch.is_floating_point(w):
            w = w.float()

        waveforms.append(w)
        sample_lengths.append(w.shape[-1])
        utts.append(item.get("utt_id", ""))
        texts.append(item.get("text", None))
        paths.append(item.get("path", ""))

    max_len = max(sample_lengths)
    padded_waveforms = torch.stack([
        F.pad(w, (0, max_len - w.shape[-1]), value=0.0) for w in waveforms
    ], dim=0)

    inputs = padded_waveforms
    if melspec_transform is not None:

        if inputs.dim() == 3 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        
        # Вычисляем спектрограмму
        inputs = melspec_transform(inputs) # -> [B, F, T]
        
        # Применяем SpecAugment, если он есть
        if spec_augmentor is not None:
            inputs = spec_augmentor(inputs)

        inputs = inputs.unsqueeze(1)


    out: Dict[str, Any] = {
        "inputs": inputs,
        "waveforms": padded_waveforms,
        "sample_lengths": torch.tensor(sample_lengths, dtype=torch.long),
        "utt_ids": utts,
        "texts": texts,
        "paths": paths,
    }

    if hop_length is not None:
        input_lengths = torch.ceil(out["sample_lengths"].float() / float(hop_length)).long()
        out["input_lengths"] = input_lengths

    if tokenizer is not None:
        all_targets = []
        target_lens = []
        for txt in texts:
            enc = tokenizer.encode(txt or "")
            all_targets.extend(enc)
            target_lens.append(len(enc))
        
        out["targets"] = torch.tensor(all_targets, dtype=torch.long) if all_targets else torch.empty(0, dtype=torch.long)
        out["target_lengths"] = torch.tensor(target_lens, dtype=torch.long)

    return out