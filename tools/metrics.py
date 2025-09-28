# tools/metrics.py
from typing import List, Tuple, Callable
import re

import editdistance

def _normalize_words(s: str):
    s = '' if s is None else str(s)
    return s.strip().lower().split() if s.strip() else []

def compute_wer(target_text: str, pred_text: str):
    target_words = _normalize_words(target_text)
    pred_words = _normalize_words(pred_text)

    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    dist = editdistance.eval(target_words, pred_words)
    return dist / len(target_words)


def compute_cer(target_text: str, pred_text: str):
    tgt = '' if target_text is None else str(target_text).strip()
    pred = '' if pred_text is None else str(pred_text).strip()

    if len(tgt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    dist = editdistance.eval(list(tgt), list(pred))
    return dist / len(tgt)


def batch_metrics(
    refs: List[str], hyps: List[str], normalize: bool = True, skip_empty_refs: bool = True
) -> Tuple[float, float, int]:
    
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have same length")
    sum_wer = 0.0
    sum_cer = 0.0
    count = 0
    for r, h in zip(refs, hyps):
        r_norm = _normalize_words(r) if normalize else (r or "")
        h_norm = _normalize_words(h) if normalize else (h or "")
        if r_norm == "" and skip_empty_refs:
            continue
        wer = compute_wer(r_norm, h_norm, normalize=False)
        cer = compute_cer(r_norm, h_norm, normalize=False)
        sum_wer += wer
        sum_cer += cer
        count += 1
    avg_wer = sum_wer / count if count > 0 else 0.0
    avg_cer = sum_cer / count if count > 0 else 0.0
    return avg_wer, avg_cer, count
