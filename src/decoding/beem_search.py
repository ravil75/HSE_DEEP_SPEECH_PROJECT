# src/decoding/beam_search.py
import torch
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional

class BeamSearchDecoder:
    def __init__(self, tokenizer, beam_size: int = 10, blank_id: int = 0):
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.blank_id = blank_id

    @staticmethod
    def _logaddexp(a: float, b: float) -> float:
        return np.logaddexp(a, b)

    def _process_frame_log(self, logp_frame: np.ndarray, beam: dict) -> dict:
        next_beam = {}
        V = logp_frame.shape[0]

        for prefix, (log_pb, log_pnb) in beam.items():
            log_total = np.logaddexp(log_pb, log_pnb)

            # blank
            lp_blank = logp_frame[self.blank_id]
            cur = next_beam.get(prefix, (-np.inf, -np.inf))
            next_beam[prefix] = (self._logaddexp(cur[0], log_total + lp_blank),
                                 cur[1])


            # non-blanks
            for idx in range(V):
                if idx == self.blank_id:
                    continue
                ch = self.tokenizer.idx2char[idx]

                new_prefix = prefix + ch
                lp = logp_frame[idx]

                # if last char of prefix equals ch, only transitions from log_pb allowed
                if len(prefix) > 0 and prefix[-1] == ch:
                    # extend only from blank-ending prefix to allow repeats separated by blank
                    cur = next_beam.get(new_prefix, (-np.inf, -np.inf))
                    next_beam[new_prefix] = (
                        cur[0],
                        self._logaddexp(cur[1], log_pb + lp)
                    )
                else:
                    # normal extension from either blank or non-blank
                    cur = next_beam.get(new_prefix, (-np.inf, -np.inf))
                    next_beam[new_prefix] = (
                        cur[0],
                        self._logaddexp(cur[1], log_total + lp)
                    )
        return next_beam

    def _prune(self, beam: dict) -> dict:
        scored = []
        for prefix, (log_pb, log_pnb) in beam.items():
            score = np.logaddexp(log_pb, log_pnb)
            scored.append((prefix, score, (log_pb, log_pnb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        kept = scored[:self.beam_size]
        return {p: probs for (p, _, probs) in kept}

    def ctc_beam_search(self, log_probs: np.ndarray) -> List[Tuple[str, float]]:
        # log_probs: [T, V]
        T, V = log_probs.shape
        beam = {"": (0.0, -np.inf)}  # empty prefix: log_pb=0, log_pnb=-inf

        for t in range(T):
            frame = log_probs[t]
            beam = self._process_frame_log(frame, beam)
            beam = self._prune(beam)

        results = []
        for prefix, (log_pb, log_pnb) in beam.items():
            total = np.logaddexp(log_pb, log_pnb)
            results.append((prefix, float(total)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def decode_batch(self, logits: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> List[str]:
        # logits: [B, T, V] or [B, T, V] with padding
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
        batch_size = log_probs.shape[0]
        results = []

        for i in range(batch_size):
            seq = log_probs[i]
            if input_lengths is not None:
                seq = seq[: input_lengths[i]]
            beam_results = self.ctc_beam_search(seq)
            best = beam_results[0][0] if beam_results else ""
            results.append(best)
        return results

