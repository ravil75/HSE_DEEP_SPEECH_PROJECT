import os
from collections import defaultdict
import argparse

import torch
from tqdm import tqdm

from src.datasets.custom_dir_dataset import CustomDirDataset
from src.models.baseline_model import SampleCTCModel
from src.utils.tokenizer import CharTokenizer

EMPTY = "^"

# Beam utils (same as before, kept simple)
def process_frame(frame_probs, beam, ind2char):
    next_beam = defaultdict(float)
    for idx, p in enumerate(frame_probs):
        ch = ind2char[idx]
        for (prefix, last), pref_p in beam.items():
            if ch == last:
                new_pref, new_last = prefix, last
            else:
                if ch != EMPTY:
                    new_pref, new_last = prefix + ch, ch
                else:
                    new_pref, new_last = prefix, ch
            next_beam[(new_pref, new_last)] += pref_p * p
    return dict(next_beam)

def top_k(state, k):
    return dict(sorted(state.items(), key=lambda it: it[1], reverse=True)[:k])

def ctc_beam_search(probs, beam_size, ind2char):
    beam = {("", EMPTY): 1.0}
    for frame in probs:
        beam = process_frame(frame, beam, ind2char)
        beam = top_k(beam, beam_size)
    return beam

def try_infer_num_classes_from_state(state_dict):
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.dim() == 2:
            return v.size(0)
    return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="path to data folder (or Libri root if using torchaudio)")
    p.add_argument("--checkpoint", default="checkpoint.pth", help="path to model checkpoint")
    p.add_argument("--output", default="predictions", help="output dir for predictions")
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    num_classes = try_infer_num_classes_from_state(state_dict)
    if num_classes is None:
        raise RuntimeError("Не удалось определить число классов из чекпойнта")

    model = SampleCTCModel(num_classes=num_classes)
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state, strict=False)

    model.to(args.device).eval()

    tokenizer = CharTokenizer()
    ind2char = {i: c for i, c in tokenizer.idx2char.items()}
    ind2char[0] = EMPTY  # blank

    # Dataset (non-torchaudio mode). If you want torchaudio, modify accordingly.
    ds = CustomDirDataset(args.data_dir)

    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc="Inference"):
            item = ds[idx]
            utt = item["utt_id"]
            wav = item["waveform"]

            # normalize shapes -> [1, 1, T] -> model expects [B, C, T] or [B, T]
            if wav.dim() == 1:
                inp = wav.unsqueeze(0).to(args.device)
            elif wav.dim() == 2:
                if wav.shape[0] == 1:
                    inp = wav.unsqueeze(0).to(args.device)  # [1,1,T]
                else:
                    inp = wav.mean(dim=0, keepdim=True).unsqueeze(0).to(args.device)
            else:
                raise ValueError(f"Неподдерживаемая форма wave: {wav.shape}")

            inp = inp.float()
            logits = model(inp)  # [B, T, C]
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()  # [T, C]

            beam = ctc_beam_search(probs, args.beam_size, ind2char)
            best_pair, best_score = max(beam.items(), key=lambda it: it[1])  # best_pair == (prefix, last)
            pred = best_pair[0]

            out_path = os.path.join(args.output, f"{utt}.beam{args.beam_size}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(pred + "\n")

    print(f"[INFO] Predictions saved in {args.output}")

if __name__ == "__main__":
    main()
