# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from src.utils.tokenizer import CharTokenizer
from src.datasets.custom_dir_dataset import CustomDirDataset
from src.datasets.collate import collate_fn
from src.models.baseline_model import SampleCTCModel

# -----------------------
DATA_ROOT = os.getenv("DATA_ROOT", "data")        # путь к данным или к корню для torchaudio download
USE_TORCHAUDIO = os.getenv("USE_TORCHAUDIO", "0") == "1" # использовать torchaudio.datasets.LIBRISPEECH
LIBRI_SUBSET = os.getenv("LIBRI_SUBSET", "train-clean-100")
DOWNLOAD = os.getenv("DOWNLOAD", "0") == "1"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
STEPS = int(os.getenv("STEPS", "100"))
LR = float(os.getenv("LR", "1e-3"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "160"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "10"))
# -----------------------

def main():
    device = torch.device(DEVICE)
    print(f"[INFO] device: {device}, data: {DATA_ROOT}, use_torchaudio: {USE_TORCHAUDIO}")

    # tokenizer
    tokenizer = CharTokenizer()

    # dataset + loader
    ds = CustomDirDataset(DATA_ROOT, use_torchaudio=USE_TORCHAUDIO,
                          librispeech_url=LIBRI_SUBSET, download=DOWNLOAD, sample_rate=SAMPLE_RATE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer=tokenizer, hop_length=HOP_LENGTH),
                        num_workers=0)

    # модель, оптимизатор, loss
    model = SampleCTCModel(num_classes=tokenizer.vocab_size, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    try:
        batch = next(iter(loader))
    except Exception as e:
        print(f"[ERROR] Failed to fetch batch from dataset: {e}")
        return

    inputs = batch["inputs"].to(device)            # [B, C, T]
    sample_lengths = batch["sample_lengths"]       # Tensor[B] (in samples)
    input_lengths = batch.get("input_lengths", None)
    targets = batch.get("targets", None)
    target_lengths = batch.get("target_lengths", None)

    if targets is None or target_lengths is None or target_lengths.sum().item() == 0:
        print("[ERROR] No transcripts in batch. Put transcriptions into dataset or use filesystem-mode with transcriptions.")
        return

    targets = targets.to(device)
    target_lengths = target_lengths.to(device)
    if input_lengths is None:
        try:
            input_lengths = model.get_output_lengths(sample_lengths).to(device)
        except Exception:
            input_lengths = torch.ceil(sample_lengths.float() / float(HOP_LENGTH)).long().to(device)

    print(f"[INFO] Starting one-batch training: batch_size={BATCH_SIZE}, steps={STEPS}, lr={LR}")

    model.train()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        logits = model(inputs)  # [B, T_out, C]
        log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)  # [T_out, B, C]
        loss = ctc(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        if step % PRINT_EVERY == 0 or step == 1:
            print(f"step {step}/{STEPS} loss={loss.item():.6f}")

    print("Done. One-batch training finished.")

if __name__ == "__main__":
    main()
