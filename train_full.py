# train_full.py
import os
import math
import time
import argparse
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.utils.tokenizer import CharTokenizer
from src.datasets.custom_dir_dataset import CustomDirDataset
from src.datasets.collate import collate_fn
from src.models.baseline_model import SampleCTCModel as DeepSpeech2
from tools.metrics import batch_metrics, compute_wer, compute_cer

# Для визуализации обучения
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# Настройки по умолчанию
DEFAULTS = {
    "data_root": "data",
    "use_torchaudio": False,
    "libri_subset": "train-clean-100",
    "download": False,
    "sample_rate": 16000,
    "hop_length": 160,
    "n_mels": 80,
    "batch_size": 4,
    "epochs": 5,
    "steps_per_epoch": None,
    "lr": 1e-3,
    "rnn_hidden_size": 768,
    "num_rnn_layers": 5,
    "grad_clip": 5.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "save_dir": "outputs",
}

def parse_args():
    """Парсим аргументы командной строки"""
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=DEFAULTS["data_root"])
    p.add_argument("--use_torchaudio", action="store_true")
    p.add_argument("--libri_subset", default=DEFAULTS["libri_subset"])
    p.add_argument("--download", action="store_true")
    p.add_argument("--sample_rate", type=int, default=DEFAULTS["sample_rate"])
    p.add_argument("--hop_length", type=int, default=DEFAULTS["hop_length"])
    p.add_argument("--n_mels", type=int, default=DEFAULTS["n_mels"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--rnn_hidden_size", type=int, default=DEFAULTS["rnn_hidden_size"])
    p.add_argument("--num_rnn_layers", type=int, default=DEFAULTS["num_rnn_layers"])
    p.add_argument("--device", default=DEFAULTS["device"])
    p.add_argument("--grad_clip", type=float, default=DEFAULTS["grad_clip"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--save_dir", default=DEFAULTS["save_dir"])
    p.add_argument("--wandb_proj", default=None)
    p.add_argument("--resume", default=None, help="путь к чекпоинту для продолжения")
    p.add_argument("--val_subset", default="test-clean", help="валидационный набор")
    return p.parse_args()

def init_wandb(args, config: Dict[str, Any]):
    """Инициализация WandB для логирования"""
    if not WANDB_AVAILABLE or args.wandb_proj is None:
        return None
    run = wandb.init(project=args.wandb_proj, config=config, reinit=True)
    return run

def build_dataloaders(args, tokenizer):
    """Создаем загрузчики данных для обучения и валидации"""
    # Тренировочные данные
    train_ds = CustomDirDataset(
        args.data_root,
        sample_rate=args.sample_rate,
        transforms=None,
        return_tensor=True,
        use_torchaudio=args.use_torchaudio,
        librispeech_url=args.libri_subset if args.use_torchaudio else None,
        download=args.download,
    )
    
    # Валидационные данные
    if args.use_torchaudio:
        val_ds = CustomDirDataset(
            args.data_root,
            sample_rate=args.sample_rate,
            transforms=None,
            return_tensor=True,
            use_torchaudio=True,
            librispeech_url=args.val_subset,
            download=args.download,
        )
    else:
        # Ищем папку с валидационными данными
        val_root = os.path.join(args.data_root, "val")
        if not os.path.exists(val_root):
            val_root = os.path.join(args.data_root, "validation")
        if os.path.isdir(val_root):
            val_ds = CustomDirDataset(val_root, sample_rate=args.sample_rate, transforms=None, return_tensor=True)
        else:
            val_ds = None

    # Функция для объединения примеров в батч
    collate = lambda b: collate_fn(b, tokenizer=tokenizer, hop_length=args.hop_length)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers) if val_ds is not None else None
    return train_loader, val_loader

def greedy_decode_batch(model, batch_inputs, device, tokenizer, sample_lengths=None):
    """Декодируем батч: берем самый вероятный символ на каждом шаге"""
    model.eval()
    with torch.no_grad():
        inp = batch_inputs.to(device)
        if sample_lengths is not None:
            logits = model(inp, sample_lengths=sample_lengths)
        else:
            logits = model(inp)
        # Берем argmax по классам
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        # Декодируем в текст
        decs = [tokenizer.decode(p) for p in preds]
    return decs

def run_one_epoch(model, loader, optimizer, ctc_loss, device, tokenizer, args, scaler=None):
    model.train()
    total_loss = 0.0
    n_steps = 0
    pbar = tqdm(loader, desc="train", leave=False)

    for batch in pbar:
        inputs = batch["inputs"].to(device)
        sample_lengths = batch["sample_lengths"]
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        
        # ВАЖНО: Добавляем проверку длин
        with torch.no_grad():
            input_lengths = model.get_output_lengths(sample_lengths).to(device)
            
            # Проверяем, что все input_lengths >= target_lengths
            valid_mask = (input_lengths >= target_lengths) & (input_lengths > 0) & (target_lengths > 0)
            if not valid_mask.all():
                print(f"Warning: Invalid lengths found. Skipping batch. "
                      f"input_lengths: {input_lengths}, target_lengths: {target_lengths}")
                continue
        
        optimizer.zero_grad()
        
        try:
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(inputs, sample_lengths=sample_lengths)
                    log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
                    
                    # Используем только валидные последовательности
                    if not valid_mask.all():
                        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                        if len(valid_indices) == 0:
                            continue
                            
                        log_probs = log_probs[:, valid_indices, :]
                        targets = targets[torch.cat([torch.arange(target_lengths[i].item()) + 
                                                   torch.sum(target_lengths[:i]) for i in valid_indices])]
                        input_lengths = input_lengths[valid_indices]
                        target_lengths = target_lengths[valid_indices]
                    
                    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                
                # Проверяем loss перед backward
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping batch with invalid loss: {loss.item()}")
                    continue
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs, sample_lengths=sample_lengths)
                log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
                
                if not valid_mask.all():
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    if len(valid_indices) == 0:
                        continue
                        
                    log_probs = log_probs[:, valid_indices, :]
                    targets = targets[torch.cat([torch.arange(target_lengths[i].item()) + 
                                               torch.sum(target_lengths[:i]) for i in valid_indices])]
                    input_lengths = input_lengths[valid_indices]
                    target_lengths = target_lengths[valid_indices]
                
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping batch with invalid loss: {loss.item()}")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            if n_steps % 10 == 0:
                pbar.set_postfix({"loss": f"{total_loss / n_steps:.4f}"})
                
        except Exception as e:
            print(f"Error in batch: {e}")
            continue

    return total_loss / max(1, n_steps)

def validate(model, loader, device, tokenizer, args, max_batches: Optional[int] = None):
    """Валидация модели"""
    model.eval()
    all_refs = []  # эталонные тексты
    all_hyps = []  # предсказанные тексты
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="valid", leave=False)
        for i, batch in enumerate(pbar):
            inputs = batch["inputs"]
            sample_lengths = batch["sample_lengths"]
            
            # Декодируем батч
            decs = greedy_decode_batch(model, inputs, device, tokenizer, sample_lengths=sample_lengths)
            refs = batch.get("texts", None)
            
            if refs is None:
                refs = [""] * len(decs)
                
            # Сохраняем результаты
            for r, h in zip(refs, decs):
                all_refs.append(r)
                all_hyps.append(h)
                
            if max_batches is not None and i+1 >= max_batches:
                break
                
    # Считаем метрики
    avg_wer, avg_cer, count = batch_metrics(all_refs, all_hyps, normalize=True, skip_empty_refs=True)
    return avg_wer, avg_cer, count

def save_checkpoint(state: dict, path: str):
    """Сохраняем чекпоинт"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)

def main():
    """Главная функция обучения"""
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if args.device != "" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device={device}, data_root={args.data_root}, use_torchaudio={args.use_torchaudio}")

    # Токенизатор и данные
    tokenizer = CharTokenizer()
    train_loader, val_loader = build_dataloaders(args, tokenizer)
    print(f"[INFO] Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset) if val_loader is not None else 0}")

    # Модель и оптимизатор
    model = DeepSpeech2(
        num_classes=tokenizer.vocab_size,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        rnn_hidden_size=args.rnn_hidden_size,
        num_rnn_layers=args.num_rnn_layers,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    # Mixed precision для GPU
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda") else None

    # Продолжение обучения если нужно
    start_epoch = 1
    best_wer = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 1)
        best_wer = ckpt.get("best_wer", best_wer)
        print(f"[INFO] Resumed from {args.resume} epoch={start_epoch} best_wer={best_wer}")

    # Инициализация WandB
    wandb_run = None
    config = vars(args)
    if args.wandb_proj:
        wandb_run = init_wandb(args, config)

    # Цикл обучения
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        
        # Одна эпоха обучения
        train_loss = run_one_epoch(model, train_loader, optimizer, ctc_loss, device, tokenizer, args, scaler=scaler)
        epoch_time = time.time() - t0

        # Валидация
        if val_loader is not None:
            val_wer, val_cer, val_count = validate(model, val_loader, device, tokenizer, args)
        else:
            val_wer, val_cer, val_count = float("nan"), float("nan"), 0

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_wer={val_wer:.4f} val_cer={val_cer:.4f} time={epoch_time:.1f}s")

        # Обновляем learning rate
        scheduler.step(train_loss)

        # Логируем в WandB
        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_wer": val_wer,
                "val_cer": val_cer,
                "time": epoch_time,
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Сохраняем чекпоинт
        ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth")
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "best_wer": best_wer,
        }, ckpt_path)

        # Сохраняем лучшую модель
        if val_wer < best_wer:
            best_wer = val_wer
            best_path = os.path.join(args.save_dir, "best_checkpoint.pth")
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "best_wer": best_wer,
            }, best_path)
            print(f"[INFO] New best model saved to {best_path} (WER={best_wer:.4f})")

    print("[INFO] Training finished.")

if __name__ == "__main__":
    main()