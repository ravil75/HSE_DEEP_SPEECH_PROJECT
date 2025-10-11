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
import numpy as np

from tqdm import tqdm

from src.utils.tokenizer import CharTokenizer
from src.datasets.custom_dir_dataset import CustomDirDataset
from src.datasets.collate import collate_fn
from src.models.deepspeech2_model import DeepSpeech2
from tools.metrics import batch_metrics, compute_wer, compute_cer
from src.utils.logging import init_logging, compute_grad_norm, log_example

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
    "lr": 3e-5,
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
    p.add_argument("--resume", default=None, help="путь к чекпоинту для продолжения")
    p.add_argument("--val_subset", default="test-clean", help="валидационный набор")

    # Logging-related
    p.add_argument("--wandb_proj", default=None, help="WandB project name (set to enable wandb logging)")
    p.add_argument("--wandb_name", default=None, help="WandB run name (optional)")
    p.add_argument("--comet_api_key", default=None, help="Comet API key (set to enable Comet logging)")
    p.add_argument("--comet_project", default=None, help="Comet project name")
    p.add_argument("--comet_workspace", default=None, help="Comet workspace")
    p.add_argument("--log_every_steps", type=int, default=50, help="log samples/grad-norm every N steps")

    return p.parse_args()

def build_dataloaders(args, tokenizer):
    """Создаем загрузчики данных для обучения и валидации"""
    train_ds = CustomDirDataset(
        args.data_root, sample_rate=args.sample_rate, use_torchaudio=args.use_torchaudio,
        librispeech_url=args.libri_subset if args.use_torchaudio else None, download=args.download,
    )
    
    val_ds = None
    if args.use_torchaudio:
        val_ds = CustomDirDataset(
            args.data_root, sample_rate=args.sample_rate, use_torchaudio=True,
            librispeech_url=args.val_subset, download=args.download,
        )
    else:
        val_root = os.path.join(args.data_root, "val")
        if os.path.isdir(val_root):
            val_ds = CustomDirDataset(val_root, sample_rate=args.sample_rate)

    collate = lambda b: collate_fn(b, tokenizer=tokenizer, hop_length=args.hop_length)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers) if val_ds is not None else None
    return train_loader, val_loader

def greedy_decode_batch(model, batch_inputs, device, tokenizer, sample_lengths=None):
    """Декодируем батч: берем самый вероятный символ на каждом шаге"""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            inp = batch_inputs.to(device)
            logits = model(inp, sample_lengths=sample_lengths)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            decs = [tokenizer.decode(p) for p in preds]
    finally:
        if was_training:
            model.train()
    return decs

def run_one_epoch(model, loader, optimizer, ctc_loss, device, tokenizer, args, scaler=None, wandb_run=None, comet_exp=None, global_step_start=0):
    model.train()
    total_loss = 0.0
    n_steps = 0
    global_step = int(global_step_start)
    pbar = tqdm(loader, desc="train", leave=False)

    for batch in pbar:
        optimizer.zero_grad()

        inputs = batch["inputs"].to(device)
        sample_lengths = batch["sample_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        # Проверка соответствия длин для CTCLoss
        with torch.no_grad():
            input_lengths = model.get_output_lengths(sample_lengths).to(device)
            if not (input_lengths >= target_lengths).all():
                print(f"[WARN] Invalid lengths found. Skipping batch. Input: {input_lengths.tolist()}, Target: {target_lengths.tolist()}")
                continue

        try:
            use_amp = (scaler is not None) and (device.type == "cuda")
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inputs, sample_lengths=sample_lengths)

                # Проверка на численную нестабильность
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("[WARN] Found NaN/Inf in logits. Skipping batch.")
                    continue

                log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Backward + optimizer step
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = compute_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = compute_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            global_step += 1

            # progress bar
            if n_steps % 10 == 0:
                pbar.set_postfix({"loss": f"{total_loss / n_steps:.4f}"})

            # Logging to W&B / Comet
            current_lr = optimizer.param_groups[0].get("lr", 0.0)
            if wandb_run is not None:
                try:
                    wandb_run.log({
                        "train/loss_step": loss.item(),
                        "train/grad_norm": grad_norm,
                        "train/lr": current_lr,
                    }, step=global_step)
                except Exception as e:
                    print(f"[WARN] wandb log failed: {e}")

            if comet_exp is not None:
                try:
                    comet_exp.log_metric("train.loss_step", loss.item(), step=global_step)
                    comet_exp.log_metric("train.grad_norm", grad_norm, step=global_step)
                    comet_exp.log_metric("train.lr", current_lr, step=global_step)
                except Exception as e:
                    print(f"[WARN] comet log failed: {e}")

            # Логирование средних WER/CER по батчу каждые N шагов
            if global_step % args.log_every_steps == 0:
                # Декодируем весь батч для вычисления средних метрик
                try:
                    all_decs = greedy_decode_batch(model, inputs, device, tokenizer, sample_lengths=sample_lengths)
                    all_refs = batch.get("texts", [""] * len(all_decs))
                    
                    batch_wers = []
                    batch_cers = []
                    for ref, pred in zip(all_refs, all_decs):
                        try:
                            wer_val = compute_wer(ref, pred)
                            cer_val = compute_cer(ref, pred)
                            batch_wers.append(wer_val)
                            batch_cers.append(cer_val)
                        except Exception:
                            continue
                    
                    if batch_wers and batch_cers:  # если есть валидные метрики
                        avg_wer = np.mean(batch_wers)
                        avg_cer = np.mean(batch_cers)
                        
                        if wandb_run is not None:
                            wandb_run.log({
                                "train/wer_batch": avg_wer,
                                "train/cer_batch": avg_cer,
                            }, step=global_step)
                            
                        if comet_exp is not None:
                            comet_exp.log_metric("train_wer_batch", avg_wer, step=global_step)
                            comet_exp.log_metric("train_cer_batch", avg_cer, step=global_step)
                            
                        print(f"[Step {global_step}] Batch metrics - WER: {avg_wer:.4f}, CER: {avg_cer:.4f}")
                            
                except Exception as e:
                    print(f"[WARN] Batch metrics computation failed: {e}")

                # Логирование одного примера (оставляем существующий код)
                try:
                    single_inp = inputs[:1]
                    single_len = sample_lengths[:1]
                    decs = greedy_decode_batch(model, single_inp, device, tokenizer, sample_lengths=single_len)
                    pred_text = decs[0] if len(decs) > 0 else ""
                except Exception:
                    pred_text = ""

                ref_text = batch.get("texts", [None])[0] if batch.get("texts") else ""
                try:
                    wer = compute_wer(ref_text, pred_text)
                    cer = compute_cer(ref_text, pred_text)
                except Exception:
                    wer, cer = 0.0, 0.0

                # waveform -> numpy 1D
                wf_np = None
                try:
                    wf = inputs[0].detach().cpu().numpy()
                    if wf.ndim == 3:
                        # [B,C,T] -> [C,T] for first sample
                        wf = wf[0]
                    if wf.ndim == 2:
                        wf_np = wf.mean(axis=0)
                    else:
                        wf_np = wf
                    max_a = max(1e-8, float(abs(wf_np).max()))
                    wf_np = wf_np / max_a
                except Exception:
                    wf_np = None

                utt_id = batch.get("utt_ids", ["sample"])[0]
                log_example(wandb_run, comet_exp, utt_id, wf_np, args.sample_rate, ref_text or "", pred_text, wer, cer, step=global_step)

        except Exception as e:
            print(f"[ERROR] Error processing batch: {e}. Skipping.")
            continue

    avg_loss = total_loss / max(1, n_steps)
    return avg_loss, global_step


def validate(model, loader, device, tokenizer, args, max_batches: Optional[int] = None):
    """Валидация модели"""
    model.eval()
    all_refs, all_hyps = [], []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="valid", leave=False)
        for i, batch in enumerate(pbar):
            decs = greedy_decode_batch(model, batch["inputs"], device, tokenizer, sample_lengths=batch["sample_lengths"])
            refs = batch.get("texts", [""] * len(decs))
            all_refs.extend(refs)
            all_hyps.extend(decs)
            if max_batches is not None and i + 1 >= max_batches:
                break
                
    avg_wer, avg_cer, count = batch_metrics(all_refs, all_hyps)
    return avg_wer, avg_cer, count

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def main():
    """Главная функция обучения"""
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[INFO] Device={device}, data_root={args.data_root}, use_torchaudio={args.use_torchaudio}")

    tokenizer = CharTokenizer()
    train_loader, val_loader = build_dataloaders(args, tokenizer)
    print(f"[INFO] Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset) if val_loader else 0}")

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
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch, best_wer = 1, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_wer = ckpt.get("best_wer", float("inf"))
        print(f"[INFO] Resumed from {args.resume} epoch={start_epoch} best_wer={best_wer:.4f}")

    wandb_run, comet_exp = init_logging(
        wandb_project=args.wandb_proj,
        wandb_name=args.wandb_name,
        comet_api_key=args.comet_api_key,
        comet_project=args.comet_project,
        comet_workspace=args.comet_workspace,
        comet_name=args.wandb_name,
        config=vars(args)
    )

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, global_step = run_one_epoch(model, train_loader, optimizer, ctc_loss, device, tokenizer, 
                                                args, scaler, wandb_run=wandb_run, comet_exp=comet_exp, 
                                                global_step_start=global_step)

        epoch_time = time.time() - t0

        val_wer, val_cer = float("nan"), float("nan")
        if val_loader:
            val_wer, val_cer, _ = validate(model, val_loader, device, tokenizer, args)

            # Log a couple of validation examples (audio + spec + text) for inspection
            try:
                val_batch = next(iter(val_loader))
                decs = greedy_decode_batch(model, val_batch["inputs"], device, tokenizer, sample_lengths=val_batch["sample_lengths"])
                n_examples = min(2, len(decs))
                for i in range(n_examples):
                    pred = decs[i]
                    ref = val_batch.get("texts", [""] * len(decs))[i]
                    utt = val_batch.get("utt_ids", ["val_sample"] * len(decs))[i]
                    wf = val_batch["inputs"][i].cpu().numpy()
                    if wf.ndim == 3:
                        wf = wf[0]
                    if wf.ndim == 2:
                        wf_np = wf.mean(axis=0)
                    else:
                        wf_np = wf
                    max_a = max(1e-8, float(abs(wf_np).max()))
                    wf_np = wf_np / max_a
                    wer_v = compute_wer(ref, pred)
                    cer_v = compute_cer(ref, pred)
                    log_example(wandb_run, comet_exp, f"val_{utt}", wf_np, args.sample_rate, ref, pred, wer_v, cer_v, step=global_step)
            except Exception as e:
                print(f"[WARN] val sample logging failed: {e}")

        print(f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_wer={val_wer:.4f} val_cer={val_cer:.4f} time={epoch_time:.1f}s")
        scheduler.step(val_wer if not math.isnan(val_wer) else train_loss)

        # Log epoch-level metrics to W&B/Comet (tied to epoch number)
        if wandb_run:
            try:
                wandb_run.log({
                    "train_loss_epoch": train_loss,
                    "val_wer": val_wer,
                    "val_cer": val_cer,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=epoch)
            except Exception as e:
                print(f"[WARN] wandb epoch log failed: {e}")

        if comet_exp:
            try:
                comet_exp.log_metric("train_loss_epoch", train_loss, step=epoch)
                comet_exp.log_metric("val_wer", val_wer, step=epoch)
                comet_exp.log_metric("val_cer", val_cer, step=epoch)
                comet_exp.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)
            except Exception as e:
                print(f"[WARN] comet epoch log failed: {e}")

        is_best = val_wer < best_wer
        if is_best:
            best_wer = val_wer
            best_path = os.path.join(args.save_dir, "best_checkpoint.pth")
            save_checkpoint({"epoch": epoch, "state_dict": model.state_dict(), "best_wer": best_wer}, best_path)
            print(f"[INFO] New best model saved to {best_path} (WER={best_wer:.4f})")

            # Save as W&B artifact / Comet asset if available
            if wandb_run:
                try:
                    wandb_run.summary["best_val_wer"] = best_wer
                    artifact = wandb_run.Artifact("best-checkpoint", type="model")
                    artifact.add_file(best_path)
                    wandb_run.log_artifact(artifact)
                except Exception as e:
                    print(f"[WARN] Failed to push checkpoint to wandb: {e}")
            if comet_exp:
                try:
                    comet_exp.log_asset(best_path)
                except Exception as e:
                    print(f"[WARN] Failed to push checkpoint to Comet: {e}")

    # Properly finish logging sessions
    if wandb_run:
        try:
            wandb_run.finish()
        except Exception:
            pass
    if comet_exp:
        try:
            comet_exp.end()
        except Exception:
            pass

    print("[INFO] Training finished.")

if __name__ == "__main__":
    main()
