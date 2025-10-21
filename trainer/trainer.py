import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from jiwer import wer, cer
import os
from model.decoder import GreedyDecoder

class Trainer:
    def __init__(self, model, config, train_loader, test_loader, text_transform):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.text_transform = text_transform
        self.decoder = GreedyDecoder(text_transform)
        self.criterion = nn.CTCLoss(blank=config.n_class - 1, zero_infinity=True)
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=config.epochs,
            pct_start=0.2)
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        self.history = {'train_loss': [], 'test_loss': [], 'test_wer': [], 'test_cer': []}
        import os
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for batch_idx, (specs, labels, spec_lengths, label_lengths) in enumerate(pbar):
            specs = specs.to(self.config.device)
            labels = labels.to(self.config.device)
            spec_lengths = spec_lengths.to(self.config.device)
            label_lengths = label_lengths.to(self.config.device)
            self.optimizer.zero_grad()
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(specs)
                    output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long
                    ).to(self.config.device)
                    loss = self.criterion(output, labels, output_lengths, label_lengths)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(specs)
                output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0),
                                           dtype=torch.long).to(self.config.device)
                loss = self.criterion(output, labels, output_lengths, label_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for specs, labels, spec_lengths, label_lengths in tqdm(self.test_loader, desc="Validation"):
                specs = specs.to(self.config.device)
                labels = labels.to(self.config.device)
                spec_lengths = spec_lengths.to(self.config.device)
                label_lengths = label_lengths.to(self.config.device)
                output = self.model(specs)
                output_lengths = torch.full(
                    size=(output.size(1),),
                    fill_value=output.size(0),
                    dtype=torch.long
                ).to(self.config.device)
                loss = self.criterion(output, labels, output_lengths, label_lengths)
                total_loss += loss.item()
                decoded_preds = self.decoder.decode(output)
                for label, label_length in zip(labels, label_lengths):
                    target_text = self.text_transform.int_to_text(label[:label_length].tolist())
                    all_targets.append(target_text)
                all_predictions.extend(decoded_preds)
        avg_loss = total_loss / len(self.test_loader)
        valid_pairs = [(pred, tgt) for pred, tgt in zip(all_predictions, all_targets) 
                if isinstance(pred, str) and isinstance(tgt, str) and pred.strip() != '' and tgt.strip() != '']
        if valid_pairs:
            valid_preds, valid_tgts = zip(*valid_pairs)
            valid_preds = list(valid_preds)
            valid_tgts = list(valid_tgts)
            wer_score = wer(valid_tgts, valid_preds)
            cer_score = cer(valid_tgts, valid_preds)
        else:
            wer_score = 1.0
            cer_score = 1.0
        return avg_loss, wer_score, cer_score
    def train(self):
        print(f"Начало обучения на {self.config.device}")
        print(f"Параметров в модели: {sum(p.numel() for p in self.model.parameters()):,}")
        best_wer = float('inf')
        for epoch in range(self.config.epochs):
            print(f"\n{'='*60}")
            print(f"Эпоха {epoch+1}/{self.config.epochs}")
            print(f"{'='*60}")
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            print("\nВалидация...")
            test_loss, test_wer, test_cer = self.validate()
            self.history['test_loss'].append(test_loss)
            self.history['test_wer'].append(test_wer)
            self.history['test_cer'].append(test_cer)
            print(f"\nРезультаты эпохи {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss:  {test_loss:.4f}")
            print(f"  Test WER:   {test_wer:.4f} ({test_wer*100:.2f}%)")
            print(f"  Test CER:   {test_cer:.4f} ({test_cer*100:.2f}%)")
            if test_wer < best_wer:
                best_wer = test_wer
                self.save_checkpoint(epoch, 'best_model.pt')
                print(f"  ✓ Новая лучшая модель сохранена (WER: {best_wer:.4f})")
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pt')
        self.save_checkpoint(self.config.epochs - 1, 'final_model.pt')
        self.save_history()
        print("\n" + "="*60)
        print("Обучение завершено!")
        print(f"Лучший WER: {best_wer:.4f} ({best_wer*100:.2f}%)")
        print("="*60)
    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    def save_history(self):
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.config.checkpoint_dir, 'training_history.csv'), index=False)
        print(f"История обучения сохранена в {self.config.checkpoint_dir}/training_history.csv")
