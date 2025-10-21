from config import Config
from logging_utils import setup_logging
from data.dataset import LibriSpeechDataset, collate_fn
from data.text_transform import TextTransform
from model.model import DeepSpeech2
from trainer.trainer import Trainer
from torch.utils.data import DataLoader
import os
import torch

def main():
    os.makedirs("./data/LibriSpeech", exist_ok=True)
    logpath = setup_logging()
    print(f"Логирование включено: {logpath}")
    print("="*60)
    print("DeepSpeech2 - Обучение на LibriSpeech")
    print("="*60)
    config = Config()
    print(f"\nКонфигурация:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
    print(f"  RNN layers: {config.n_rnn_layers}")
    print(f"  RNN dim: {config.rnn_dim}")
    text_transform = TextTransform()
    print(f"\nАлфавит: {len(text_transform.char_map)} символов + blank")
    print("\n" + "="*60)
    print("Подготовка данных")
    print("="*60)
    train_dataset = LibriSpeechDataset(
        root=config.data_root,
        url=config.train_url,
        config=config,
        text_transform=text_transform
    )
    test_dataset = LibriSpeechDataset(
        root=config.data_root,
        url=config.test_url,
        config=config,
        text_transform=text_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True if config.device=="cuda" else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True if config.device=="cuda" else False
    )
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("\n" + "="*60)
    print("Создание модели")
    print("="*60)
    model = DeepSpeech2(config).to(config.device)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        text_transform=text_transform
    )
    trainer.train()
    print("\nГотово! Проверьте папку ./checkpoints для сохраненных моделей.")

if __name__ == "__main__":
    main()
