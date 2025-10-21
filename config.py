import torch

class Config:
    """Конфигурация модели и обучения"""
    data_root = "./data/LibriSpeech"
    train_url = "train-clean-100"
    test_url = "test-clean"
    sample_rate = 16000
    n_mels = 80
    n_fft = 400
    hop_length = 160
    win_length = 400
    n_cnn_layers = 2
    n_rnn_layers = 5
    rnn_dim = 512
    n_class = 29
    n_feats = n_mels
    stride = 2
    dropout = 0.1
    learning_rate = 5e-4
    batch_size = 20
    epochs = 50
    num_workers = 4
    use_amp = True
    grad_clip = 1.0
    weight_decay = 1e-6
    checkpoint_dir = "./checkpoints"
    save_every = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
