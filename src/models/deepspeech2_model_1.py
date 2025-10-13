# src/models/deepspeech2_model_1.py.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Вспомогательный класс для LayerNorm в свертках, как в статье
class CNNLayerNorm(nn.Module):
    """Layer normalization built for CNNs input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x: (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)

# ResidualCNN блок из статьи
class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x) # Используем GELU как в статье
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)

# BidirectionalGRU модуль из статьи
class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

# Основная модель, собранная из блоков выше
class ArticleDeepSpeech(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2 from the article"""
    def __init__(
        self,
        num_classes: int,
        n_feats: int, # Это ваши n_mels
        n_cnn_layers: int = 3,
        n_rnn_layers: int = 5,
        rnn_dim: int = 512,
        stride: int = 2,
        dropout: float = 0.1,
        hop_length: int = 160 # нужно для get_output_lengths
    ):
        super().__init__()
        self.hop_length = hop_length
        # n_feats для CNN будет в 2 раза меньше из-за stride=2 в первой свертке
        cnn_n_feats = n_feats // stride
        
        # Начальная свертка для уменьшения размерности по частоте
        self.cnn = nn.Conv2d(1, 32, kernel_size=3, stride=(stride, 1), padding=3 // 2)

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=cnn_n_feats)
            for _ in range(n_cnn_layers)
        ])
        
        self.fully_connected = nn.Linear(cnn_n_feats * 32, rnn_dim)
        
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(
                rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                hidden_size=rnn_dim, dropout=dropout, batch_first=True
            )
            for i in range(n_rnn_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # BiRNN удваивает размерность
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, sample_lengths: Optional[torch.Tensor] = None):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (B, C * F, T)
        x = x.transpose(1, 2)  # (B, T, C * F)
        
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        logits = self.classifier(x)
        return logits

    def get_output_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        """Вычисляет длину выхода модели в фреймах."""
        # Это упрощенный расчет, так как свертки не меняют T
        if sample_lengths.dtype != torch.float:
            sample_lengths = sample_lengths.float()
        frames = torch.ceil(sample_lengths / float(self.hop_length)).long()
        return frames