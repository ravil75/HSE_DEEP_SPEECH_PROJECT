import torch.nn as nn
import torch.nn.functional as F
from model.blocks import ResidualCNN, BidirectionalGRU

class DeepSpeech2(nn.Module):
    def __init__(self, config):
        super(DeepSpeech2, self).__init__()
        self.config = config
        self.cnn = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3//2)
        self.rescnn_layers = nn.Sequential(
            ResidualCNN(32, 32, kernel=3, stride=2, dropout=config.dropout, n_feats=config.n_feats),
            ResidualCNN(32, 32, kernel=3, stride=2, dropout=config.dropout, n_feats=config.n_feats)
        )
        self.fully_connected = nn.Linear(config.n_feats // 4 * 32, config.rnn_dim)
        self.birnn_layers = nn.Sequential(
            *[
                BidirectionalGRU(
                    rnn_dim=config.rnn_dim if i == 0 else config.rnn_dim * 2,
                    hidden_size=config.rnn_dim,
                    dropout=config.dropout,
                    batch_first=True
                )
                for i in range(config.n_rnn_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.rnn_dim * 2, config.rnn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.rnn_dim, config.n_class)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)
        return x.transpose(0, 1)
