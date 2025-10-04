# src/models/baseline_model.py
import torch
import torch.nn as nn
import torchaudio


class SampleCTCModel(nn.Module):
    """
    временный пример простой модели для ASR с CTC loss.
    """

    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        hidden: int = 256,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.hidden = int(hidden)

        # feature extractor
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB()

        # small conv block (operates on [B, n_mels, T])
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.n_mels, out_channels=self.n_mels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_mels),
            nn.ReLU(),
        )

        # RNN: input_size = n_mels, bidir GRU
        self.rnn = nn.GRU(input_size=self.n_mels, hidden_size=self.hidden, num_layers=2, batch_first=True, bidirectional=True)

        # project to classes
        self.fc = nn.Linear(self.hidden * 2, self.num_classes)

    def forward(self, waveforms: torch.Tensor, sample_lengths=None):
        """
        waveforms: [B, C, T] or [B, T]
        returns logits: [B, T_out, C_classes]
        """
        if waveforms.dim() == 3:
            # [B, C, T] -> squeeze channel if C==1
            if waveforms.shape[1] == 1:
                x = waveforms.squeeze(1)  # [B, T]
            else:
                # if multiple channels, average them
                x = waveforms.mean(dim=1)
        elif waveforms.dim() == 2:
            x = waveforms
        else:
            raise ValueError(f"Unsupported waveform shape: {waveforms.shape}")

        # (..., time) -> returns [B, n_mels, T_feats]
        m = self.melspec(x)  # [B, n_mels, T_f]
        m = self.amptodb(m)  # dB scale
        c = self.conv(m)     # [B, n_mels, T_f]
        feats = c.transpose(1, 2)  # [B, T_f, n_mels]

        rnn_out, _ = self.rnn(feats)  # [B, T_f, hidden*2]
        logits = self.fc(rnn_out)     # [B, T_f, num_classes]
        return logits

    def get_output_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        if sample_lengths.dtype != torch.float:
            sample_lengths = sample_lengths.float()
        frames = torch.ceil(sample_lengths / float(self.hop_length)).long()
        return frames
