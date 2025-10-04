import torch
import torch.nn as nn
import torchaudio


class SampleCTCModel(nn.Module):
    """
    Простая модель для ASR с CTC loss.
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

        # Извлечение признаков
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.n_mels, out_channels=self.n_mels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.n_mels),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(
            input_size=self.n_mels, hidden_size=self.hidden,
            num_layers=2, batch_first=True, bidirectional=True
        )

        self.fc = nn.Linear(self.hidden * 2, self.num_classes)

    def forward(self, waveforms: torch.Tensor, sample_lengths=None):
        """
        waveforms: [B, C, T] or [B, T]
        sample_lengths: (не используется, но принимается для совместимости)
        returns logits: [B, T_out, C_classes]
        """
        if waveforms.dim() == 3:
            x = waveforms.squeeze(1) if waveforms.shape[1] == 1 else waveforms.mean(dim=1)
        elif waveforms.dim() == 2:
            x = waveforms
        else:
            raise ValueError(f"Unsupported waveform shape: {waveforms.shape}")

        m = self.melspec(x)

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ---
        # Добавляем clamp для предотвращения log(0) -> -inf
        m = self.amptodb(m.clamp(min=1e-5))
        # ---------------------------

        c = self.conv(m)
        feats = c.transpose(1, 2)

        rnn_out, _ = self.rnn(feats)
        logits = self.fc(rnn_out)
        return logits

    def get_output_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        """Вычисляет длину выхода модели в фреймах."""
        if sample_lengths.dtype != torch.float:
            sample_lengths = sample_lengths.float()
        frames = torch.ceil(sample_lengths / float(self.hop_length)).long()
        return frames