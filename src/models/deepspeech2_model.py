# src/models/deepspeech2_model.py
import torch
import torch.nn as nn
import torchaudio
from typing import Optional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        rnn_hidden_size: int = 768,
        num_rnn_layers: int = 5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.num_rnn_layers = int(num_rnn_layers)

        # Преобразование аудио в мел-спектрограмму
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB()

        # Сверточные слои для извлечения признаков
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Вычисляем размер входа для RNN
        rnn_input_size = self._get_rnn_input_size()

        # Рекуррентные слои GRU
        self.rnns = nn.ModuleList()
        for i in range(self.num_rnn_layers):
            in_size = rnn_input_size if i == 0 else self.rnn_hidden_size * 2
            self.rnns.append(
                nn.GRU(
                    input_size=in_size,
                    hidden_size=self.rnn_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        # Нормализация между слоями RNN
        self.norms = nn.ModuleList([nn.LayerNorm(self.rnn_hidden_size * 2) for _ in range(self.num_rnn_layers)])

        # Финальный классификатор
        self.fc = nn.Linear(self.rnn_hidden_size * 2, self.num_classes)

    def _get_rnn_input_size(self) -> int:
        """Считаем сколько признаков получается после сверток"""
        with torch.no_grad():
            dummy = torch.randn(1, 1, self.n_mels, 100)
            out = self.conv(dummy)
        return int(out.shape[1] * out.shape[2])

    def forward(self, waveforms: torch.Tensor, sample_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Приводим вход к формату [B, T]
        if waveforms.dim() == 3 and waveforms.shape[1] == 1:
            waveforms = waveforms.squeeze(1)
        elif waveforms.dim() == 3 and waveforms.shape[1] > 1:
            waveforms = waveforms.mean(dim=1)
        elif waveforms.dim() != 2:
            raise ValueError(f"Неподдерживаемая форма waveform: {waveforms.shape}")

        device = waveforms.device

        # Извлекаем мел-спектрограмму
        x = self.melspec(waveforms)
        x = self.amptodb(x)

        # Применяем свертки
        x = x.unsqueeze(1)
        x = self.conv(x)

        # Меняем shape для RNN [B, T_out, features]
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Обработка переменной длины через pack_padded_sequence
        if sample_lengths is not None:
            rnn_lengths = self.get_output_lengths(sample_lengths.to(device))
            rnn_lengths = torch.clamp(rnn_lengths, min=1).long().to(device)

            # Сортируем по убыванию длины для упаковки
            sorted_lens, perm_idx = rnn_lengths.sort(descending=True)
            x_sorted = x[perm_idx]
            
            packed = pack_padded_sequence(x_sorted, sorted_lens.cpu(), batch_first=True, enforce_sorted=True)

            # Проходим через все RNN слои
            packed_seq = packed
            for i in range(self.num_rnn_layers):
                packed_out, _ = self.rnns[i](packed_seq)
                padded, lengths_after = pad_packed_sequence(packed_out, batch_first=True)
                padded = self.norms[i](padded)
                packed_seq = pack_padded_sequence(padded, lengths_after.cpu(), batch_first=True, enforce_sorted=True)

            # Распаковываем и возвращаем исходный порядок
            padded_final, _ = pad_packed_sequence(packed_seq, batch_first=True)
            inv_perm = perm_idx.argsort()
            x = padded_final[inv_perm]
        else:
            # Простой forward без упаковки
            for i in range(self.num_rnn_layers):
                x, _ = self.rnns[i](x)
                x = self.norms[i](x)

        # Финальный линейный слой
        logits = self.fc(x)
        return logits

    def get_output_lengths(self, sample_lengths: torch.Tensor, use_ceil_for_mels: bool = True) -> torch.Tensor:
        """Считаем сколько временных шагов будет после сверток"""
        L = sample_lengths.cpu().long().clone()
        
        # Мел-фреймы
        if use_ceil_for_mels:
            L = torch.ceil(L.float() / float(self.hop_length)).long()
        else:
            L = torch.clamp(((L - self.n_fft) // self.hop_length) + 1, min=0)

        # Формула для сверток
        def conv_time_len(L_in, kernel_time, stride_time, pad_time):
            out = (L_in + 2 * pad_time - kernel_time) // stride_time + 1
            return torch.clamp(out, min=0)

        L = conv_time_len(L, kernel_time=11, stride_time=2, pad_time=5)
        L = conv_time_len(L, kernel_time=11, stride_time=1, pad_time=5)

        return L
