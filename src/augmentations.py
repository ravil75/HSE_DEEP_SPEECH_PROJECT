# src/augmentations.py
import torch
import torch.nn as nn
import torchaudio
import random
import glob
import os
from typing import List, Optional, Tuple

class WaveformAugmentations(nn.Module):
    def __init__(
        self,
        noise_dir: Optional[str] = None,
        noise_snr_db: float = 10.0,
        gaussian_noise_level: float = 0.01,
        gain_min: float = 0.75,
        gain_max: float = 1.25,
        p_noise: float = 0.5,
        p_gain: float = 0.5,
        preload_noise: bool = False,
    ):
        super().__init__()
        self.noise_snr_db = noise_snr_db
        self.gaussian_noise_level = gaussian_noise_level
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.p_noise = p_noise
        self.p_gain = p_gain

        self.noise_files: List[str] = []
        if noise_dir:
            exts = ("*.wav", "*.flac", "*.mp3")
            for e in exts:
                self.noise_files += glob.glob(os.path.join(noise_dir, e))
            self._preloaded_noise: List[Tuple[torch.Tensor,int]] = []
            if preload_noise and len(self.noise_files) > 0:
                for p in self.noise_files:
                    w, sr = torchaudio.load(p)
                    self._preloaded_noise.append((w, sr))

    @staticmethod
    def _rms(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return torch.sqrt(torch.mean(tensor.float() ** 2, dim=-1) + eps)

    def _mix_real_noise(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if len(self.noise_files) == 0:
            return waveform

        if hasattr(self, "_preloaded_noise") and len(self._preloaded_noise) > 0:
            n_wave, n_sr = random.choice(self._preloaded_noise)
        else:
            noise_path = random.choice(self.noise_files)
            n_wave, n_sr = torchaudio.load(noise_path)

        if n_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=n_sr, new_freq=sr)
            n_wave = resampler(n_wave)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if n_wave.size(0) != waveform.size(0):
            if n_wave.size(0) == 1:
                n_wave = n_wave.repeat(waveform.size(0), 1)
            else:
                n_wave = n_wave[: waveform.size(0), :]

        tw = waveform.size(-1)
        tn = n_wave.size(-1)
        if tn < tw:
            reps = int((tw // tn) + 1)
            n_wave = n_wave.repeat(1, reps)
            tn = n_wave.size(-1)

        start = random.randint(0, tn - tw)
        n_seg = n_wave[:, start:start+tw]

        rms_signal = self._rms(waveform)
        rms_noise = self._rms(n_seg)
        snr_lin = 10 ** (self.noise_snr_db / 20.0)
        desired_rms_noise = rms_signal / (snr_lin + 1e-10)
        scale = (desired_rms_noise / (rms_noise + 1e-10)).unsqueeze(-1)
        n_scaled = n_seg * scale.to(n_seg.device).to(n_seg.dtype)

        return waveform + n_scaled

    def add_noise(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if random.random() > self.p_noise:
            return waveform

        device = waveform.device
        dtype = waveform.dtype

        if len(self.noise_files) > 0:
            augmented = self._mix_real_noise(waveform.clone().to(device), sr)
            return augmented.to(dtype)
        else:
            noise = torch.randn_like(waveform, device=device, dtype=dtype) * self.gaussian_noise_level
            return waveform + noise

    def apply_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p_gain:
            return waveform
        gain = random.uniform(self.gain_min, self.gain_max)
        return waveform * gain

    def forward(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if not self.training:
            return waveform
        augmented = waveform.clone()
        augmented = self.apply_gain(augmented)
        augmented = self.add_noise(augmented, sr)
        return augmented


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_masks: int = 2,
        time_masks: int = 2,
        freq_width: int = 27,
        time_width: int = 100,
        p: float = 1.0,
    ):
        super().__init__()
        self.p = p
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width

    def _apply_to_single(self, spec: torch.Tensor) -> torch.Tensor:
        out = spec
        for _ in range(self.freq_masks):
            out = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_width)(out)
        for _ in range(self.time_masks):
            out = torchaudio.transforms.TimeMasking(time_mask_param=self.time_width)(out)
        return out

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p or not self.training:
            return spec

        s = spec
        if s.dim() == 2:
            return self._apply_to_single(s)
        elif s.dim() == 3:
            return self._apply_to_single(s)
        elif s.dim() == 4:
            out = []
            for i in range(s.size(0)):
                out.append(self._apply_to_single(s[i]))
            return torch.stack(out, dim=0)
        else:
            lead = spec.shape[:-2]
            flat = spec.view(-1, spec.shape[-2], spec.shape[-1])
            out = []
            for i in range(flat.size(0)):
                out.append(self._apply_to_single(flat[i]))
            out = torch.stack(out, dim=0)
            return out.view(*lead, spec.shape[-2], spec.shape[-1])
