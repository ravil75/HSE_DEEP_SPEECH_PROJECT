import torch
import torchaudio
import torchaudio.transforms as transforms
import random
import torch.nn as nn

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root, url, config, text_transform):
        self.config = config
        self.text_transform = text_transform
        self.url = url
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root,
            url=url,
            download=True
        )
        self.audio_transform = transforms.MelSpectrogram(sample_rate=config.sample_rate,
            n_mels=config.n_mels, n_fft=config.n_fft, hop_length=config.hop_length,
            win_length=config.win_length)
        self.time_mask = transforms.TimeMasking(time_mask_param=30)
        self.freq_mask = transforms.FrequencyMasking(freq_mask_param=13)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        waveform, sample_rate, transcript, *_ = self.dataset[idx]
        is_train = "train" in self.url.lower() or self.url.lower().startswith("dev")
        if is_train:
            if random.random() < 0.5:
                noise = torch.randn_like(waveform) * 0.003
                waveform = waveform + noise
            if random.random() < 0.5:
                factor = random.choice([0.9, 1.0, 1.1])
                if factor != 1.0:
                    new_sr = int(sample_rate * factor)
                    waveform = transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)(waveform)
                    if new_sr != sample_rate:
                        waveform = transforms.Resample(orig_freq=new_sr, new_freq=sample_rate)(waveform)
        spec = self.audio_transform(waveform)
        spec = torch.log(spec + 1e-9)
        spec = (spec - spec.mean()) / (spec.std() + 1e-9)
        spec = spec.squeeze(0).transpose(0, 1)
        if is_train:
            if random.random() < 0.5:
                spec = self.time_mask(spec)
            if random.random() < 0.5:
                spec = self.freq_mask(spec)
        label = torch.tensor(self.text_transform.text_to_int(transcript))
        return spec, label, spec.shape[0], len(label)

def collate_fn(batch):
    specs, labels, spec_lengths, label_lengths = zip(*batch)
    specs_padded = nn.utils.rnn.pad_sequence(specs, batch_first=True)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    spec_lengths = torch.tensor(spec_lengths)
    label_lengths = torch.tensor(label_lengths)
    return specs_padded, labels_padded, spec_lengths, label_lengths
