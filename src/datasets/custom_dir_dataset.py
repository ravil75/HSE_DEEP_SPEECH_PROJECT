# src/datasets/custom_dir_dataset.py
from typing import Optional, List, Dict
import os
import glob

import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH

AUDIO_EXTS = [".flac", ".wav"]


def _list_audio_files(directory: str) -> List[str]:
    files: List[str] = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
    return sorted(files)


def _parse_transcription_files(root_dir: str) -> Dict[str, str]:
    """
    Ищет все файлы *.trans.txt под root_dir и парсит их.
    Формат строки обычно: <utterance-id> <transcription...>
    Возвращает dict utt_id -> transcription
    """
    mapping: Dict[str, str] = {}
    # найти все файлы с суффиксом .trans.txt
    pattern = os.path.join(root_dir, "**", "*.trans.txt")
    for trans_path in glob.glob(pattern, recursive=True):
        try:
            with open(trans_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    # разбиение: первый токен — utt_id, остальное — текст
                    parts = line.split(maxsplit=1)
                    if len(parts) == 1:
                        utt, text = parts[0], ""
                    else:
                        utt, text = parts[0], parts[1]
                    mapping[utt] = text
        except Exception:
            # если чтение упало, пропускаем файл (простая и безопасная обработка)
            continue
    return mapping


class CustomDirDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int = 16000,
        transforms=None,
        return_tensor: bool = True,
        use_torchaudio: bool = False,
        librispeech_url: Optional[str] = "train-clean-100",
        download: bool = False,
    ):
        """Dataset для аудиофайлов: поддерживает локальную структуру (root/audio или root/transcriptions)
        или загрузку/проверку LibriSpeech через torchaudio. Возвращает dict с полями
        'waveform', 'sr', 'utt_id', 'text', 'path'. Поддерживает ресемпл, приведение к моно и transforms.
        Args:
            root (str): корневая папка с данными
            sample_rate (int): целевая частота дискретизации аудио
            transforms: функция преобразования аудио (например, для извлечения признаков)
            return_tensor (bool): если True, приводит выход transforms к torch.Tensor
            use_torchaudio (bool): если True, использует torchaudio.datasets.LIBRISPEECH для загрузки/проверки данных
            librispeech_url (str or None): URL набора LibriSpeech для загрузки/проверки (например, "train-clean-100").
                Если None, не использует torchaudio. По умолчанию "train-clean-100".
            download (bool): если True и use_torchaudio=True, скачивает данные LibriSpeech, если их нет
        """

        if torchaudio is None:
            raise ImportError("torchaudio is required by CustomDirDataset. Install torch and torchaudio.")

        self.root = root
        self.sample_rate = int(sample_rate)
        self.transforms = transforms
        self.return_tensor = bool(return_tensor)
        self.use_torchaudio = bool(use_torchaudio)
        self.librispeech_url = librispeech_url
        self.download = bool(download)

        # для кэширования ресемплеров
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        if self.use_torchaudio:
            # 1) скачиваем/проверяем LibriSpeech через torchaudio
            _ = LIBRISPEECH(self.root, url=self.librispeech_url, download=self.download)

            # 2) определим корневую папку с аудио
            audio_root = os.path.join(self.root, "LibriSpeech", self.librispeech_url)
            self.audio_root = audio_root
            # 3) сканируем все аудиофайлы
            self.audio_files = _list_audio_files(audio_root)
            if len(self.audio_files) == 0:
                # проверим на случай, если данные лежат прямо в root
                self.audio_files = _list_audio_files(self.root)
            if len(self.audio_files) == 0:
                raise FileNotFoundError(f"No audio files found for LibriSpeech under {audio_root} or {self.root}")

            # 4) парсим все
            self.transcriptions = _parse_transcription_files(audio_root)

        else:
            self.audio_dir = os.path.join(self.root, "audio")
            if not os.path.isdir(self.audio_dir):
                raise FileNotFoundError(f"audio directory not found: {self.audio_dir}")
            self.audio_files = _list_audio_files(self.audio_dir)
            if len(self.audio_files) == 0:
                raise FileNotFoundError(f"no audio files found in {self.audio_dir}")

            self.trans_dir = os.path.join(self.root, "transcriptions")
            self.transcriptions = {}
            if os.path.isdir(self.trans_dir):
                for p in glob.glob(os.path.join(self.trans_dir, "**/*.txt"), recursive=True):
                    utt = os.path.splitext(os.path.basename(p))[0]
                    try:
                        with open(p, "r", encoding="utf-8") as fh:
                            txt = fh.read().strip()
                    except Exception:
                        txt = ""
                    self.transcriptions[utt] = txt

    def __len__(self):
        return len(self.audio_files)

    def _resample_if_needed(self, waveform: torch.Tensor, sr: int):
        if sr != self.sample_rate:
            key = int(sr)
            if key not in self._resamplers:
                # кэшируем resampler для данного исходного sr
                self._resamplers[key] = torchaudio.transforms.Resample(orig_freq=key, new_freq=self.sample_rate)
            resampler = self._resamplers[key]
            waveform = resampler(waveform)
            sr = self.sample_rate
        return waveform, sr

    def _ensure_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        # если многоканальный, усредняем до моно
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def __getitem__(self, idx: int) -> Dict:
        # получаем путь к аудиофайлу
        path = self.audio_files[idx]
        utt_str = os.path.splitext(os.path.basename(path))[0]

        waveform, sr = torchaudio.load(path)  # [channels, time]
        waveform = self._ensure_mono(waveform)
        waveform, sr = self._resample_if_needed(waveform, int(sr))

        text = self.transcriptions.get(utt_str, None)
        path_info = path
        feats = waveform

        if self.transforms is not None:
            feats = self.transforms(feats, sr)

        if self.return_tensor and not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats)

        return {
            "waveform": feats,
            "sr": sr,
            "utt_id": utt_str,
            "text": text,
            "path": path_info,
        }
