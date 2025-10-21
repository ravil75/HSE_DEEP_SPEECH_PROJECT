import torch
import torchaudio
import random
from config import Config
from data.dataset import LibriSpeechDataset, collate_fn
from data.text_transform import TextTransform
from model.model import DeepSpeech2
from model.decoder import BeamCTCDecoder

def load_model(checkpoint_path, config, device):
    model = DeepSpeech2(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_wav(wav_path, config):
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != config.sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.sample_rate)(waveform)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length
    )
    spec = mel_spec(waveform)
    spec = torch.log(spec + 1e-9)
    spec = (spec - spec.mean()) / (spec.std() + 1e-9)
    spec = spec.squeeze(0).transpose(0, 1).unsqueeze(0)  # (1, time, n_mels)
    return spec

def predict_wav(model, text_transform, wav_path, config):
    spec = preprocess_wav(wav_path, config).to(config.device)
    with torch.no_grad():
        output = model(spec)
    beam_decoder = BeamCTCDecoder(text_transform, beam_width=10)
    pred = beam_decoder.batch_decode(output)[0]
    print(f"Prediction for '{wav_path}':\n{text_transform.int_to_text(pred) if isinstance(pred, list) else pred}")

def random_predictions(model, text_transform, config, n=10):
    test_dataset = LibriSpeechDataset(
        root=config.data_root,
        url=config.test_url,
        config=config,
        text_transform=text_transform
    )
    beam_decoder = BeamCTCDecoder(text_transform, beam_width=10)
    indices = random.sample(range(len(test_dataset)), n)
    for idx in indices:
        spec, label, spec_len, label_len = test_dataset[idx]
        input_tensor = spec.unsqueeze(0).to(config.device)
        with torch.no_grad():
            output = model(input_tensor)
        pred = beam_decoder.batch_decode(output)[0]
        true = text_transform.int_to_text(label.tolist())
        print(f"\nID: {idx}")
        print("GT   :", true)
        print("PRED :", pred)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pt')
    parser.add_argument('--wav', type=str, default=None, help='Path to a wav file to transcribe')
    parser.add_argument('--num_examples', type=int, default=10, help='Num random examples from test-clean')
    args = parser.parse_args()

    config = Config()
    text_transform = TextTransform()
    model = load_model(args.checkpoint, config, config.device)

    if args.wav:
        predict_wav(model, text_transform, args.wav, config)
    else:
        random_predictions(model, text_transform, config, n=args.num_examples)
