# src/utils/logging.py
import io
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

try:
    import wandb
    _WANDB = True
except Exception:
    wandb = None
    _WANDB = False

try:
    from comet_ml import Experiment as CometExperiment
    _COMET = True
except Exception:
    CometExperiment = None
    _COMET = False


def init_logging(wandb_project: Optional[str] = None, wandb_name: Optional[str] = None,
                 comet_api_key: Optional[str] = None, comet_project: Optional[str] = None,
                 comet_workspace: Optional[str] = None, comet_name: Optional[str] = None,
                 config: dict = None):
    """
    Returns (wandb_run_or_None, comet_exp_or_None)
    """
    wandb_run = None
    comet_exp = None

    if _COMET and comet_api_key:
        try:
            comet_exp = CometExperiment(api_key=comet_api_key, project_name=comet_project, workspace=comet_workspace)
            if comet_name:
                comet_exp.set_name(comet_name)
            if config:
                comet_exp.log_parameters(config)
        except Exception as e:
            print(f"[WARN] Failed to init Comet: {e}")
            comet_exp = None

    if _WANDB and wandb_project:
        try:
            wandb_run = wandb.init(project=wandb_project, name=wandb_name, config=config, reinit=True)
        except Exception as e:
            print(f"[WARN] Failed to init W&B: {e}")
            wandb_run = None

    return wandb_run, comet_exp


def compute_grad_norm(model) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total_norm += float(param_norm.item()) ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def plot_spectrogram(wave: np.ndarray, sr: int, n_fft=1024, hop_length=256, n_mels=80):
    S = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(S_db, origin="lower", aspect="auto")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def _to_wav_bytes(wave_np: np.ndarray, sr: int) -> bytes:
    w = wave_np
    if w.dtype != np.float32:
        w = w.astype(np.float32)
    int16 = (w * 32767).astype('int16')
    bio = io.BytesIO()
    sf.write(bio, int16, sr, format='WAV', subtype='PCM_16')
    bio.seek(0)
    return bio.read()


def log_example(wandb_run, comet_exp, utt_id: str, waveform_np: Optional[np.ndarray], sr: int,
                ref_text: str, pred_text: str, wer: float, cer: float, step: Optional[int] = None):
    summary = f"REF: {ref_text} | PRED: {pred_text} | WER={wer:.3f} CER={cer:.3f}"

    if _WANDB and wandb_run is not None and waveform_np is not None:
        try:
            log_data = {}
            log_data[f"audio_examples/{utt_id}"] = wandb.Audio(
                waveform_np, 
                caption=summary,
                sample_rate=sr
            )
            
            fig = plot_spectrogram(waveform_np, sr)
            log_data[f"spec_examples/{utt_id}"] = wandb.Image(
                fig, 
                caption=summary
            )
            plt.close(fig)
            wandb_run.log(log_data, step=step)

        except Exception as e:
            print(f"[WARN] WandB logging failed for {utt_id}: {e}")

    if _COMET and comet_exp is not None and waveform_np is not None:
        try:
            wav_bytes = _to_wav_bytes(waveform_np, sr)
            comet_exp.log_audio(wav_bytes, file_name=f"{utt_id}.wav", step=step, sample_rate=sr)
            fig = plot_spectrogram(waveform_np, sr)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            comet_exp.log_image(buf, name=f"spec_{utt_id}.png")
            plt.close(fig)
            comet_exp.log_text(summary)
        except Exception as e:
            print(f"[WARN] Comet logging failed for {utt_id}: {e}")
