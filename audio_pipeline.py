import subprocess
import torch
import torchaudio
import soundfile as sf
import numpy as np

from demucs.pretrained import get_model
from demucs.apply import apply_model
from silero_vad import load_silero_vad, get_speech_timestamps

# -----------------------------------
# CONFIG
# -----------------------------------

SAMPLE_RATE = 16000

# -----------------------------------
# LOAD MODELS
# -----------------------------------

print("Loading Silero VAD...")
vad_model = load_silero_vad()

print("Loading Demucs...")
demucs_model = get_model("htdemucs")
demucs_model.cpu()
demucs_model.eval()


# -----------------------------------
# AUDIO EXTRACTION
# -----------------------------------

def extract_audio(video_path, wav_out):

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "2",
        "-ar", str(SAMPLE_RATE),
        "-vn",
        wav_out
    ], check=True)


# -----------------------------------
# DEMUCS SPEECH SEPARATION
# -----------------------------------

def separate_speech(wav_path):

    print("Loading audio...")

    wav, sr = sf.read(wav_path)

    wav = torch.tensor(wav).float()

    # если mono → делаем stereo
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(1).repeat(1, 2)

    # shape → [channels, time]
    wav = wav.T

    wav = wav.unsqueeze(0)

    length = wav.shape[-1]

    # padding для demucs
    target_multiple = 1024
    pad = (target_multiple - (length % target_multiple)) % target_multiple

    if pad > 0:
        wav = torch.nn.functional.pad(wav, (0, pad))

    print("Running Demucs...")

    with torch.no_grad():

        sources = apply_model(
            demucs_model,
            wav,
            device="cpu",
            split=True,
            overlap=0.25,
            progress=True
        )

    vocals = sources[0, 3]

    vocals = vocals[..., :length]

    vocals = vocals.mean(0)

    return vocals, sr


# -----------------------------------
# ENERGY FILTER
# -----------------------------------

def is_noise_energy(segment, threshold=0.0005):

    energy = torch.mean(segment ** 2)

    return energy < threshold


# -----------------------------------
# VAD
# -----------------------------------

def run_vad(wav, sr):

    print("Running Silero VAD...")

    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sr
    )

    return timestamps


# -----------------------------------
# SAVE SEGMENTS
# -----------------------------------

def save_segments(wav, sr, segments, out_dir):

    import os

    os.makedirs(out_dir, exist_ok=True)

    count = 0

    for seg in segments:

        start = seg["start"]
        end = seg["end"]

        segment = wav[start:end]

        if is_noise_energy(segment):
            continue

        filename = os.path.join(out_dir, f"segment_{count}.wav")

        sf.write(filename, segment.numpy(), sr)

        count += 1

    print(f"Saved {count} segments")

    return count