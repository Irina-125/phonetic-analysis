import soundfile as sf
import numpy as np

def split_audio(wav_path, chunk_sec=300):

    audio, sr = sf.read(wav_path)

    chunk_samples = chunk_sec * sr

    chunks = []

    for i in range(0, len(audio), chunk_samples):

        chunk = audio[i:i+chunk_samples]

        if len(chunk) < sr:
            continue

        chunks.append(chunk)

    return chunks, sr