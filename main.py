import os
import sys
import torch
import pandas as pd
import torchaudio
import soundfile as sf

from config import *

from audio_pipeline import (
    extract_audio,
    separate_speech,
    run_vad,
    is_noise_energy
)

from audio_splitter import split_audio
from embeddings import get_embedding
from clustering import auto_cluster
from report import generate_report


def process_video(video_path):

    base = os.path.basename(video_path)

    print("\n==============================")
    print("Processing:", base)
    print("==============================\n")

    os.makedirs("output", exist_ok=True)

    wav_file = "audio.wav"

    # ============================
    # 1. EXTRACT AUDIO
    # ============================

    print("Extracting audio with FFmpeg...")
    extract_audio(video_path, wav_file)

    # ============================
    # 2. SPLIT AUDIO (important!)
    # ============================

    print("Splitting audio into chunks...")

    chunks, sr = split_audio(wav_file)

    speech_all = []

    # ============================
    # 3. DEMUCS PER CHUNK
    # ============================

    for i, chunk in enumerate(chunks):

        print(f"\nProcessing chunk {i+1}/{len(chunks)}")

        temp_file = f"temp_chunk_{i}.wav"

        sf.write(temp_file, chunk, sr)

        speech, sr = separate_speech(temp_file)

        speech_all.append(speech)

        os.remove(temp_file)

    speech = torch.cat(speech_all)

    speech = speech.unsqueeze(0)

    # ============================
    # 4. VAD
    # ============================

    print("\nRunning VAD...")

    timestamps = run_vad(speech.squeeze(), sr)

    segments = []
    embeddings = []

    # ============================
    # 5. EXTRACT EMBEDDINGS
    # ============================

    print("Extracting speaker embeddings...")

    for ts in timestamps:

        s = ts["start"]
        e = ts["end"]

        segment = speech[:, s:e]

        duration = (e - s) / sr

        if duration < MIN_SEGMENT_SEC:
            continue

        if is_noise_energy(segment):
            continue

        emb = get_embedding(segment)

        segments.append((s / sr, e / sr))
        embeddings.append(emb)

    if len(embeddings) == 0:
        print("No speech detected.")
        return

    # ============================
    # 6. CLUSTER SPEAKERS
    # ============================

    print("\nClustering speakers...")

    labels = auto_cluster(embeddings)

    metadata = []

    # ============================
    # 7. CREATE DATASET (5s)
    # ============================

    print("\nSaving 5-second dataset segments...")

    for (start, end), speaker in zip(segments, labels):

        cur = start

        while cur < end:

            nxt = min(cur + SEGMENT_LENGTH, end)

            s_sample = int(cur * sr)
            e_sample = int(nxt * sr)

            chunk = speech[:, s_sample:e_sample]

            filename = f"output/{base}_spk{speaker}_{cur:.2f}-{nxt:.2f}.wav"

            torchaudio.save(
                filename,
                chunk,
                sr
            )

            metadata.append({
                "file": filename,
                "speaker": speaker,
                "start": cur,
                "end": nxt
            })

            cur = nxt

    # ============================
    # 8. SAVE METADATA
    # ============================

    csv_path = f"output/{base}.csv"

    df = pd.DataFrame(metadata)

    df.to_csv(csv_path, index=False)

    print("\nMetadata saved:", csv_path)

    # ============================
    # 9. SPEAKER REPORT
    # ============================

    generate_report(csv_path)

    print("\nPipeline finished successfully.\n")


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print("python main.py video.mp4\n")
        sys.exit()

    process_video(sys.argv[1])