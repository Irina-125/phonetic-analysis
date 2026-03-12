import os
import soundfile as sf
from pydub import AudioSegment

def segment_speaker(wav_file="audio.wav", diarization_segments=None, output_dir="output", segment_length=5):
    """
    wav_file: путь к аудио (16kHz, mono)
    diarization_segments: список словарей {"start": float, "end": float, "speaker": str}
    segment_length: длина фрагмента в секундах
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio = AudioSegment.from_wav(wav_file)

    for seg in diarization_segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        speaker_audio = audio[start_ms:end_ms]

        # Нарезаем на куски по segment_length секунд
        for i, chunk_start in enumerate(range(0, len(speaker_audio), segment_length*1000)):
            chunk = speaker_audio[chunk_start:chunk_start + segment_length*1000]

            if len(chunk) < 1000:  # игнорируем слишком короткие куски
                continue

            filename = f"{seg['speaker']}_{int(seg['start'] + chunk_start/1000)}s.wav"
            filepath = os.path.join(output_dir, filename)
            chunk.export(filepath, format="wav")
            print(f"Saved: {filepath}")