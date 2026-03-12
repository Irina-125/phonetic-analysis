import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from segment_speaker import segment_speaker

AUDIO_FILE = "audio.wav"

print("Loading Whisper model...")

model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8"
)

print("Transcribing audio...")

segments, info = model.transcribe(
    AUDIO_FILE,
    beam_size=5,
    word_timestamps=True
)

whisper_segments = []

for segment in segments:
    whisper_segments.append(
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        }
    )

print("Loading diarization model...")

HF_TOKEN = "hf_OUR_TOKEN"

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

print("Running diarization...")

diarization = pipeline(AUDIO_FILE)

speaker_segments = []

for turn, _, speaker in diarization.itertracks(yield_label=True):
    speaker_segments.append(
        {
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        }
    )

def find_speaker(time):
    for seg in speaker_segments:
        if seg["start"] <= time <= seg["end"]:
            return seg["speaker"]
    return "Unknown"

print("Merging transcription + speakers...")

output = []

for seg in whisper_segments:
    speaker = find_speaker(seg["start"])
    line = f"{speaker}: {seg['text']}"
    output.append(line)

print("Saving transcript...")

with open("transcript.txt", "w", encoding="utf-8") as f:
    for line in output:
        f.write(line + "\n")

print("Done. Transcript saved as transcript.txt")

# diarization_segments = [{"start": .., "end": .., "speaker": ..}, ...]
# используем тот же формат, что у нас в pipeline

segment_speaker(
    wav_file="audio.wav",
    diarization_segments=speaker_segments,
    output_dir="output",
    segment_length=5  # по 5 секунд
)