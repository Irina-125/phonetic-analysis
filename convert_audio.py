from pydub import AudioSegment
import sys
import os

def convert_to_wav(input_file, output_file="audio.wav"):
    print("Loading audio...")
    audio = AudioSegment.from_file(input_file)

    print("Converting to mono 16kHz...")
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    audio.export(output_file, format="wav")

    print("Saved:", output_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_audio.py input.mp3")
        sys.exit()

    convert_to_wav(sys.argv[1])