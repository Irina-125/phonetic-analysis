import os
from pydub import AudioSegment

# =====================================================
#           FFMPEG CONFIG (обязательный блок)
# =====================================================

# Базовый путь к папке ffmpeg
ffmpeg_base = r"C:/Users/podgo/Downloads/ffmpeg-8.0-essentials_build/ffmpeg-8.0-essentials_build/bin"

ffmpeg_path = os.path.join(ffmpeg_base, "ffmpeg.exe")
ffprobe_path = os.path.join(ffmpeg_base, "ffprobe.exe")

print("\n🔍 Проверка FFmpeg:")

# Проверка ffmpeg
if os.path.exists(ffmpeg_path):
    print(f"  ✅ FFmpeg найден: {ffmpeg_path}")
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
else:
    raise FileNotFoundError(f"❌ FFmpeg НЕ найден: {ffmpeg_path}")

# Проверка ffprobe
if os.path.exists(ffprobe_path):
    print(f"  ✅ FFprobe найден: {ffprobe_path}")
    AudioSegment.ffprobe = ffprobe_path
else:
    raise FileNotFoundError(f"❌ FFprobe НЕ найден: {ffprobe_path}")

# Добавляем в PATH
os.environ["PATH"] += os.pathsep + ffmpeg_base
os.environ["FFMPEG_BINARY"] = ffmpeg_path

print("✅ FFmpeg успешно настроен\n")