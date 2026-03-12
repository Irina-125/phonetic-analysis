# phonetic-analysis
# 🎙️ Speaker Dataset Builder

Автоматическая pipeline для предобработки речи, разделения дикторов и формирования размеченного датасета из видео- и аудиофайлов.

Проект реализует два независимых режима работы: **Production Pipeline** — для быстрого формирования датасета дикторов, и **Research Pipeline** — для получения транскрипта с атрибуцией реплик.

---

## 📋 Содержание

- [Обзор архитектуры](#-обзор-архитектуры)
- [Структура проекта](#-структура-проекта)
- [Требования](#-требования)
- [Установка](#-установка)
- [Настройка FFmpeg](#-настройка-ffmpeg)
- [Конфигурация](#-конфигурация)
- [Использование](#-использование)
  - [Production Pipeline](#production-pipeline-mainpy)
  - [Research Pipeline](#research-pipeline-research_pipelinepy)
  - [Конвертация аудио](#конвертация-аудио-convert_audiopy)
- [Описание модулей](#-описание-модулей)
- [Используемые модели](#-используемые-модели)
- [Выходные данные](#-выходные-данные)
- [Известные ограничения](#-известные-ограничения)

---

## 🏗 Обзор архитектуры

Проект содержит два независимых pipeline:

### Production Pipeline

Формирует размеченный датасет 5-секундных аудиосегментов с идентификаторами дикторов — без транскрипции.

```
Видео/Аудио
    │
    ▼
FFmpeg → WAV 16kHz stereo          # audio_pipeline.py → extract_audio()
    │
    ▼
AudioSplitter → чанки по 300 с     # audio_splitter.py → split_audio()
    │
    ▼
Demucs htdemucs → вокал            # audio_pipeline.py → separate_speech()
    │  (split=True, overlap=0.25)
    ▼
Energy Filter → отброс тишины      # audio_pipeline.py → is_noise_energy()
    │  (порог: mean(x²) < 0.0005)
    ▼
Silero VAD → временны́е метки       # audio_pipeline.py → run_vad()
    │
    ▼
Noise Classifier → отброс шума     # noise_classifier.py → is_noise()
    │  (SpeechBrain audioset)
    ▼
ECAPA-TDNN → 192-dim эмбеддинги    # embeddings.py → get_embedding()
    │
    ▼
K-Means + Silhouette → метки       # clustering.py → auto_cluster()
    │  (перебор k от 2 до 8)
    ▼
Нарезка по 5 сек + CSV             # main.py
    │
    ▼
output/*.wav  +  metadata.csv  +  speaker_report.csv
```

### Research Pipeline

Строит текстовый транскрипт с разметкой «кто говорил» и нарезает аудио по дикторам.

```
audio.wav
    │
    ├──▶ Whisper large-v3 → сегменты текста + тайминги
    │
    ├──▶ pyannote/speaker-diarization-3.1 → повороты дикторов
    │
    ▼
Слияние: find_speaker(time) → «SPEAKER_XX: текст»
    │
    ▼
transcript.txt  +  output/<speaker>_<time>s.wav
```

---

## 📁 Структура проекта

```
.
├── main.py                  # Точка входа Production Pipeline
├── research_pipeline.py     # Точка входа Research Pipeline
│
├── audio_pipeline.py        # FFmpeg-извлечение, Demucs, VAD, Energy Filter
├── audio_splitter.py        # Разбиение WAV на чанки по N секунд
├── embeddings.py            # ECAPA-TDNN: извлечение 192-dim эмбеддингов
├── clustering.py            # K-Means + Silhouette: автовыбор числа дикторов
├── noise_classifier.py      # SpeechBrain audioset: фильтр нежелательных звуков
├── production_classifier.py # Альтернативный шумоклассификатор (class-based)
├── segment_speaker.py       # Нарезка WAV по сегментам диаризации (pydub)
├── report.py                # Генерация speaker_report.csv
├── convert_audio.py         # Утилита конвертации любого аудио → WAV 16кГц
├── ffmp.py                  # Настройка путей FFmpeg (Windows)
│
├── config.py                # Все гиперпараметры и константы
├── requirements.txt         # Зависимости pip
│
└── output/                  # Создаётся автоматически
    ├── <video>_spk0_0.00-5.00.wav
    ├── <video>_spk1_12.50-17.50.wav
    ├── <video>.csv
    └── speaker_report.csv
```

---

## ⚙️ Требования

| Компонент | Версия |
|-----------|--------|
| Python | 3.10+ |
| PyTorch | 2.1.2 |
| torchaudio | 2.1.2 |
| CUDA | Не обязателен (CPU-режим по умолчанию) |
| FFmpeg | 6.0+ |
| ОЗУ | ≥ 8 ГБ (рекомендуется 16 ГБ для длинных файлов) |

> **Примечание.** Demucs в CPU-режиме требует значительного времени (~1–3 мин на 5 мин аудио). Для ускорения установите `DEVICE = "cuda"` в `config.py` при наличии GPU.

---

## 🚀 Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/<your-username>/speaker-dataset-builder.git
cd speaker-dataset-builder

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Установить pydub (нужен для research_pipeline и segment_speaker)
pip install pydub

# 5. Установить faster-whisper и pyannote (только для Research Pipeline)
pip install faster-whisper
pip install pyannote.audio
```

### HuggingFace токен (только для Research Pipeline)

Модель `pyannote/speaker-diarization-3.1` требует принятия условий использования и токена HuggingFace:

1. Перейдите на [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) и примите условия.
2. Создайте токен на [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Вставьте токен в `research_pipeline.py`:

```python
HF_TOKEN = "hf_ВАШ_ТОКЕН_ЗДЕСЬ"
```

---

## 🔧 Настройка FFmpeg

### Linux / macOS

FFmpeg, как правило, доступен в системном PATH. Проверьте:

```bash
ffmpeg -version
```

Если не установлен:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg
```

### Windows

1. Скачайте FFmpeg с [ffmpeg.org/download.html](https://ffmpeg.org/download.html) (вариант `essentials_build`).
2. Распакуйте архив, например, в `C:\ffmpeg\`.
3. Откройте `ffmp.py` и укажите актуальный путь:

```python
ffmpeg_base = r"C:\ffmpeg\bin"
```

4. Импортируйте модуль в начале вашего скрипта:

```python
import ffmp  # должен быть первым импортом
```

Либо добавьте папку `bin` в системный PATH, и тогда `ffmp.py` не понадобится.

---

## ⚙️ Конфигурация

Все параметры собраны в `config.py`:

```python
DEVICE = "cpu"               # "cpu" или "cuda"
SAMPLE_RATE = 16000          # Гц — стандарт для всех речевых моделей

SEGMENT_LENGTH = 5           # Длина выходного сегмента датасета, секунды
CHUNK_SIZE = 30              # (резерв) размер чанка, секунды
MIN_SEGMENT_SEC = 0.7        # Сегменты короче этого значения пропускаются

MAX_SPEAKERS = 20            # Верхняя граница перебора k в кластеризации
MIN_SPEAKERS = 1             # Нижняя граница
SILHOUETTE_MIN_SEGMENTS = 3  # Минимум сегментов для запуска Silhouette

NOISE_KEYWORDS = [           # Метки audioset → сегмент считается шумом
    "applause", "laughter",
    "music", "crowd", "cheering"
]
NOISE_PROB_THRESHOLD = 0.55  # Порог уверенности шумоклассификатора
```

---

## 🖥 Использование

### Production Pipeline (`main.py`)

Запускается с видеофайлом в качестве аргумента:

```bash
python main.py video.mp4
```

Поддерживаются любые форматы, которые умеет читать FFmpeg: `.mp4`, `.mkv`, `.avi`, `.mov`, `.mp3`, `.m4a`, `.flac` и т. д.

**Что происходит пошагово:**

1. FFmpeg извлекает аудио → `audio.wav` (16 кГц, stereo).
2. `AudioSplitter` нарезает файл на чанки по 300 секунд.
3. Для каждого чанка Demucs выделяет вокальную дорожку.
4. Silero VAD находит временны́е метки речи.
5. Для каждого сегмента ECAPA-TDNN извлекает 192-мерный эмбеддинг.
6. `auto_cluster()` подбирает оптимальное число дикторов по Silhouette Score.
7. Каждый сегмент нарезается на куски по 5 секунд и сохраняется с меткой диктора.

**Выходные файлы:**

```
output/
├── video.mp4_spk0_0.00-5.00.wav      # Диктор 0, секунды 0–5
├── video.mp4_spk0_5.00-10.00.wav
├── video.mp4_spk1_23.40-28.40.wav    # Диктор 1
├── video.mp4.csv                      # Метаданные всех сегментов
└── speaker_report.csv                 # Суммарное время каждого диктора
```

**Формат `metadata.csv`:**

| file | speaker | start | end |
|------|---------|-------|-----|
| video.mp4_spk0_0.00-5.00.wav | 0 | 0.0 | 5.0 |
| video.mp4_spk1_23.40-28.40.wav | 1 | 23.4 | 28.4 |

---

### Research Pipeline (`research_pipeline.py`)

Принимает готовый файл `audio.wav` (положите его рядом со скриптом или укажите путь в переменной `AUDIO_FILE`):

```bash
python research_pipeline.py
```

> Перед запуском убедитесь, что `audio.wav` существует. Для конвертации используйте `convert_audio.py` (см. ниже).

**Что происходит:**

1. Whisper `large-v3` (int8, CPU) транскрибирует аудио с пословными таймингами.
2. `pyannote/speaker-diarization-3.1` определяет повороты дикторов.
3. Каждой реплике Whisper назначается метка диктора из диаризации.
4. Итоговый транскрипт записывается в `transcript.txt`.
5. `segment_speaker()` нарезает WAV по отрезкам диаризации на 5-секундные куски.

**Выходные файлы:**

```
transcript.txt                          # SPEAKER_00: текст реплики...
output/
├── SPEAKER_00_0s.wav
├── SPEAKER_00_5s.wav
├── SPEAKER_01_18s.wav
└── ...
```

---

### Конвертация аудио (`convert_audio.py`)

Утилита для приведения любого аудиофайла к стандарту pipeline (моно, 16 кГц, WAV):

```bash
python convert_audio.py input.mp3
python convert_audio.py input.flac output_name.wav   # с явным именем выхода
```

По умолчанию сохраняет результат как `audio.wav` в текущей директории.

---

## 📦 Описание модулей

### `audio_pipeline.py`

Основной модуль предобработки. Содержит три ключевые функции:

- **`extract_audio(video_path, wav_out)`** — вызывает FFmpeg через `subprocess` с параметрами `-ac 2 -ar 16000 -vn` для получения стерео WAV.
- **`separate_speech(wav_path)`** — загружает WAV как тензор `[1, 2, T]`, добавляет паддинг кратный 1024, запускает Demucs `apply_model()` и возвращает моно-вокал `sources[0, 3].mean(0)`.
- **`run_vad(wav, sr)`** — запускает Silero VAD и возвращает список `[{"start": int, "end": int}, ...]` в сэмплах.
- **`is_noise_energy(segment, threshold=0.0005)`** — отбрасывает сегменты с энергией `mean(x²)` ниже порога.

### `audio_splitter.py`

- **`split_audio(wav_path, chunk_sec=300)`** — читает WAV через `soundfile`, нарезает на чанки по `chunk_sec * sr` сэмплов. Чанки короче 1 секунды пропускаются. Возвращает `(list[np.ndarray], sr)`.

### `embeddings.py`

- Загружает `speechbrain/spkrec-ecapa-voxceleb` при импорте модуля.
- **`get_embedding(segment)`** — принимает тензор `[1, T]`, возвращает numpy-вектор `[192]` через `encoder.encode_batch()`.

### `clustering.py`

- **`auto_cluster(embeddings, max_speakers=8)`** — стакует эмбеддинги в матрицу `[N, 192]`. При `N < 3` возвращает нулевые метки. Для каждого `k` от 2 до `min(max_speakers, N-1)` обучает `KMeans(n_init=10)` и вычисляет Silhouette Score. Возвращает метки с максимальным скором. При неудаче — все нули.

> В коде также присутствует закомментированная альтернатива на `AgglomerativeClustering` с `linkage="average"`.

### `noise_classifier.py`

- Загружает `speechbrain/audioset-classifier` при импорте модуля.
- **`is_noise(segment)`** — если сегмент короче 1 секунды, считает его шумом. Иначе запускает классификацию по AudioSet. Возвращает `True`, если предсказанная метка содержит ключевое слово из `NOISE_KEYWORDS` с уверенностью выше `NOISE_PROB_THRESHOLD`.

### `segment_speaker.py`

- **`segment_speaker(wav_file, diarization_segments, output_dir, segment_length=5)`** — принимает список `{"start", "end", "speaker"}`, нарезает WAV через `pydub.AudioSegment` на куски по `segment_length` секунд. Фрагменты короче 1 секунды пропускаются. Имя файла: `<speaker>_<start>s.wav`.

### `report.py`

- **`generate_report(metadata_path)`** — читает CSV метаданных, группирует по `speaker`, суммирует `(end - start)` и сохраняет в `output/speaker_report.csv`.

---

## 🤖 Используемые модели

| Модель | Источник | Назначение |
|--------|----------|------------|
| `htdemucs` | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | Разделение речи от фона/музыки |
| `silero-vad` v5.1 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | Детекция временны́х меток речи |
| `speechbrain/spkrec-ecapa-voxceleb` | HuggingFace | 192-dim эмбеддинги дикторов |
| `speechbrain/audioset-classifier` | HuggingFace | Классификация шума / нежелательных звуков |
| `whisper large-v3` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | ASR транскрипция (int8, CPU) |
| `pyannote/speaker-diarization-3.1` | HuggingFace | Диаризация дикторов |

Все модели загружаются автоматически при первом запуске и кешируются локально.

---

## 📊 Выходные данные

### `output/speaker_report.csv`

```
speaker,total_speech_seconds
0,142.5
1,98.3
2,67.1
```

### `output/<video>.csv`

```
file,speaker,start,end
video.mp4_spk0_0.00-5.00.wav,0,0.0,5.0
video.mp4_spk0_5.00-10.00.wav,0,5.0,10.0
video.mp4_spk1_23.40-28.40.wav,1,23.4,28.4
```

### `transcript.txt` (Research Pipeline)

```
SPEAKER_00: Добрый день, как я могу вам помочь?
SPEAKER_01: Здравствуйте, у меня вопрос по вашему продукту.
SPEAKER_00: Конечно, слушаю вас.
Unknown: ...
```

---

## 🧠 DialectAI — Модель классификации диалектов (`dialect_v3___копия.ipynb`)

Jupyter-ноутбук реализует полный цикл обучения нейросети для **классификации диалекта и профессии говорящего** по аудиозаписи. Это production-ready код с задокументированными исправлениями 20 багов и 9 аудит-правок.

---

### Назначение

Модель принимает на вход WAV-файл (до 5 секунд) и предсказывает:
- **Регион происхождения** говорящего (по числу уникальных регионов в датасете)
- **Профессиональную принадлежность** (по числу уникальных профессий в датасете)

Дополнительно — кластеризует все эмбеддинги датасета, строит карту диалектов и сохраняет её в `dialect_map.png`.

---

### Архитектура модели `DialectModel`

```
audio (B, T_samples)  →  float32, 16kHz, длина = 5 * 16000 = 80000
        │
        ▼
HubertEncoder                          # facebook/hubert-large-ls960-ft
  ├─ AutoFeatureExtractor              # нормализация входа (исправл. BUG-03)
  ├─ HubertModel (frozen CNN)          # 768M параметров, gradient checkpointing
  └─ → (B, T_frames, 1024)            # ~50 фреймов на секунду
        │
        ▼
SpeakerNormalizer                      # Instance Norm по временной оси (исправл. BUG-01)
  ├─ InstanceNorm1d(1024)              # убирает speaker-specific mean/std
  └─ (опц.) вычитание ECAPA проекции  # (B, 192) → (B, 1024) speaker bias
        │
        ▼
PhoneticTransformer                    # контекстное моделирование фонем
  ├─ TransformerEncoderLayer × 4       # batch_first=True (исправл. BUG-06)
  ├─ Pre-LayerNorm, 8 голов, FFN×4    # стабильнее на малых данных
  └─ → (B, T_frames, 1024)
        │
        ▼
AttentionPooling                       # взвешенное усреднение (исправл. BUG-13)
  └─ → (B, 1024)
        │
        ▼
Linear(1024→256) + BatchNorm1d        # проекция эмбеддинга
  └─ L2 normalize → (B, 256)
        │
        ├──▶ ArcFaceLoss(256, n_regions)  # Additive Angular Margin (исправл. BUG-04/05)
        │    m=0.5, s=64                   # train: с margin / inference: без
        │    → (B, n_regions) logits
        │
        └──▶ Linear(256, n_jobs)           # профессия
             → (B, n_jobs) logits
```

**Функция потерь:**

```
L = CrossEntropy(region) + CrossEntropy(job) + 0.3 × TripletMarginLoss
```

Triplet loss использует **Online Hard Mining** — самые сложные пары в батче.

---

### Структура датасета

```
dataset/
├── Москва/
│   ├── engineer_001.wav
│   ├── teacher_002.wav
│   └── ...
├── Санкт-Петербург/
│   └── ...
└── <регион>/
    └── <профессия>_NNN.wav
```

- Папки первого уровня — **регионы** (метки классификации)
- Имена файлов: `<профессия>_<номер>.wav` — профессия извлекается из имени файла до первого `_`
- Все WAV обрезаются / паддятся до `MAX_SAMPLES = 5 × 16000 = 80 000` сэмплов

---

### Порядок запуска ноутбука

```
1. Ячейка "ШАГ 1"  →  установка зависимостей
2. Kernel → Restart Kernel
3. Ячейка "ШАГ 2"  →  проверка версий (все ✅ обязательно)
4. Далее — ячейки по порядку без пропусков
```

**Зависимости ноутбука** (отличаются от `requirements.txt` основного pipeline):

| Пакет | Версия |
|-------|--------|
| numpy | 1.26.4 |
| torch | 2.2.2 |
| torchaudio | 2.2.2 |
| transformers | 4.39.3 |
| librosa | 0.10.2 |
| soundfile | 0.12.1 |
| speechbrain | 1.0.0 |
| scikit-learn | 1.4.2 |
| umap-learn | 0.5.6 |
| hdbscan | 0.8.38 |
| matplotlib | 3.8.4 |

> ⚠️ Версии зафиксированы намеренно — SpeechBrain 1.0 сломал обратную совместимость с 0.5.x. Не смешивайте с `requirements.txt` основного pipeline.

---

### Артефакты обучения

Все результаты сохраняются в `runs/dialect_v1/` (путь задаётся константой `OUT_DIR`):

| Файл | Содержимое |
|------|------------|
| `best_model.pt` | Веса модели (лучший checkpoint по val_loss) |
| `region_enc.pkl` | LabelEncoder для регионов |
| `job_enc.pkl` | LabelEncoder для профессий |
| `meta.pkl` | `n_regions`, `n_jobs` |
| `umap_20d.pkl` | UMAP-редуктор 20D (для кластеризации) |
| `umap_2d.pkl` | UMAP-редуктор 2D (для визуализации) |
| `dialect_clusters.csv` | Кластеры + координаты UMAP для каждого файла |
| `dialect_map.png` | Карта диалектов (HDBSCAN + ground truth side-by-side) |
| `training_curves.png` | Графики loss и accuracy по эпохам |

---

### Кластеризация диалектов

После обучения ноутбук автоматически кластеризует все эмбеддинги датасета:

```
256-dim embeddings
      │
      ▼
UMAP (n_components=20, metric=cosine)   ← для HDBSCAN (исправл. BUG-12)
      │
      ▼
HDBSCAN (min_cluster_size=10)           ← шум помечается как -1
      │
      ▼
UMAP (n_components=2)                   ← только для визуализации (исправл. BUG-12)
      │
      ▼
dialect_map.png  +  dialect_clusters.csv
```

> **BUG-12:** В исходном коде `n_components=2` использовался и для HDBSCAN, и для визуализации. 2D UMAP теряет слишком много информации — кластеры получались плохими. Исправлено: 20D для HDBSCAN, отдельный 2D-редуктор только для графика.

---

### Inference

```python
from dialect_v3 import Predictor, load_artifacts, load_trained_model
from pathlib import Path

OUT_DIR = Path("runs/dialect_v1")

region_enc, job_enc, meta = load_artifacts(OUT_DIR)
model = load_trained_model(OUT_DIR, meta)
predictor = Predictor(model, region_enc, job_enc)

result = predictor.predict("path/to/audio.wav")
# result = {
#   "top_region":   "Москва",
#   "top_job":      "engineer",
#   "region_probs": {"Москва": 0.812, "Санкт-Петербург": 0.134, ...},
#   "job_probs":    {"engineer": 0.741, "teacher": 0.201, ...},
#   "embedding":    np.ndarray (1, 256)
# }
```

Вывод в консоль содержит bar-chart вероятностей для каждого класса:

```
====================================================
📍 Регион:
   Москва                 ████████████████████████   0.812
   Санкт-Петербург        ████                       0.134
💼 Профессия:
   engineer               █████████████████████      0.741
   teacher                █████                      0.201
====================================================
```

---

### Исправленные баги

В ноутбуке задокументированы и исправлены **20 багов** и **9 аудит-правок**:

| Категория | Количество | Примеры |
|-----------|-----------|---------|
| Критические (падение кода) | 8 | BUG-02, 03, 07, 08, 13, 15 — NameError, отсутствие наследования nn.Module |
| Логические ошибки обучения | 7 | BUG-01, 04, 05, 06, 09, 10, 12 — неправильный ArcFace, накопление градиентов, плохая кластеризация |
| Отсутствующий функционал | 5 | BUG-11, 16, 17, 18, 19 — нет device handling, LabelEncoder, Dataset, val-loop |
| Аудит API / безопасность | 9 | AUD-01..15 — устаревший SpeechBrain API, weights_only=True, guard на collapsed embeddings |

---

## ⚠️ Известные ограничения

- **Скорость на CPU.** Demucs — наиболее тяжёлая операция. Для файлов длиннее 30 минут рекомендуется GPU или предварительная нарезка.
- **HF-токен в коде.** В `research_pipeline.py` токен HuggingFace хранится в переменной. Перед публикацией вынесите его в переменную окружения (`os.environ["HF_TOKEN"]`) или `.env`-файл.
- **Пути FFmpeg захардкожены** в `ffmp.py` под конкретную Windows-машину — скорректируйте перед использованием.
- **Качество кластеризации** зависит от числа и длины сегментов. При менее чем 3 сегментах все реплики будут приписаны одному диктору.
- **Research Pipeline** не интегрирован с Production Pipeline — они работают независимо и используют разные стратегии сегментации дикторов.

---

## 📄 Лицензия

MIT License — свободное использование, модификация и распространение с указанием авторства.
