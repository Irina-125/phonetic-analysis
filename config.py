import torch

DEVICE = "cpu"
SAMPLE_RATE = 16000
SEGMENT_LENGTH = 5  # секунды

CHUNK_SIZE =  30 
MIN_SEGMENT_SEC = 0.7

MAX_SPEAKERS = 20
MIN_SPEAKERS = 1

SILHOUETTE_MIN_SEGMENTS = 3

NOISE_KEYWORDS = [
    "applause",
    "laughter",
    "music",
    "crowd",
    "cheering"
]

NOISE_PROB_THRESHOLD = 0.55