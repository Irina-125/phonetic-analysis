from speechbrain.pretrained import EncoderClassifier
from config import *
import torch

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/audioset-classifier",
    run_opts={"device": DEVICE}
)

def is_noise(segment):

    if segment.shape[1] < SAMPLE_RATE:
        return True

    with torch.no_grad():
        out_prob, score, index, text_lab = classifier.classify_batch(segment)

    label = text_lab[0].lower()
    prob = score[0].item()

    if any(k in label for k in NOISE_KEYWORDS) and prob > NOISE_PROB_THRESHOLD:
        return True

    return False