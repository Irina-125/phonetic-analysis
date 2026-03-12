from speechbrain.pretrained import EncoderClassifier
import torch
from config import *

class NoiseClassifier:

    def __init__(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/audioset-classifier",
            run_opts={"device": DEVICE}
        )

    def is_noise(self, segment):

        if segment.shape[1] < SAMPLE_RATE:
            return True

        with torch.no_grad():
            out_prob, score, index, text_lab = self.model.classify_batch(segment)

        label = text_lab[0]
        prob = score[0].item()

        noise_keywords = ["applause", "laughter", "music", "crowd"]

        if any(word in label.lower() for word in noise_keywords):
            if prob > 0.5:
                return True

        return False