from speechbrain.pretrained import EncoderClassifier
from config import *
import torch

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

def get_embedding(segment):

    if segment.dim() == 1:
        segment = segment.unsqueeze(0)

    with torch.no_grad():
        emb = encoder.encode_batch(segment)

    return emb.squeeze().cpu().numpy()