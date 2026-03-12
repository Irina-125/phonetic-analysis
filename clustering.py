import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from config import *

#def auto_cluster(embeddings):

    #X = np.stack(embeddings)

    #if len(X) < SILHOUETTE_MIN_SEGMENTS:
        #return np.zeros(len(X))

    #best_score = -1
    #best_labels = None

    #for k in range(MIN_SPEAKERS, min(MAX_SPEAKERS, len(X)) + 1):

    #   clustering = AgglomerativeClustering(
    #        n_clusters=k,
    #        linkage="average"
    #    )

    #    labels = clustering.fit_predict(X)

    #    if len(set(labels)) < 2:
    #        continue

    #    score = silhouette_score(X, labels)

    #    if score > best_score:
    #        best_score = score
    #        best_labels = labels

#    return best_labels



def auto_cluster(embeddings, max_speakers=8):

    X = np.vstack(embeddings)
    n_samples = len(X)

    # если сегментов мало — один спикер
    if n_samples < 3:
        return np.zeros(n_samples)

    best_score = -1
    best_labels = None

    max_k = min(max_speakers, n_samples - 1)

    for k in range(2, max_k + 1):

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)

        unique_labels = len(set(labels))

        # если каждый сегмент стал отдельным кластером — пропускаем
        if unique_labels >= n_samples:
            continue

        try:
            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_labels = labels

        except ValueError:
            continue

    # fallback если silhouette не смог выбрать
    if best_labels is None:
        return np.zeros(n_samples)

    return best_labels