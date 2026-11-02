import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
from gensim.models import Word2Vec

STOPWORDS = set("a an and are as at be by for from in is it of on or that the this to was were with".split())


def tokenize(text, stopwords=False):
    words = re.findall(r"[a-z0-9]+", text.lower())
    if stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return words


def kmeans(vectors, k, steps=100):
    """Assign word vectors to the nearest centroid, then update centroids."""
    if k < 1 or k > len(vectors):
        raise ValueError("K must be between 1 and the vocabulary size")
    rng = np.random.RandomState(42)
    centers = vectors[rng.choice(len(vectors), k, replace=False)].copy()

    for _ in range(steps):
        distances = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            group = vectors[labels == j]
            if len(group) > 0:
                new_centers[j] = group.mean(axis=0)
        if np.allclose(centers, new_centers):
            centers = new_centers
            break
        centers = new_centers

    distances = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)
    return centers, labels


def train(chunks, n=100, k=32, stopwords=False, weighting="count"):
    """Train Word2Vec, cluster words, and calculate optional IDF weights."""
    sentences = [tokenize(c["text"], stopwords) for c in chunks]
    sentences = [s for s in sentences if s]
    if not sentences:
        raise ValueError("No text found")

    model = Word2Vec(sentences, vector_size=n, window=5, min_count=1,
                     sg=1, workers=1, seed=42, epochs=30)
    vocab = model.wv.index_to_key
    centers, labels = kmeans(model.wv.vectors, k)
    clusters = {word: int(label) for word, label in zip(vocab, labels)}

    counts = Counter()
    for sentence in sentences:
        counts.update(set(sentence))
    idf = {word: float(np.log((len(sentences) + 1) / (counts[word] + 1)) + 1)
           for word in vocab}
    settings = {"n": n, "k": k, "stopwords": stopwords, "weighting": weighting}
    return model, centers, clusters, idf, settings


def histogram(text, clusters, idf, settings):
    """Count word clusters and normalize to a K-dimensional vector."""
    words = tokenize(text, settings["stopwords"])
    hist = np.zeros(settings["k"], dtype=float)
    if not words:
        return hist

    total = 0.0
    for word in words:
        weight = idf.get(word, 1.0) if settings["weighting"] == "idf" else 1.0
        total += weight
        if word in clusters:
            hist[clusters[word]] += weight
    return hist / total if total else hist


def save_model(folder, model, centers, clusters, idf, settings):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save(str(folder / "word2vec.model"))
    np.save(folder / "centroids.npy", centers)
    for name, value in [("clusters", clusters), ("idf", idf), ("settings", settings)]:
        (folder / (name + ".json")).write_text(json.dumps(value, indent=2))


def load_model(folder):
    folder = Path(folder)
    def read(name):
        return json.loads((folder / (name + ".json")).read_text())
    return read("clusters"), read("idf"), read("settings")


def search(question, chunks, vectors, clusters, idf, settings, top=4):
    """Rank chunks by cosine similarity of their histograms."""
    q = histogram(question, clusters, idf, settings)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * q_norm + 1e-12)
    order = np.argsort(-scores)[:top]
    return [(chunks[i], float(scores[i])) for i in order]
