"""LangChain adapter for the saved Part 5 histogram embedding model."""
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from custom_embedding import histogram, load_model


class HistogramEmbeddings(Embeddings):
    def __init__(self, model_dir):
        self.clusters, self.idf, self.settings = load_model(Path(model_dir))

    def encode(self, text):
        vec = np.asarray(histogram(text, self.clusters, self.idf, self.settings), dtype=float)
        size = np.linalg.norm(vec)
        if size > 0:
            vec = vec / size
        return vec.tolist()

    def embed_documents(self, texts):
        return [self.encode(text) for text in texts]

    def embed_query(self, text):
        return self.encode(text)
