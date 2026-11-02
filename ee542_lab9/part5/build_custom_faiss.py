"""Build the Part 5 FAISS index from the lecture chunks and saved IDF K64 model."""
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from faiss_adapter import HistogramEmbeddings

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "all_chunks.json"
MODEL = ROOT / "part5" / "models" / "idf_k64"
INDEX = ROOT / "data" / "part5" / "faiss_histogram_index"


def main():
    if not CHUNKS.exists():
        raise FileNotFoundError("Missing lecture chunks: " + str(CHUNKS))
    if not (MODEL / "settings.json").exists():
        raise FileNotFoundError("Missing saved model: " + str(MODEL))

    with open(CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = HistogramEmbeddings(str(MODEL))
    if len(embeddings.embed_query("What are the benefits of RDMA?")) != 64:
        raise ValueError("Expected a 64-dimensional embedding. Check the saved model.")

    texts = [c["text"] for c in chunks]
    metadata = [{"chunk_id": c["chunk_id"], "source": c["source"], "page": c["page"]}
                for c in chunks]
    store = FAISS.from_texts(texts, embeddings, metadatas=metadata)
    INDEX.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX))
    print("Vectors:", store.index.ntotal)
    print("Dimensions:", store.index.d)
    print("Saved:", INDEX)
    for doc in store.similarity_search("What are the benefits of RDMA?", k=4):
        print(doc.metadata["chunk_id"], doc.metadata["source"], doc.metadata["page"])


if __name__ == "__main__":
    main()
