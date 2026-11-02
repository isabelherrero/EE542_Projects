
import json
import os
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


ROOT = Path(__file__).resolve().parents[1]

# Part 4: Lecture 7 + Lecture 9
CHUNKS_FILE = ROOT / "data" / "all_chunks.json"

# Separate from all Part 3 indexes.
INDEX_DIR = ROOT / "data" / "part4" / "faiss_openai_index"

# Reuse our OpenAI API key.
load_dotenv(ROOT / "part3" / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing.")

chunks = json.loads(
    CHUNKS_FILE.read_text(encoding="utf-8")
)

texts = [chunk["text"] for chunk in chunks]

metadata = [
    {
        "chunk_id": chunk["chunk_id"],
        "source": chunk["source"],
        "page": chunk["page"],
    }
    for chunk in chunks
]

print("=== Part 4: Cloud Computing Study Assistant ===")
print(f"Total text chunks: {len(chunks)}")

source_counts = Counter(
    chunk["source"] for chunk in chunks
)

for source, count in source_counts.items():
    print(f"{source}: {count} chunks")

# OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

print("\nGenerating OpenAI embeddings...")

vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadata,
)

INDEX_DIR.mkdir(parents=True, exist_ok=True)

vectorstore.save_local(str(INDEX_DIR))

print(f"FAISS vectors stored: {vectorstore.index.ntotal}")
print(f"FAISS dimensions: {vectorstore.index.d}")
print(f"FAISS index saved: {INDEX_DIR}")

# Verify retrieval from both lecture PDFs.
test_questions = [
    "What are the benefits of RDMA?",
    "What is Amazon EC2?",
]

for question in test_questions:
    print(f"\nQuestion: {question}")

    results = vectorstore.similarity_search(
        question, k=3
    )

    for i, doc in enumerate(results, start=1):
        print(
            f"Result {i}: "
            f"{doc.metadata['source']}, "
            f"page {doc.metadata['page']}"
        )
        print(f"Text: {doc.page_content[:160]}")
