"""Run Part 5 training, indexing, search, and evaluation from the terminal."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from custom_embedding import train, histogram, save_model, load_model, search


def read_lines(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def read_chunks(path):
    with open(path, encoding="utf-8") as file:
        chunks = json.load(file)

    for chunk in chunks:
        chunk["id"] = chunk["chunk_id"]

    return chunks

def read_tests(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "inspect", "index", "search", "evaluate"])
    parser.add_argument("--model-dir", default="models/demo")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--stopwords", action="store_true")
    parser.add_argument("--weighting", choices=["count", "idf"], default="count")
    parser.add_argument("--question", default="What are the benefits of RDMA?")
    args = parser.parse_args()
    folder = Path(args.model_dir)

    chunks_file = "data/all_chunks.json"
    text = "RDMA provides low latency and high throughput"
    top = 4
    tests = "data/cloud_questions_final.json"
    
    if args.action == "train":
        chunks = read_chunks(chunks_file)
        model, centers, clusters, idf, settings = train(
            chunks, args.n, args.k, args.stopwords, args.weighting)
        save_model(folder, model, centers, clusters, idf, settings)
        print("Chunks:", len(chunks), "Words:", len(clusters))
        print("Word vectors:", model.wv.vectors.shape, "Centroids:", centers.shape)
        return

    clusters, idf, settings = load_model(folder)
    if args.action == "inspect":
        vec = histogram(text, clusters, idf, settings)
        print("Text:", text)
        print("Word clusters:", [(w, clusters.get(w)) for w in text.lower().split()])
        print("Histogram:", np.round(vec, 4).tolist(), "Sum:", round(vec.sum(), 4))
        print("Centroids:", np.load(folder / "centroids.npy").shape)
        return

    if args.action == "index":
        chunks = read_chunks(chunks_file)
        vectors = np.array([histogram(c["text"], clusters, idf, settings)
                            for c in chunks], dtype=float)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", vectors)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(chunk) + "\n")
        print("Indexed", len(chunks), "chunks; matrix shape:", vectors.shape)
        return

    chunks = read_lines(folder / "chunks.jsonl")
    vectors = np.load(folder / "vectors.npy")

    if args.action == "search":
        results = search(args.question, chunks, vectors, clusters, idf, settings, top)
        for rank, (chunk, score) in enumerate(results, 1):
            print("%d. %s (similarity %.3f)" % (rank, chunk["id"], score))
            print(chunk["text"][:300], "\n")
        return

    tests = read_tests(tests)
    rows = []
    for test in tests:
        results = search(test["question"], chunks, vectors,
                         clusters, idf, settings, top)
        ids = [item["id"] for item, _ in results]
        rank = next((i for i, item in enumerate(ids, 1)
                     if item in test["relevant_ids"]), 0)
        rows.append({"question": test["question"], "hit": int(rank > 0),
                     "reciprocal_rank": 1/rank if rank else 0, "rank": rank})
    with open("results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Hit@%d: %.3f" % (top, np.mean([r["hit"] for r in rows])))
    print("MRR@%d: %.3f" % (top, np.mean([r["reciprocal_rank"] for r in rows])))

if __name__ == "__main__":
    main()
