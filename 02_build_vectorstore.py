"""Build a FAISS vector store from seed documents and SDG-generated QA pairs.

Reads the artifacts produced by 01_extract_knowledge.py (seed_docs.json and
knowledge.csv), chunks and embeds them with sentence-transformers, and
persists a FAISS index alongside metadata for retrieval.
"""

import os
import re
import json
import argparse

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def _split_oversized(text, max_chars):
    """Sub-split a text block that exceeds max_chars by paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", text)
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 < max_chars:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current.strip())
            if len(para) >= max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i : i + max_chars].strip())
            else:
                current = para
                continue
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_text(text, max_chars=2500):
    """Split markdown text into chunks by heading boundaries."""
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    chunks, current_chunk = [], ""
    for section in sections:
        if not section.strip():
            continue
        if len(current_chunk) + len(section) < max_chars:
            current_chunk += section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(section) >= max_chars:
                chunks.extend(_split_oversized(section, max_chars))
                current_chunk = ""
                continue
            current_chunk = section
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [c for c in chunks if c]


def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector store from seed docs and SDG QA pairs."
    )
    parser.add_argument("--knowledge-dir", type=str, default="./knowledge",
                        help="Directory containing seed_docs.json and knowledge.csv (default: ./knowledge)")
    parser.add_argument("--output", type=str, default="./vectorstore",
                        help="Output directory for the vector store (default: ./vectorstore)")
    parser.add_argument("--embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformers model for embeddings (default: all-MiniLM-L6-v2)")
    parser.add_argument("--max-chunk-chars", type=int, default=2500,
                        help="Max characters per chunk for seed docs (default: 2500)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size (default: 64)")
    args = parser.parse_args()

    seed_docs_path = os.path.join(args.knowledge_dir, "seed_docs.json")
    knowledge_path = os.path.join(args.knowledge_dir, "knowledge.csv")

    # --- 1. Load and chunk seed documents ---
    documents = []

    if os.path.exists(seed_docs_path):
        with open(seed_docs_path, "r") as f:
            seed_docs = json.load(f)
        print(f"Loaded {len(seed_docs)} seed documents from {seed_docs_path}")
        for doc in seed_docs:
            chunks = chunk_text(doc["content"], max_chars=args.max_chunk_chars)
            for chunk in chunks:
                documents.append({
                    "text": chunk,
                    "source": "seed_doc",
                    "subtopic": doc.get("subtopic", ""),
                })
        print(f"  Chunked into {len(documents)} text segments")
    else:
        print(f"WARNING: {seed_docs_path} not found, skipping seed documents.")

    # --- 2. Load SDG QA pairs ---
    if os.path.exists(knowledge_path):
        df = pd.read_csv(knowledge_path)
        print(f"Loaded {len(df)} QA pairs from {knowledge_path}")

        q_col = "question" if "question" in df.columns else None
        r_col = "response" if "response" in df.columns else None

        if q_col and r_col:
            for _, row in df.iterrows():
                q = str(row[q_col]).strip()
                a = str(row[r_col]).strip()
                if q and a and q != "nan" and a != "nan":
                    documents.append({
                        "text": f"Q: {q}\nA: {a}",
                        "source": "sdg_qa",
                        "subtopic": "",
                    })
        else:
            print(f"  WARNING: Expected 'question' and 'response' columns. "
                  f"Found: {list(df.columns)}")
    else:
        print(f"WARNING: {knowledge_path} not found, skipping QA pairs.")

    if not documents:
        print("ERROR: No documents to index. Run 01_extract_knowledge.py first.")
        return

    # Deduplicate
    seen = set()
    unique_docs = []
    for doc in documents:
        if doc["text"] not in seen:
            seen.add(doc["text"])
            unique_docs.append(doc)
    if len(unique_docs) < len(documents):
        print(f"Removed {len(documents) - len(unique_docs)} duplicates.")
    documents = unique_docs

    print(f"\nTotal documents to embed: {len(documents)}")

    # --- 3. Generate embeddings ---
    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)
    dimension = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dimension}")

    texts = [doc["text"] for doc in documents]
    print(f"Generating embeddings (batch size: {args.batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # --- 4. Build FAISS index ---
    print("Building FAISS index (IndexFlatIP for cosine similarity)...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors")

    # --- 5. Save to disk ---
    os.makedirs(args.output, exist_ok=True)

    index_path = os.path.join(args.output, "index.faiss")
    faiss.write_index(index, index_path)

    metadata = [
        {"text": doc["text"], "source": doc["source"], "subtopic": doc["subtopic"]}
        for doc in documents
    ]
    metadata_path = os.path.join(args.output, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    config = {
        "embedding_model": args.embedding_model,
        "dimension": dimension,
        "num_vectors": index.ntotal,
        "max_chunk_chars": args.max_chunk_chars,
    }
    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nVector store saved to {args.output}/")
    print(f"  {index_path}    ({index.ntotal} vectors, {dimension}d)")
    print(f"  {metadata_path}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()
