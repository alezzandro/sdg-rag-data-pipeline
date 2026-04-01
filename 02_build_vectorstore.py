"""Build a FAISS + BM25 hybrid vector store from seed documents and SDG QA pairs.

Reads the artifacts produced by 01_extract_knowledge.py (seed_docs.json and
knowledge.csv), chunks and embeds them with sentence-transformers, builds a
FAISS index for dense retrieval, and a BM25 index for sparse retrieval.

The hybrid store is used by 03_query_rag.py and 04_evaluate_rag.py with
reciprocal rank fusion (RRF) to combine dense and sparse scores.
"""

import os
import re
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


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


def chunk_text(text, max_chars=2000):
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


def contextualize_chunk(chunk_text, subtopic="", source=""):
    """Prepend context header to make each chunk self-contained."""
    parts = []
    if subtopic:
        parts.append(f"Topic: {subtopic}")
    if source:
        parts.append(f"Source: {source}")
    if parts:
        return "\n".join(parts) + "\n\n" + chunk_text
    return chunk_text


def tokenize_for_bm25(text):
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS + BM25 hybrid vector store from seed docs and SDG QA pairs."
    )
    parser.add_argument("--knowledge-dir", type=str, default="./knowledge",
                        help="Directory containing seed_docs.json and knowledge.csv (default: ./knowledge)")
    parser.add_argument("--output", type=str, default="./vectorstore",
                        help="Output directory for the vector store (default: ./vectorstore)")
    parser.add_argument("--embedding-model", type=str,
                        default="BAAI/bge-base-en-v1.5",
                        help="Sentence-transformers model for embeddings (default: BAAI/bge-base-en-v1.5)")
    parser.add_argument("--max-chunk-chars", type=int, default=2000,
                        help="Max characters per chunk for seed docs (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size (default: 64)")
    parser.add_argument("--no-bm25", action="store_true",
                        help="Skip BM25 index creation (dense-only mode)")
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
            subtopic = doc.get("subtopic", "")
            chunks = chunk_text(doc["content"], max_chars=args.max_chunk_chars)
            for chunk in chunks:
                ctx_text = contextualize_chunk(chunk, subtopic=subtopic, source="reference document")
                documents.append({
                    "text": ctx_text,
                    "text_raw": chunk,
                    "source": "seed_doc",
                    "subtopic": subtopic,
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
                    qa_text = f"Q: {q}\nA: {a}"
                    documents.append({
                        "text": qa_text,
                        "text_raw": qa_text,
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

    # --- 5. Build BM25 index ---
    bm25_built = False
    if not args.no_bm25:
        if BM25Okapi is None:
            print("WARNING: rank_bm25 not installed, skipping BM25 index. "
                  "Install with: pip install rank_bm25")
        else:
            print("Building BM25 index for hybrid retrieval...")
            tokenized_corpus = [tokenize_for_bm25(doc["text"]) for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_built = True
            print(f"  BM25 index built over {len(tokenized_corpus)} documents")

    # --- 6. Save to disk ---
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

    # Detect query instruction for BGE-family models
    query_instruction = ""
    model_name_lower = args.embedding_model.lower()
    if "bge-" in model_name_lower and "v1.5" in model_name_lower:
        query_instruction = "Represent this sentence for searching relevant passages: "

    config = {
        "embedding_model": args.embedding_model,
        "dimension": dimension,
        "num_vectors": index.ntotal,
        "max_chunk_chars": args.max_chunk_chars,
        "query_instruction": query_instruction,
        "has_bm25": bm25_built,
    }
    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if bm25_built:
        bm25_path = os.path.join(args.output, "bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)

    print(f"\nVector store saved to {args.output}/")
    print(f"  {index_path}    ({index.ntotal} vectors, {dimension}d)")
    print(f"  {metadata_path}")
    print(f"  {config_path}")
    if bm25_built:
        print(f"  {os.path.join(args.output, 'bm25.pkl')}")
    if query_instruction:
        print(f"  Query instruction: \"{query_instruction}\"")


if __name__ == "__main__":
    main()
