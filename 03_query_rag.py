"""RAG-augmented query against a small model served via vLLM.

Loads the FAISS + BM25 hybrid vector store built by 02_build_vectorstore.py,
retrieves the most relevant chunks using reciprocal rank fusion (RRF),
optionally reranks with a cross-encoder, filters by relevance threshold,
and sends the augmented prompt to a small model.

Use --no-context to also query the model without RAG context for a
side-by-side comparison.
"""

import os
import re
import json
import pickle
import argparse

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


RAG_SYSTEM_TEMPLATE = """\
{system_prompt}

You are given reference material below that MAY be relevant to the user's \
question. Follow these rules strictly:

1. If the reference material directly addresses the question, use it to \
produce a detailed, accurate answer with specific examples.
2. If the reference material is only tangentially related or does NOT \
address the question, IGNORE it entirely and answer using your own \
knowledge. Do NOT force irrelevant references into your answer.
3. Never mention "the reference material" or "the provided context" in \
your answer — write as if the knowledge is your own.

--- REFERENCE MATERIAL ---
{context}
--- END REFERENCE MATERIAL ---"""


def load_vectorstore(path):
    """Load FAISS index, BM25 index (if available), metadata, and config."""
    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    index_path = os.path.join(path, "index.faiss")
    index = faiss.read_index(index_path)

    metadata_path = os.path.join(path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    bm25 = None
    bm25_path = os.path.join(path, "bm25.pkl")
    if config.get("has_bm25") and os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

    return index, metadata, config, bm25


def tokenize_for_bm25(text):
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def retrieve_hybrid(index, metadata, bm25, query_embedding, query_text,
                    top_k=5, top_k_initial=20, rrf_k=60):
    """Retrieve using reciprocal rank fusion of dense (FAISS) and sparse (BM25) scores."""
    fetch_k = max(top_k_initial, top_k * 4)

    query_vec = np.array([query_embedding], dtype=np.float32)
    dense_scores, dense_indices = index.search(query_vec, fetch_k)

    dense_ranks = {}
    for rank, idx in enumerate(dense_indices[0]):
        if idx >= 0:
            dense_ranks[int(idx)] = rank + 1

    sparse_ranks = {}
    if bm25 is not None:
        tokens = tokenize_for_bm25(query_text)
        bm25_scores = bm25.get_scores(tokens)
        top_sparse_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
        for rank, idx in enumerate(top_sparse_indices):
            if bm25_scores[idx] > 0:
                sparse_ranks[int(idx)] = rank + 1

    all_indices = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    rrf_scores = []
    for idx in all_indices:
        score = 0.0
        if idx in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[idx])
        if idx in sparse_ranks:
            score += 1.0 / (rrf_k + sparse_ranks[idx])
        dense_score = float(dense_scores[0][list(dense_indices[0]).index(idx)]) if idx in dense_ranks else 0.0
        rrf_scores.append((idx, score, dense_score))

    rrf_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, rrf_score, dense_score in rrf_scores[:top_k_initial]:
        results.append({
            "text": metadata[idx]["text"],
            "source": metadata[idx]["source"],
            "subtopic": metadata[idx].get("subtopic", ""),
            "score": dense_score,
            "rrf_score": rrf_score,
            "index": idx,
        })
    return results


def retrieve_dense_only(index, metadata, query_embedding, top_k=20):
    """Retrieve using dense FAISS search only (fallback when BM25 unavailable)."""
    query_vec = np.array([query_embedding], dtype=np.float32)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({
            "text": metadata[idx]["text"],
            "source": metadata[idx]["source"],
            "subtopic": metadata[idx].get("subtopic", ""),
            "score": float(score),
            "rrf_score": float(score),
            "index": idx,
        })
    return results


def rerank(results, query, reranker_model, top_k=5):
    """Rerank retrieval results using a cross-encoder model."""
    if not results:
        return results
    pairs = [(query, r["text"]) for r in results]
    scores = reranker_model.predict(pairs)
    for i, score in enumerate(scores):
        results[i]["rerank_score"] = float(score)
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]


def apply_threshold(results, min_score):
    """Filter out chunks below the minimum relevance score."""
    if min_score <= 0:
        return results
    score_key = "rerank_score" if "rerank_score" in results[0] else "score"
    return [r for r in results if r.get(score_key, 0) >= min_score]


def query_model(client, model, messages, max_tokens=1024):
    """Send a chat completion request to the model."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(
        description="Query a small model with RAG-augmented context from the hybrid vector store."
    )
    parser.add_argument("--vectorstore", type=str, default="./vectorstore",
                        help="Path to the vector store directory (default: ./vectorstore)")
    parser.add_argument("--model", type=str, required=True,
                        help="LLM model identifier (e.g. hosted_vllm/ibm-granite/granite-3.3-2b-instruct)")
    parser.add_argument("--url", type=str, required=True,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--token", type=str, required=True,
                        help="API key / bearer token")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask the model")
    parser.add_argument("--system-prompt", type=str,
                        default="You are a helpful technical assistant.",
                        help="System prompt for the model")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of final chunks to use as context (default: 5)")
    parser.add_argument("--top-k-initial", type=int, default=20,
                        help="Number of candidates to retrieve before reranking (default: 20)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max tokens to generate (default: 1024)")
    parser.add_argument("--no-context", action="store_true",
                        help="Also query without RAG context for side-by-side comparison")
    parser.add_argument("--reranker", type=str,
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="Cross-encoder model for reranking (default: cross-encoder/ms-marco-MiniLM-L-6-v2)")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Disable cross-encoder reranking")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum relevance score threshold; chunks below this are dropped "
                             "(default: 0.0 = no threshold)")
    args = parser.parse_args()

    api_model = args.model
    if api_model.startswith("hosted_vllm/"):
        api_model = api_model[len("hosted_vllm/"):]

    # --- Load vector store ---
    print(f"Loading vector store from {args.vectorstore}/")
    index, metadata, config, bm25 = load_vectorstore(args.vectorstore)
    print(f"  {config['num_vectors']} vectors, {config['dimension']}d "
          f"({config['embedding_model']})")
    if bm25 is not None:
        print(f"  BM25 index loaded (hybrid retrieval enabled)")
    else:
        print(f"  No BM25 index (dense-only retrieval)")

    # --- Load embedding model ---
    print(f"Loading embedding model: {config['embedding_model']}")
    embed_model = SentenceTransformer(config["embedding_model"])

    query_instruction = config.get("query_instruction", "")
    query_text = args.question
    embed_input = query_instruction + query_text if query_instruction else query_text

    query_embedding = embed_model.encode(embed_input, normalize_embeddings=True)

    # --- Retrieve candidates ---
    print(f"Retrieving top-{args.top_k_initial} candidates...")
    if bm25 is not None:
        results = retrieve_hybrid(
            index, metadata, bm25, query_embedding, query_text,
            top_k=args.top_k, top_k_initial=args.top_k_initial,
        )
    else:
        results = retrieve_dense_only(index, metadata, query_embedding, top_k=args.top_k_initial)

    # --- Rerank ---
    reranker_model = None
    if not args.no_reranker and results:
        print(f"Reranking with cross-encoder: {args.reranker}")
        reranker_model = CrossEncoder(args.reranker)
        results = rerank(results, query_text, reranker_model, top_k=args.top_k)
    else:
        results = results[:args.top_k]

    # --- Apply threshold ---
    if args.min_score > 0 and results:
        pre_count = len(results)
        results = apply_threshold(results, args.min_score)
        if len(results) < pre_count:
            print(f"  Threshold {args.min_score}: kept {len(results)}/{pre_count} chunks")
        if not results:
            print(f"  All chunks below threshold — model will use its own knowledge")

    # --- Display retrieved context ---
    separator = "-" * 72
    print(f"\n{'='*72}")
    print("RETRIEVED CONTEXT")
    print(separator)
    if results:
        for i, r in enumerate(results, 1):
            source_label = r["source"]
            if r["subtopic"]:
                source_label += f" / {r['subtopic']}"
            score_info = f"score: {r['score']:.4f}"
            if "rerank_score" in r:
                score_info += f", rerank: {r['rerank_score']:.4f}"
            if "rrf_score" in r:
                score_info += f", rrf: {r['rrf_score']:.4f}"
            print(f"\n[{i}] ({score_info}, source: {source_label})")
            preview = r["text"][:300]
            if len(r["text"]) > 300:
                preview += "..."
            print(preview)
    else:
        print("\n(no chunks passed the relevance threshold)")
    print(f"\n{'='*72}")

    # --- Build RAG prompt and query ---
    client = OpenAI(base_url=args.url, api_key=args.token)

    print(f"\nQuerying model: {api_model}")
    print(f"Question: {args.question}\n")

    if args.no_context:
        bare_messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.question},
        ]
        bare_answer = query_model(client, api_model, bare_messages, args.max_new_tokens)

        print(f"\n{'='*72}")
        print(f"ANSWER WITHOUT RAG  ({api_model})")
        print(separator)
        print(bare_answer)

    if results:
        context_block = "\n\n".join(
            f"[{i+1}] {r['text']}" for i, r in enumerate(results)
        )
        rag_system = RAG_SYSTEM_TEMPLATE.format(
            system_prompt=args.system_prompt,
            context=context_block,
        )
    else:
        rag_system = args.system_prompt

    rag_messages = [
        {"role": "system", "content": rag_system},
        {"role": "user", "content": args.question},
    ]
    rag_answer = query_model(client, api_model, rag_messages, args.max_new_tokens)

    context_label = f"top-{len(results)} context" if results else "no relevant context found"
    print(f"\n{'='*72}")
    print(f"ANSWER WITH RAG  ({api_model}, {context_label})")
    print(separator)
    print(rag_answer)
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
