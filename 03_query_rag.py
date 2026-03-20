"""RAG-augmented query against a small model served via vLLM.

Loads the FAISS vector store built by 02_build_vectorstore.py, retrieves
the most relevant chunks for a user question, and sends the augmented
prompt to a small model through an OpenAI-compatible API.

Use --no-context to also query the model without RAG context for a
side-by-side comparison.
"""

import os
import json
import argparse

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI


RAG_SYSTEM_TEMPLATE = """\
{system_prompt}

Use the following reference material to answer the user's question. \
If the reference material does not contain enough information, say so \
and answer based on your general knowledge.

--- REFERENCE MATERIAL ---
{context}
--- END REFERENCE MATERIAL ---"""


def load_vectorstore(path):
    """Load a FAISS index, metadata, and config from disk."""
    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    index_path = os.path.join(path, "index.faiss")
    index = faiss.read_index(index_path)

    metadata_path = os.path.join(path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return index, metadata, config


def retrieve(index, metadata, query_embedding, top_k=5):
    """Retrieve top-k most similar chunks from the FAISS index."""
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
        })
    return results


def query_model(client, model, messages, max_tokens=512):
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
        description="Query a small model with RAG-augmented context from the vector store."
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
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate (default: 512)")
    parser.add_argument("--no-context", action="store_true",
                        help="Also query without RAG context for side-by-side comparison")
    args = parser.parse_args()

    # Strip hosted_vllm/ prefix for OpenAI client
    api_model = args.model
    if api_model.startswith("hosted_vllm/"):
        api_model = api_model[len("hosted_vllm/"):]

    # --- Load vector store ---
    print(f"Loading vector store from {args.vectorstore}/")
    index, metadata, config = load_vectorstore(args.vectorstore)
    print(f"  {config['num_vectors']} vectors, {config['dimension']}d "
          f"({config['embedding_model']})")

    # --- Embed the question ---
    print(f"Loading embedding model: {config['embedding_model']}")
    embed_model = SentenceTransformer(config["embedding_model"])
    query_embedding = embed_model.encode(
        args.question, normalize_embeddings=True
    )

    # --- Retrieve relevant chunks ---
    print(f"Retrieving top-{args.top_k} chunks...")
    results = retrieve(index, metadata, query_embedding, top_k=args.top_k)

    separator = "-" * 72
    print(f"\n{'='*72}")
    print("RETRIEVED CONTEXT")
    print(separator)
    for i, r in enumerate(results, 1):
        source_label = r["source"]
        if r["subtopic"]:
            source_label += f" / {r['subtopic']}"
        print(f"\n[{i}] (score: {r['score']:.4f}, source: {source_label})")
        preview = r["text"][:300]
        if len(r["text"]) > 300:
            preview += "..."
        print(preview)
    print(f"\n{'='*72}")

    # --- Build RAG prompt and query ---
    context_block = "\n\n".join(
        f"[{i+1}] {r['text']}" for i, r in enumerate(results)
    )
    rag_system = RAG_SYSTEM_TEMPLATE.format(
        system_prompt=args.system_prompt,
        context=context_block,
    )

    client = OpenAI(base_url=args.url, api_key=args.token)

    print(f"\nQuerying model: {api_model}")
    print(f"Question: {args.question}\n")

    rag_messages = [
        {"role": "system", "content": rag_system},
        {"role": "user", "content": args.question},
    ]
    rag_answer = query_model(client, api_model, rag_messages, args.max_new_tokens)

    if args.no_context:
        # Also query without RAG for comparison
        bare_messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.question},
        ]
        bare_answer = query_model(client, api_model, bare_messages, args.max_new_tokens)

        print(f"\n{'='*72}")
        print(f"ANSWER WITHOUT RAG  ({api_model})")
        print(separator)
        print(bare_answer)
        print(f"\n{'='*72}")
        print(f"ANSWER WITH RAG  ({api_model}, top-{args.top_k} context)")
        print(separator)
        print(rag_answer)
        print(f"{'='*72}")
    else:
        print(f"\n{'='*72}")
        print(f"ANSWER  ({api_model}, top-{args.top_k} RAG context)")
        print(separator)
        print(rag_answer)
        print(f"{'='*72}")


if __name__ == "__main__":
    main()
