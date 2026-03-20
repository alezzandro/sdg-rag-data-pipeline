"""Evaluate the RAG pipeline using SDG-generated QA pairs as ground truth.

For each question in knowledge.csv, retrieves context from the vector store,
queries the small model, and compares the RAG answer against the reference
answer using ROUGE-L and token-level F1.

Optionally queries the model without RAG context (--with-baseline) to
measure the improvement from retrieval augmentation.
"""

import os
import sys
import json
import signal
import argparse
import subprocess
import time
import random
from collections import Counter

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

LOGFILE = "evaluate_rag.log"
PIDFILE = "evaluate_rag.pid"


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
    with open(os.path.join(path, "config.json"), "r") as f:
        config = json.load(f)
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return index, metadata, config


def retrieve(index, metadata, query_embedding, top_k=5):
    """Retrieve top-k chunks from the FAISS index."""
    query_vec = np.array([query_embedding], dtype=np.float32)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({"text": metadata[idx]["text"], "score": float(score)})
    return results


def query_model(client, model, messages, max_tokens=512):
    """Send a chat completion request."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def token_f1(prediction, reference):
    """Compute token-level F1 between prediction and reference strings."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction, reference):
    """Compute ROUGE-L F1 score."""
    if rouge_scorer is None:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


# ---------------------------------------------------------------------------
# Background / PID management
# ---------------------------------------------------------------------------

def _read_pid():
    if not os.path.exists(PIDFILE):
        return None
    with open(PIDFILE) as f:
        try:
            pid = int(f.read().strip())
        except ValueError:
            return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        os.remove(PIDFILE)
        return None


def _stop_process():
    pid = _read_pid()
    if pid is None:
        print("No running evaluation process found.")
        return
    print(f"Stopping evaluation process (PID {pid})...")
    os.kill(pid, signal.SIGTERM)
    for _ in range(15):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except OSError:
            break
    else:
        print("Process did not exit gracefully, sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
    if os.path.exists(PIDFILE):
        os.remove(PIDFILE)
    print("Evaluation process stopped.")


def _show_status():
    pid = _read_pid()
    if pid is None:
        print("No running evaluation process found.")
        return
    print(f"Evaluation process is running (PID {pid}).")
    if os.path.exists(LOGFILE):
        print(f"Log file: {os.path.abspath(LOGFILE)}")
        with open(LOGFILE) as f:
            lines = f.readlines()
        progress = [l for l in lines if "Evaluated" in l or "Summary" in l]
        if progress:
            print(f"Last progress: {progress[-1].strip()}")
        else:
            print("Status: starting up...")


def _cleanup_pidfile():
    if not os.path.exists(PIDFILE):
        return
    with open(PIDFILE) as f:
        try:
            pid = int(f.read().strip())
        except ValueError:
            return
    if pid == os.getpid():
        os.remove(PIDFILE)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline using SDG QA pairs as ground truth."
    )
    parser.add_argument("--vectorstore", type=str, default="./vectorstore",
                        help="Path to the vector store directory (default: ./vectorstore)")
    parser.add_argument("--knowledge-dir", type=str, default="./knowledge",
                        help="Directory containing knowledge.csv (default: ./knowledge)")
    parser.add_argument("--model", type=str,
                        help="LLM model identifier")
    parser.add_argument("--url", type=str,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--token", type=str,
                        help="API key / bearer token")
    parser.add_argument("--system-prompt", type=str,
                        default="You are a helpful technical assistant.",
                        help="System prompt for the model")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate (default: 512)")
    parser.add_argument("--output", type=str, default="evaluation_results.csv",
                        help="Output CSV path (default: evaluation_results.csv)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Evaluate a random subset of N questions")
    parser.add_argument("--with-baseline", action="store_true",
                        help="Also query without RAG for baseline comparison")
    parser.add_argument("--background", action="store_true",
                        help="Run in background")
    parser.add_argument("--status", action="store_true",
                        help="Show the status of a background evaluation")
    parser.add_argument("--stop", action="store_true",
                        help="Stop a running background evaluation")
    args = parser.parse_args()

    if args.status:
        _show_status()
        return

    if args.stop:
        _stop_process()
        return

    if args.background:
        existing_pid = _read_pid()
        if existing_pid is not None:
            print(f"ERROR: Evaluation is already running (PID {existing_pid}).",
                  file=sys.stderr)
            sys.exit(1)

        log_path = os.path.abspath(LOGFILE)
        bg_args = [a for a in sys.argv if a != "--background"]
        with open(LOGFILE, "w") as log_f:
            proc = subprocess.Popen(
                [sys.executable] + bg_args,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        with open(PIDFILE, "w") as pf:
            pf.write(str(proc.pid))

        print(f"Evaluation started in background (PID {proc.pid}).")
        print(f"\n{'='*60}")
        print(f"  Log file: {log_path}")
        print(f"  Monitor:  tail -f {LOGFILE}")
        print(f"  Status:   python3.12 04_evaluate_rag.py --status")
        print(f"  Stop:     python3.12 04_evaluate_rag.py --stop")
        print(f"{'='*60}")
        return

    # --- Validate required args ---
    missing = [n for n in ("model", "url", "token") if getattr(args, n) is None]
    if missing:
        parser.error(f"the following arguments are required: "
                     f"{', '.join('--' + m for m in missing)}")

    # --- Load knowledge.csv ---
    knowledge_path = os.path.join(args.knowledge_dir, "knowledge.csv")
    if not os.path.exists(knowledge_path):
        print(f"ERROR: {knowledge_path} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(knowledge_path)
    if "question" not in df.columns or "response" not in df.columns:
        print(f"ERROR: knowledge.csv must have 'question' and 'response' columns.",
              file=sys.stderr)
        sys.exit(1)

    df = df.dropna(subset=["question", "response"])
    qa_pairs = list(zip(df["question"].tolist(), df["response"].tolist()))

    if args.sample_size and args.sample_size < len(qa_pairs):
        qa_pairs = random.sample(qa_pairs, args.sample_size)
        print(f"Sampled {args.sample_size} questions for evaluation.")

    print(f"Evaluating {len(qa_pairs)} questions.")

    # --- Load vector store ---
    print(f"Loading vector store from {args.vectorstore}/")
    index, metadata, config = load_vectorstore(args.vectorstore)
    print(f"  {config['num_vectors']} vectors ({config['embedding_model']})")

    print(f"Loading embedding model: {config['embedding_model']}")
    embed_model = SentenceTransformer(config["embedding_model"])

    # Strip hosted_vllm/ prefix for OpenAI client
    api_model = args.model
    if api_model.startswith("hosted_vllm/"):
        api_model = api_model[len("hosted_vllm/"):]
    client = OpenAI(base_url=args.url, api_key=args.token)

    # --- Evaluate ---
    results = []
    for i, (question, reference) in enumerate(qa_pairs, 1):
        question = str(question).strip()
        reference = str(reference).strip()

        print(f"\n[{i}/{len(qa_pairs)}] {question[:80]}...")

        # Embed and retrieve
        q_embedding = embed_model.encode(question, normalize_embeddings=True)
        chunks = retrieve(index, metadata, q_embedding, top_k=args.top_k)
        context_block = "\n\n".join(
            f"[{j+1}] {c['text']}" for j, c in enumerate(chunks)
        )

        # RAG query
        rag_system = RAG_SYSTEM_TEMPLATE.format(
            system_prompt=args.system_prompt,
            context=context_block,
        )
        rag_messages = [
            {"role": "system", "content": rag_system},
            {"role": "user", "content": question},
        ]
        try:
            rag_answer = query_model(client, api_model, rag_messages, args.max_new_tokens)
        except Exception as e:
            print(f"  RAG query failed: {e}")
            rag_answer = ""

        row = {
            "question": question,
            "reference": reference,
            "rag_answer": rag_answer,
            "rag_token_f1": token_f1(rag_answer, reference),
            "rag_rouge_l": compute_rouge_l(rag_answer, reference),
            "top_chunk_score": chunks[0]["score"] if chunks else 0.0,
        }

        # Baseline (no RAG) query
        if args.with_baseline:
            bare_messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": question},
            ]
            try:
                bare_answer = query_model(client, api_model, bare_messages, args.max_new_tokens)
            except Exception as e:
                print(f"  Baseline query failed: {e}")
                bare_answer = ""

            row["baseline_answer"] = bare_answer
            row["baseline_token_f1"] = token_f1(bare_answer, reference)
            row["baseline_rouge_l"] = compute_rouge_l(bare_answer, reference)

        results.append(row)
        print(f"  RAG F1={row['rag_token_f1']:.3f}  ROUGE-L={row.get('rag_rouge_l', 'N/A')}")
        if args.with_baseline:
            print(f"  Base F1={row['baseline_token_f1']:.3f}  "
                  f"ROUGE-L={row.get('baseline_rouge_l', 'N/A')}")

        if i % 10 == 0:
            print(f"Evaluated {i}/{len(qa_pairs)} questions.")

    # --- Save results ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Questions evaluated:  {len(results)}")
    print(f"  RAG Token F1 (avg):   {results_df['rag_token_f1'].mean():.4f}")
    if "rag_rouge_l" in results_df.columns and results_df["rag_rouge_l"].notna().any():
        print(f"  RAG ROUGE-L (avg):    {results_df['rag_rouge_l'].mean():.4f}")
    if args.with_baseline:
        print(f"  Baseline Token F1:    {results_df['baseline_token_f1'].mean():.4f}")
        if "baseline_rouge_l" in results_df.columns and results_df["baseline_rouge_l"].notna().any():
            print(f"  Baseline ROUGE-L:     {results_df['baseline_rouge_l'].mean():.4f}")
        improvement = results_df["rag_token_f1"].mean() - results_df["baseline_token_f1"].mean()
        print(f"  RAG improvement (F1): {improvement:+.4f}")
    print(f"{'='*60}")

    _cleanup_pidfile()


if __name__ == "__main__":
    main()
