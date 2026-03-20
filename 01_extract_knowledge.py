"""Extract knowledge from a teacher model in three phases.

Phase 1 — Topic Decomposition:
    Ask the teacher model to break a domain/topic into structured subtopics.

Phase 2 — Seed Document Generation:
    For each subtopic, ask the teacher model to write a detailed article
    capturing its knowledge.

Phase 3 — SDG Expansion:
    Feed each seed document through an SDG Hub flow to produce structured
    QA pairs for maximum coverage.

Outputs (all written to --output-dir):
    subtopics.json   — structured list of subtopics
    seed_docs.json   — detailed articles for each subtopic
    knowledge.csv    — SDG-expanded QA pairs
"""

import os
import sys
import json
import signal
import argparse
import re
import subprocess
import time
import asyncio

import pandas as pd
from openai import OpenAI
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

os.environ["LITELLM_LOG"] = "INFO"

LOGFILE = "extract_knowledge.log"
PIDFILE = "extract_knowledge.pid"

REASONING_TAG_PATTERN = re.compile(
    r"<(think|reasoning|reflection)>.*?</\1>", flags=re.DOTALL
)


def strip_reasoning_traces(text):
    """Remove chain-of-thought tags emitted by reasoning models."""
    if not isinstance(text, str):
        return text
    return REASONING_TAG_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------------
# Phase 1 — Topic Decomposition
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
You are a domain expert in {domain}.

Break the topic "{topic}" into exactly {num_subtopics} subtopics that \
together provide comprehensive coverage. Each subtopic should be specific \
enough for a standalone technical article of 1000-2000 words.

Return ONLY a JSON array (no markdown fences, no commentary) where each \
element is an object with two keys:
  "title"       — concise subtopic title
  "description" — 1-2 sentence scope description

Example format:
[
  {{"title": "Example Subtopic", "description": "Covers X, Y, and Z."}}
]
"""


def decompose_topic(client, model, domain, topic, num_subtopics, timeout):
    """Ask the teacher model to decompose a topic into subtopics."""
    prompt = DECOMPOSE_PROMPT.format(
        domain=domain, topic=topic, num_subtopics=num_subtopics
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
        timeout=timeout,
    )
    raw = resp.choices[0].message.content
    raw = strip_reasoning_traces(raw)

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    subtopics = json.loads(raw)
    if not isinstance(subtopics, list):
        raise ValueError("Model did not return a JSON array")
    return subtopics


# ---------------------------------------------------------------------------
# Phase 2 — Seed Document Generation
# ---------------------------------------------------------------------------

GENERATE_PROMPT = """\
You are a domain expert in {domain}, writing reference documentation about \
"{topic}".

Write a comprehensive, detailed technical article about the following subtopic:

**{title}**
{description}

Requirements:
- Write 1000-2000 words of factual, detailed content
- Use markdown formatting with headings, bullet points, and code blocks where appropriate
- Include specific configuration examples, commands, or API references when relevant
- Write as an authoritative reference, not a tutorial
- Do not include a title heading (the title is already known)
"""


def generate_seed_doc(client, model, domain, topic, subtopic, timeout):
    """Ask the teacher model to generate a detailed article for one subtopic."""
    prompt = GENERATE_PROMPT.format(
        domain=domain,
        topic=topic,
        title=subtopic["title"],
        description=subtopic["description"],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=8192,
        timeout=timeout,
    )
    content = resp.choices[0].message.content
    return strip_reasoning_traces(content)


async def generate_seed_docs_batch(
    client, model, domain, topic, subtopics, timeout, max_concurrency
):
    """Generate seed documents for a batch of subtopics concurrently."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _gen(subtopic):
        async with semaphore:
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                generate_seed_doc,
                client, model, domain, topic, subtopic, timeout,
            )
            return {
                "subtopic": subtopic["title"],
                "title": subtopic["title"],
                "content": content,
            }

    tasks = [_gen(st) for st in subtopics]
    return await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Phase 3 — SDG Expansion (reuses SDG Hub flow from finetune pipeline)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Background / PID management (same pattern as finetune pipeline)
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
        print("No running extraction process found.")
        return
    print(f"Stopping extraction process (PID {pid})...")
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
    print("Extraction process stopped.")


def _show_status():
    pid = _read_pid()
    if pid is None:
        print("No running extraction process found.")
        return
    print(f"Extraction process is running (PID {pid}).")
    if os.path.exists(LOGFILE):
        print(f"Log file: {os.path.abspath(LOGFILE)}")
        with open(LOGFILE) as f:
            lines = f.readlines()
        progress = [
            l for l in lines
            if "Phase" in l or "Checkpoint" in l or "Done" in l
        ]
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract knowledge from a teacher model using topic "
        "decomposition, seed document generation, and SDG expansion."
    )
    parser.add_argument("--model", type=str,
                        help="LLM model identifier (e.g. hosted_vllm/Qwen/Qwen2.5-14B-Instruct-AWQ)")
    parser.add_argument("--url", type=str,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--token", type=str,
                        help="API key / bearer token")
    parser.add_argument("--domain", type=str, default="General",
                        help="Knowledge domain (default: General)")
    parser.add_argument("--topic", type=str,
                        help="Topic to extract knowledge about")
    parser.add_argument("--num-subtopics", type=int, default=20,
                        help="Number of subtopics to generate (default: 20)")
    parser.add_argument("--output-dir", type=str, default="./knowledge",
                        help="Directory for output artifacts (default: ./knowledge)")
    parser.add_argument("--flow", type=str,
                        default="Key Facts Knowledge Tuning Dataset Generation Flow",
                        help="SDG Hub flow name for Phase 3")
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="Max concurrent LLM requests (default: 4)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-request timeout in seconds (default: 1800)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Subtopics per batch for checkpointing (default: 5)")
    parser.add_argument("--max-chunk-chars", type=int, default=2500,
                        help="Max characters per chunk for SDG expansion (default: 2500)")
    parser.add_argument("--keep-cot", action="store_true",
                        help="Keep reasoning/chain-of-thought tags in output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "decompose", "generate", "expand"],
                        help="Run a specific phase only (default: all)")
    parser.add_argument("--background", action="store_true",
                        help="Run in background with output logged to extract_knowledge.log")
    parser.add_argument("--status", action="store_true",
                        help="Show the status of a background extraction process")
    parser.add_argument("--stop", action="store_true",
                        help="Stop a running background extraction process")
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
            print(f"ERROR: Extraction is already running (PID {existing_pid}).",
                  file=sys.stderr)
            print("Stop it first with: python3.12 01_extract_knowledge.py --stop",
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

        print(f"Extraction started in background (PID {proc.pid}).")
        print(f"\n{'='*60}")
        print(f"  Log file: {log_path}")
        print(f"")
        print(f"  Monitor progress:")
        print(f"    tail -f {LOGFILE}")
        print(f"")
        print(f"  Check status:")
        print(f"    python3.12 01_extract_knowledge.py --status")
        print(f"")
        print(f"  Stop the process:")
        print(f"    python3.12 01_extract_knowledge.py --stop")
        print(f"{'='*60}")
        return

    # --- Validate required args ---
    missing = [n for n in ("model", "url", "token", "topic")
               if getattr(args, n) is None]
    if missing:
        parser.error(f"the following arguments are required: "
                     f"{', '.join('--' + m for m in missing)}")

    os.makedirs(args.output_dir, exist_ok=True)

    subtopics_path = os.path.join(args.output_dir, "subtopics.json")
    seed_docs_path = os.path.join(args.output_dir, "seed_docs.json")
    knowledge_path = os.path.join(args.output_dir, "knowledge.csv")
    seed_checkpoint = os.path.join(args.output_dir, ".seed_docs.checkpoint.json")
    sdg_checkpoint = os.path.join(args.output_dir, ".knowledge.checkpoint.csv")
    sdg_meta = sdg_checkpoint + ".meta"

    # Strip the hosted_vllm/ prefix for direct OpenAI client calls
    api_model = args.model
    if api_model.startswith("hosted_vllm/"):
        api_model = api_model[len("hosted_vllm/"):]

    client = OpenAI(base_url=args.url, api_key=args.token)

    run_phase1 = args.phase in ("all", "decompose")
    run_phase2 = args.phase in ("all", "generate")
    run_phase3 = args.phase in ("all", "expand")

    # =======================================================================
    # Phase 1 — Topic Decomposition
    # =======================================================================
    if run_phase1:
        if args.resume and os.path.exists(subtopics_path):
            print(f"Phase 1: Skipping — {subtopics_path} already exists (--resume)")
            with open(subtopics_path, "r") as f:
                subtopics = json.load(f)
        else:
            print(f"\n{'='*60}")
            print(f"Phase 1: Topic Decomposition")
            print(f"  Domain: {args.domain}")
            print(f"  Topic:  {args.topic}")
            print(f"  Target: {args.num_subtopics} subtopics")
            print(f"{'='*60}\n")

            subtopics = decompose_topic(
                client, api_model, args.domain, args.topic,
                args.num_subtopics, args.timeout,
            )
            with open(subtopics_path, "w") as f:
                json.dump(subtopics, f, indent=2)

            print(f"Generated {len(subtopics)} subtopics:")
            for i, st in enumerate(subtopics, 1):
                print(f"  {i:2d}. {st['title']}")
            print(f"\nSaved to {subtopics_path}")
    else:
        if os.path.exists(subtopics_path):
            with open(subtopics_path, "r") as f:
                subtopics = json.load(f)
        else:
            print(f"ERROR: {subtopics_path} not found. Run phase 'decompose' first.",
                  file=sys.stderr)
            sys.exit(1)

    # =======================================================================
    # Phase 2 — Seed Document Generation
    # =======================================================================
    if run_phase2:
        seed_docs = []
        docs_done = 0

        if args.resume and os.path.exists(seed_checkpoint):
            with open(seed_checkpoint, "r") as f:
                seed_docs = json.load(f)
            docs_done = len(seed_docs)
            print(f"Phase 2: Resuming — {docs_done}/{len(subtopics)} seed docs done")

        if not args.resume and os.path.exists(seed_checkpoint):
            os.remove(seed_checkpoint)

        remaining_subtopics = subtopics[docs_done:]

        if remaining_subtopics:
            print(f"\n{'='*60}")
            print(f"Phase 2: Seed Document Generation")
            print(f"  {len(remaining_subtopics)} subtopics remaining")
            print(f"  Concurrency: {args.max_concurrency}")
            print(f"{'='*60}\n")

            total_batches = (len(remaining_subtopics) + args.batch_size - 1) // args.batch_size
            for batch_idx in range(total_batches):
                start = batch_idx * args.batch_size
                end = min(start + args.batch_size, len(remaining_subtopics))
                batch = remaining_subtopics[start:end]

                abs_start = docs_done + start + 1
                abs_end = docs_done + end
                print(f"\n--- Batch {batch_idx + 1}/{total_batches} "
                      f"(subtopics {abs_start}-{abs_end} of {len(subtopics)}) ---")

                results = asyncio.run(
                    generate_seed_docs_batch(
                        client, api_model, args.domain, args.topic,
                        batch, args.timeout, args.max_concurrency,
                    )
                )

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"  FAILED: {batch[i]['title']} — {result}")
                    else:
                        seed_docs.append(result)
                        print(f"  OK: {result['title']} "
                              f"({len(result['content'])} chars)")

                with open(seed_checkpoint, "w") as f:
                    json.dump(seed_docs, f, indent=2)
                print(f"Checkpoint saved ({len(seed_docs)}/{len(subtopics)} seed docs).")

        # Finalize seed docs
        with open(seed_docs_path, "w") as f:
            json.dump(seed_docs, f, indent=2)
        if os.path.exists(seed_checkpoint):
            os.remove(seed_checkpoint)
        print(f"\nPhase 2 complete — {len(seed_docs)} seed docs saved to {seed_docs_path}")
    else:
        if os.path.exists(seed_docs_path):
            with open(seed_docs_path, "r") as f:
                seed_docs = json.load(f)
        else:
            print(f"ERROR: {seed_docs_path} not found. Run phase 'generate' first.",
                  file=sys.stderr)
            sys.exit(1)

    # =======================================================================
    # Phase 3 — SDG Expansion
    # =======================================================================
    if run_phase3:
        print(f"\n{'='*60}")
        print(f"Phase 3: SDG Expansion")
        print(f"  Flow: {args.flow}")
        print(f"{'='*60}\n")

        # Chunk all seed docs into SDG-compatible input
        data_list = []
        for doc in seed_docs:
            chunks = chunk_text(doc["content"], max_chars=args.max_chunk_chars)
            for chunk in chunks:
                data_list.append({
                    "document": chunk,
                    "document_outline": f"{args.topic} — {doc['subtopic']}",
                    "domain": args.domain,
                })

        # Deduplicate
        seen = set()
        unique_data = []
        for item in data_list:
            if item["document"] not in seen:
                seen.add(item["document"])
                unique_data.append(item)
        data_list = unique_data

        print(f"Prepared {len(data_list)} chunks from {len(seed_docs)} seed docs.")

        # Checkpoint / resume logic
        chunks_done = 0
        if args.resume and os.path.exists(sdg_meta):
            with open(sdg_meta, "r") as mf:
                meta = json.load(mf)
            chunks_done = meta.get("chunks_processed", 0)
            print(f"Resuming: {chunks_done}/{len(data_list)} chunks already processed.")
        elif not args.resume:
            for fp in (sdg_checkpoint, sdg_meta):
                if os.path.exists(fp):
                    os.remove(fp)

        remaining = data_list[chunks_done:]
        if not remaining:
            print("All chunks already processed. Finalizing...")
        else:
            print(f"{len(remaining)} chunks remaining to process.")

            # Load SDG flow
            print(f"Loading flow: {args.flow}")
            FlowRegistry.discover_flows()
            flow_path = FlowRegistry.get_flow_path(args.flow)
            if flow_path is None:
                print(f"Error: flow '{args.flow}' not found.")
                _cleanup_pidfile()
                return

            flow = Flow.from_yaml(flow_path)
            flow.set_model_config(
                model=args.model,
                api_base=args.url,
                api_key=args.token,
                timeout=args.timeout,
            )

            sdg_batch_size = args.batch_size * 5
            already_done = len(data_list) - len(remaining)
            total_batches = (len(remaining) + sdg_batch_size - 1) // sdg_batch_size
            for batch_idx in range(total_batches):
                start = batch_idx * sdg_batch_size
                end = min(start + sdg_batch_size, len(remaining))
                batch = remaining[start:end]
                batch_ds = Dataset.from_list(batch)

                abs_start = already_done + start + 1
                abs_end = already_done + end
                print(f"\n--- SDG Batch {batch_idx + 1}/{total_batches} "
                      f"(chunks {abs_start}-{abs_end} of {len(data_list)}) ---")
                try:
                    result = flow.generate(batch_ds, max_concurrency=args.max_concurrency)
                except Exception as e:
                    print(f"\nBatch {batch_idx + 1} failed: {e}")
                    print(f"Progress saved. Re-run with --resume to continue.")
                    _cleanup_pidfile()
                    return

                batch_df = result.to_pandas()
                write_header = not os.path.exists(sdg_checkpoint)
                batch_df.to_csv(sdg_checkpoint, mode="a", index=False, header=write_header)

                with open(sdg_meta, "w") as mf:
                    json.dump({"chunks_processed": abs_end}, mf)
                print(f"Checkpoint saved ({abs_end}/{len(data_list)} chunks done).")

        # Finalize knowledge.csv
        if os.path.exists(sdg_checkpoint):
            df = pd.read_csv(sdg_checkpoint)
            if not args.keep_cot:
                print("Stripping reasoning traces from output columns...")
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].apply(strip_reasoning_traces)
            df.to_csv(knowledge_path, index=False)
            for fp in (sdg_checkpoint, sdg_meta):
                if os.path.exists(fp):
                    os.remove(fp)
            print(f"\nPhase 3 complete — {len(df)} QA pairs saved to {knowledge_path}")
        elif os.path.exists(knowledge_path):
            df = pd.read_csv(knowledge_path)
            print(f"\nPhase 3: {knowledge_path} already exists ({len(df)} rows).")
        else:
            print("WARNING: No SDG output produced.")

    _cleanup_pidfile()
    print(f"\nDone — all artifacts in {args.output_dir}/")


if __name__ == "__main__":
    main()
