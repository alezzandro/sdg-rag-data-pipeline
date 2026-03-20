# sdg-rag-data-pipeline

Extract knowledge directly from a teacher model, index it in a vector
store, and augment a smaller model's answers with retrieval-augmented
generation (RAG). No external documentation required — knowledge comes
from the teacher model itself.

This pipeline replaces the fine-tuning approach (see
`example-projects/sdg-finetune-pipeline/`) which produced degraded
answers due to catastrophic forgetting on small models.

| Step | Script | Tool | Purpose |
|------|--------|------|---------|
| — | `00_serve_model.py` | [vLLM](https://github.com/vllm-project/vllm) | Serve models (teacher for extraction, small for RAG) |
| 1 | `01_extract_knowledge.py` | [OpenAI API](https://github.com/openai/openai-python) + [SDG Hub](https://github.com/instructlab/sdg) | 3-phase knowledge extraction from teacher model |
| 2 | `02_build_vectorstore.py` | [sentence-transformers](https://github.com/UKPLab/sentence-transformers) + [FAISS](https://github.com/facebookresearch/faiss) | Embed and index knowledge in a vector store |
| 3 | `03_query_rag.py` | FAISS + OpenAI API | RAG-augmented queries against a small model |
| 4 | `04_evaluate_rag.py` | FAISS + OpenAI API + [rouge-score](https://github.com/google-research/google-research/tree/master/rouge) | Evaluate RAG quality using SDG QA pairs |

## How It Works

### Step 1: Knowledge Extraction (3 phases)

1. **Topic Decomposition** — The teacher model breaks a domain/topic
   into structured subtopics
2. **Seed Document Generation** — The teacher model writes a detailed
   technical article for each subtopic
3. **SDG Expansion** — Each seed document is fed through an SDG Hub flow
   to produce structured QA pairs

### Steps 2–4: RAG Pipeline

The seed documents and QA pairs are chunked, embedded, and stored in a
FAISS vector store. At query time, the most relevant chunks are
retrieved and injected into the small model's prompt as context.

## Target Environment

This pipeline is tested on **RHEL AI 1.5** running on an **AWS
g6.xlarge** instance (1 x NVIDIA L4 24 GB). It should work on any RHEL
or RHEL AI system with an NVIDIA GPU and `podman`.

| Component | Requirement |
|-----------|-------------|
| OS | RHEL AI 1.5 / RHEL 9 |
| GPU | NVIDIA L4 24 GB (tested) — any CUDA-capable GPU with 16+ GB VRAM |
| Container runtime | `podman` (pre-installed on RHEL AI) |
| NVIDIA GPU drivers | Pre-installed on RHEL AI |
| `nvidia-container-toolkit` | Pre-installed on RHEL AI; provides CDI for GPU passthrough |

> On non-RHEL-AI hosts, install the NVIDIA Container Toolkit following
> the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Host Setup

RHEL AI ships with Python 3.9 on the host, which is too old for this
pipeline. Everything runs inside a **UBI9** container with Python 3.12,
using podman to pass through the NVIDIA GPU.

### 1. Enable user lingering (required for remote sessions)

When you SSH into the machine and later disconnect, systemd kills all
processes belonging to your user — including podman containers. Enable
**lingering** so containers survive SSH disconnects:

```bash
sudo loginctl enable-linger $(whoami)
```

This only needs to be done once. Without it, any running container will
be stopped when you log out.

> If podman shows errors like `"invalid internal status"` after a
> disconnect, run `podman system migrate` to reset.

### 2. Build the container image

```bash
git clone <this-repo-url>
cd sdg-rag-data-pipeline
podman build -t sdg-rag-pipeline .
```

The `Containerfile` installs Python 3.12, system libraries, and all pip
dependencies except vLLM (which requires CUDA at install time and must
be installed inside the running container).

### 3. Create and start the container

Create a persistent, detached container with GPU access and a named
volume for the Hugging Face model cache:

```bash
podman run -d --name sdg-rag-container \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v hf-cache:/root/.cache/huggingface \
  sdg-rag-pipeline \
  sleep infinity
```

The container runs `sleep infinity` as PID 1, so it stays alive even
when you disconnect your exec session. The `hf-cache` volume persists
downloaded model weights across container restarts.

| Flag | Purpose |
|------|---------|
| `-d` | Run detached (in background) |
| `--name sdg-rag-container` | Name for easy reference |
| `--device nvidia.com/gpu=all` | CDI GPU passthrough |
| `--security-opt=label=disable` | Prevent SELinux from blocking GPU access |
| `-v hf-cache:/root/.cache/huggingface` | Persist model downloads |

### 4. Connect to the container

```bash
podman exec -it sdg-rag-container /bin/bash
```

You can disconnect and reconnect at any time — the container and all
background processes inside it keep running.

### 5. First-time setup inside the container

Install vLLM (requires GPU access, so it cannot be done at build time):

```bash
pip3.12 install vllm
```

Clone the repository inside the container:

```bash
cd /workspace
git clone <this-repo-url>
cd sdg-rag-data-pipeline
```

## Quick Start

All commands below are run inside the container.

```bash
# Start the teacher model (runs in background)
python3.12 00_serve_model.py --preset 14b
tail -f vllm_server.log
# Wait for: "INFO:     Application startup complete." (~2 minutes)
# Press Ctrl+C to stop tailing (the server keeps running)

# Step 1 — Extract knowledge from teacher model (runs in background)
python3.12 01_extract_knowledge.py \
  --model "hosted_vllm/Qwen/Qwen2.5-14B-Instruct-AWQ" \
  --url http://localhost:8000/v1 --token dummy \
  --domain "Infrastructure" \
  --topic "OpenShift Virtualization Networking" \
  --num-subtopics 20 \
  --resume --background

# Monitor progress
tail -f extract_knowledge.log

# Stop the teacher model once extraction is done (frees GPU)
python3.12 00_serve_model.py --stop

# Step 2 — Build vector store (fast, runs on CPU)
python3.12 02_build_vectorstore.py

# Start the small model for RAG queries
python3.12 00_serve_model.py --model "ibm-granite/granite-3.3-2b-instruct"
tail -f vllm_server.log
# Wait for ready...

# Step 3 — Query with RAG (side-by-side comparison)
python3.12 03_query_rag.py \
  --model "hosted_vllm/ibm-granite/granite-3.3-2b-instruct" \
  --url http://localhost:8000/v1 --token dummy \
  --question "How do I configure a VM with a secondary network interface?" \
  --system-prompt "You are an expert in OpenShift Virtualization networking." \
  --no-context

# Step 4 — Evaluate RAG quality (runs in background)
python3.12 04_evaluate_rag.py \
  --model "hosted_vllm/ibm-granite/granite-3.3-2b-instruct" \
  --url http://localhost:8000/v1 --token dummy \
  --system-prompt "You are an expert in OpenShift Virtualization networking." \
  --with-baseline --background
```

## Serving a Local LLM

`00_serve_model.py` starts a [vLLM](https://github.com/vllm-project/vllm)
server in the **background** that exposes an OpenAI-compatible API. The
server is used twice: first for the teacher model (knowledge extraction)
and then for the small model (RAG queries).

> **Important:** The teacher and small models both need the GPU. Stop
> the teacher with `--stop` before serving the small model.

### Recommended Models for the L4 24 GB

| Preset | Model | Quantization | VRAM | Quality |
|--------|-------|-------------|------|---------|
| `7b` | `Qwen/Qwen2.5-7B-Instruct` | FP16 | ~16 GB | Good |
| `14b` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | AWQ 4-bit | ~10 GB | Better |

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--preset` | No | `7b` | Predefined model configuration (`7b` or `14b`) |
| `--model` | No | | HuggingFace model ID (overrides `--preset`) |
| `--quantization` | No | | Quantization method: `awq` or `gptq` |
| `--port` | No | `8000` | Port for the API server |
| `--max-model-len` | No | `16384` | Maximum context length |
| `--gpu-memory-utilization` | No | `0.90` | Fraction of GPU memory to use |
| `--tensor-parallel-size` | No | `1` | Number of GPUs for tensor parallelism |
| `--stop` | No | | Stop a running vLLM server |
| `--status` | No | | Show whether the server is running |

## Step 1: Extract Knowledge

`01_extract_knowledge.py` extracts knowledge from the teacher model in
three phases, producing `subtopics.json`, `seed_docs.json`, and
`knowledge.csv` in the output directory.

### Phase 1 — Topic Decomposition

The teacher model decomposes the topic into structured subtopics. Each
subtopic gets a title and description, saved to `subtopics.json`.

### Phase 2 — Seed Document Generation

For each subtopic, the teacher model generates a detailed technical
article (1000–2000 words). Articles are generated concurrently and
checkpointed after each batch.

### Phase 3 — SDG Expansion

Each seed document is chunked and fed through an SDG Hub flow (default:
Key Facts) to produce structured QA pairs. Output is saved to
`knowledge.csv`.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | | LLM identifier (e.g. `hosted_vllm/Qwen/Qwen2.5-14B-Instruct-AWQ`) |
| `--url` | Yes | | OpenAI-compatible API base URL |
| `--token` | Yes | | API key / bearer token |
| `--topic` | Yes | | Topic to extract knowledge about |
| `--domain` | No | `General` | Knowledge domain label |
| `--num-subtopics` | No | `20` | Number of subtopics to generate |
| `--output-dir` | No | `./knowledge` | Directory for output artifacts |
| `--flow` | No | `Key Facts Knowledge Tuning Dataset Generation Flow` | SDG Hub flow name |
| `--max-concurrency` | No | `4` | Max concurrent LLM requests |
| `--timeout` | No | `1800` | Per-request timeout in seconds |
| `--batch-size` | No | `5` | Subtopics per batch for checkpointing |
| `--max-chunk-chars` | No | `2500` | Max characters per chunk for SDG |
| `--keep-cot` | No | | Keep reasoning/chain-of-thought tags |
| `--resume` | No | | Resume from last checkpoint |
| `--phase` | No | `all` | Run a specific phase: `decompose`, `generate`, `expand`, or `all` |
| `--background` | No | | Run in background |
| `--status` | No | | Show status of a background process |
| `--stop` | No | | Stop a running background process |

### Examples

```bash
# Full extraction in background
python3.12 01_extract_knowledge.py \
  --model "hosted_vllm/Qwen/Qwen2.5-14B-Instruct-AWQ" \
  --url http://localhost:8000/v1 --token dummy \
  --domain "Infrastructure" \
  --topic "OpenShift Virtualization Networking" \
  --num-subtopics 20 \
  --resume --background

# Run only topic decomposition
python3.12 01_extract_knowledge.py \
  --model "hosted_vllm/Qwen/Qwen2.5-14B-Instruct-AWQ" \
  --url http://localhost:8000/v1 --token dummy \
  --topic "Kubernetes Security" \
  --phase decompose

# Check status
python3.12 01_extract_knowledge.py --status
```

> **Resilient processing:** Each phase checks for existing output. With
> `--resume`, completed phases are skipped. Within Phases 2 and 3,
> batch checkpointing allows recovery from failures.

## Step 2: Build the Vector Store

`02_build_vectorstore.py` chunks the seed documents, combines them with
the SDG QA pairs, generates embeddings with sentence-transformers, and
builds a FAISS index.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--knowledge-dir` | No | `./knowledge` | Directory containing `seed_docs.json` and `knowledge.csv` |
| `--output` | No | `./vectorstore` | Output directory for the vector store |
| `--embedding-model` | No | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformers model |
| `--max-chunk-chars` | No | `2500` | Max characters per chunk |
| `--batch-size` | No | `64` | Embedding batch size |

### Examples

```bash
# Build with defaults
python3.12 02_build_vectorstore.py

# Use a different embedding model
python3.12 02_build_vectorstore.py \
  --embedding-model "nomic-ai/nomic-embed-text-v1.5"
```

## Step 3: Query with RAG

`03_query_rag.py` retrieves relevant context from the vector store and
queries the small model with an augmented prompt.

Use `--no-context` to also query the model without RAG and see both
answers side by side.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--vectorstore` | No | `./vectorstore` | Path to the vector store |
| `--model` | Yes | | LLM identifier |
| `--url` | Yes | | API base URL |
| `--token` | Yes | | API key |
| `--question` | Yes | | Question to ask |
| `--system-prompt` | No | `You are a helpful technical assistant.` | System prompt |
| `--top-k` | No | `5` | Number of chunks to retrieve |
| `--max-new-tokens` | No | `512` | Max tokens to generate |
| `--no-context` | No | | Also query without RAG for comparison |

### Examples

```bash
# Simple RAG query
python3.12 03_query_rag.py \
  --model "hosted_vllm/ibm-granite/granite-3.3-2b-instruct" \
  --url http://localhost:8000/v1 --token dummy \
  --question "How do I expose a VM with a Kubernetes service?"

# Side-by-side comparison (with and without RAG)
python3.12 03_query_rag.py \
  --model "hosted_vllm/ibm-granite/granite-3.3-2b-instruct" \
  --url http://localhost:8000/v1 --token dummy \
  --question "What is the difference between masquerade and bridge networking?" \
  --system-prompt "You are an expert in OpenShift Virtualization networking." \
  --no-context
```

## Step 4: Evaluate RAG Quality

`04_evaluate_rag.py` runs each question from `knowledge.csv` through the
RAG pipeline and compares the answer against the reference using ROUGE-L
and token-level F1.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--vectorstore` | No | `./vectorstore` | Path to the vector store |
| `--knowledge-dir` | No | `./knowledge` | Directory containing `knowledge.csv` |
| `--model` | Yes | | LLM identifier |
| `--url` | Yes | | API base URL |
| `--token` | Yes | | API key |
| `--system-prompt` | No | `You are a helpful technical assistant.` | System prompt |
| `--top-k` | No | `5` | Number of chunks to retrieve |
| `--max-new-tokens` | No | `512` | Max tokens to generate |
| `--output` | No | `evaluation_results.csv` | Output CSV path |
| `--sample-size` | No | | Evaluate a random subset of N questions |
| `--with-baseline` | No | | Also query without RAG for comparison |
| `--background` | No | | Run in background |
| `--status` | No | | Show status |
| `--stop` | No | | Stop background process |

### Examples

```bash
# Full evaluation with baseline comparison
python3.12 04_evaluate_rag.py \
  --model "hosted_vllm/ibm-granite/granite-3.3-2b-instruct" \
  --url http://localhost:8000/v1 --token dummy \
  --with-baseline --background

# Quick evaluation on a subset
python3.12 04_evaluate_rag.py \
  --model "hosted_vllm/ibm-granite/granite-3.3-2b-instruct" \
  --url http://localhost:8000/v1 --token dummy \
  --sample-size 20
```

## Managing the Container

### Reconnecting after SSH disconnect

The container keeps running after you disconnect (thanks to `loginctl
enable-linger` and the `sleep infinity` entrypoint). To reconnect:

```bash
ssh cloud-user@<your-host>
podman exec -it sdg-rag-container /bin/bash
cd /workspace/sdg-rag-data-pipeline
```

### Useful commands

```bash
# Check container status
podman ps

# Check vLLM server status
python3.12 00_serve_model.py --status

# Check extraction status
python3.12 01_extract_knowledge.py --status

# Check evaluation status
python3.12 04_evaluate_rag.py --status

# Stop everything and remove the container
python3.12 04_evaluate_rag.py --stop
python3.12 01_extract_knowledge.py --stop
python3.12 00_serve_model.py --stop
exit
podman stop sdg-rag-container
podman rm sdg-rag-container
```

### Recovering from podman errors

If podman shows `"invalid internal status"` after a session disconnect
(usually means lingering was not enabled):

```bash
podman system migrate
podman start sdg-rag-container
podman exec -it sdg-rag-container /bin/bash
```

## Project Structure

```
00_serve_model.py          # Serve models via vLLM (reused from finetune pipeline)
01_extract_knowledge.py    # Step 1 — 3-phase knowledge extraction from teacher model
02_build_vectorstore.py    # Step 2 — embed + build FAISS index
03_query_rag.py            # Step 3 — RAG-augmented queries
04_evaluate_rag.py         # Step 4 — batch evaluation with metrics
Containerfile              # UBI9 container image with pipeline dependencies
README.md
example-projects/          # Previous finetune pipeline (reference)
```

## License

Apache-2.0
