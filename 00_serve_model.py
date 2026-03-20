#!/usr/bin/env python3
"""Start a local vLLM OpenAI-compatible API server in the background.

The server runs as a background process with output redirected to a log
file.  Once the startup message appears in the log, you can use
01_extract_knowledge.py or 03_query_rag.py against it:

    python3.12 01_extract_knowledge.py \
        --model "hosted_vllm/<MODEL>" \
        --url http://localhost:8000/v1 \
        --token dummy \
        ...
"""

import argparse
import os
import signal
import shutil
import subprocess
import sys
import time

LOGFILE = "vllm_server.log"
PIDFILE = "vllm_server.pid"
READY_MARKER = "Application startup complete"

PRESETS = {
    "7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "quantization": None,
        "description": "7B FP16 — ~16 GB VRAM, good quality",
    },
    "14b": {
        "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "quantization": "awq",
        "description": "14B 4-bit AWQ — ~10 GB VRAM, better quality",
    },
}


def _check_gpu():
    """Quick sanity check that an NVIDIA GPU is visible."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            text=True,
        )
        for line in out.strip().splitlines():
            print(f"  GPU detected: {line.strip()}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _read_pid():
    """Read the PID from the pidfile, return None if missing or stale."""
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


def _stop_server():
    """Stop a running vLLM server."""
    pid = _read_pid()
    if pid is None:
        print("No running vLLM server found.")
        return
    print(f"Stopping vLLM server (PID {pid})...")
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except OSError:
            break
    else:
        print(f"Server did not exit gracefully, sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
    for f in (PIDFILE,):
        if os.path.exists(f):
            os.remove(f)
    print("Server stopped.")


def _show_status():
    """Show the status of the vLLM server."""
    pid = _read_pid()
    if pid is None:
        print("No running vLLM server found.")
        return
    print(f"vLLM server is running (PID {pid}).")
    if os.path.exists(LOGFILE):
        print(f"Log file: {os.path.abspath(LOGFILE)}")
        with open(LOGFILE) as f:
            content = f.read()
        if READY_MARKER in content:
            print("Status: READY (accepting requests)")
        else:
            print("Status: STARTING (model still loading...)")


def main():
    parser = argparse.ArgumentParser(
        description="Start/stop a local vLLM OpenAI-compatible API server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Presets (use --preset instead of --model):",
            *(f"  {k:6s}  {v['description']}" for k, v in PRESETS.items()),
        ]),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preset", choices=PRESETS.keys(),
                       help="Use a predefined model configuration")
    group.add_argument("--model", type=str,
                       help="HuggingFace model ID (overrides --preset)")

    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq"],
                        help="Quantization method (required for 4-bit models)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the API server (default: 8000)")
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Maximum context length (default: 16384)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (default: 0.90)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--stop", action="store_true",
                        help="Stop a running vLLM server")
    parser.add_argument("--status", action="store_true",
                        help="Show the status of the vLLM server")
    args = parser.parse_args()

    if args.status:
        _show_status()
        return

    if args.stop:
        _stop_server()
        return

    # Check if a server is already running
    existing_pid = _read_pid()
    if existing_pid is not None:
        print(f"ERROR: vLLM server is already running (PID {existing_pid}).",
              file=sys.stderr)
        print(f"Stop it first with: python3.12 00_serve_model.py --stop",
              file=sys.stderr)
        sys.exit(1)

    # Resolve model from preset or explicit flag
    if args.model:
        model = args.model
        quantization = args.quantization
    elif args.preset:
        preset = PRESETS[args.preset]
        model = preset["model"]
        quantization = args.quantization or preset["quantization"]
        print(f"Using preset '{args.preset}': {preset['description']}")
    else:
        preset = PRESETS["7b"]
        model = preset["model"]
        quantization = args.quantization or preset["quantization"]
        print(f"No model specified — defaulting to preset '7b': {preset['description']}")

    # GPU check
    print("\nChecking GPU availability...")
    if not _check_gpu():
        print("ERROR: No NVIDIA GPU detected. vLLM requires a CUDA-capable GPU.",
              file=sys.stderr)
        sys.exit(1)

    # Verify vllm is installed (CLI binary or Python module)
    use_vllm_cli = bool(shutil.which("vllm"))
    if not use_vllm_cli:
        try:
            subprocess.check_call(
                [sys.executable, "-c", "import vllm"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ERROR: vLLM not found. Install with: pip install vllm",
                  file=sys.stderr)
            sys.exit(1)

    # Build the vllm serve command
    if use_vllm_cli:
        cmd = ["vllm", "serve", model]
    else:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", model]

    cmd += [
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    if quantization:
        cmd += ["--quantization", quantization]

    # Start the server in the background
    log_path = os.path.abspath(LOGFILE)
    print(f"\nStarting vLLM server in background...")
    print(f"  Model:          {model}")
    if quantization:
        print(f"  Quantization:   {quantization}")
    print(f"  Port:           {args.port}")
    print(f"  Log file:       {log_path}")
    print(f"  Command:        {' '.join(cmd)}")

    with open(LOGFILE, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    with open(PIDFILE, "w") as pf:
        pf.write(str(proc.pid))

    print(f"\n  Server started (PID {proc.pid}).")
    print(f"\n{'='*68}")
    print(f"  Wait for the model to load — this usually takes 1–3 minutes.")
    print(f"  Watch the log with:")
    print(f"    tail -f {LOGFILE}")
    print(f"")
    print(f"  The server is ready when you see:")
    print(f'    "INFO:     Application startup complete."')
    print(f"")
    print(f"  Other useful commands:")
    print(f"    python3.12 00_serve_model.py --status    # check server status")
    print(f"    python3.12 00_serve_model.py --stop      # stop the server")
    print(f"    ps aux | grep vllm                       # find vLLM processes")
    print(f"    kill {proc.pid}                                # stop manually")
    print(f"{'='*68}")
    print(f"\n  Once ready, run 01_extract_knowledge.py with:")
    print(f'    --model "hosted_vllm/{model}"')
    print(f"    --url http://localhost:{args.port}/v1")
    print(f"    --token dummy")
    print()


if __name__ == "__main__":
    main()
