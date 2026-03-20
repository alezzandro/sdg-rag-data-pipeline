FROM registry.access.redhat.com/ubi9/ubi

RUN dnf install -y git gcc python3.12 python3.12-pip python3.12-devel libxcb mesa-libGL && \
    dnf clean all

WORKDIR /workspace

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip3.12 install --no-cache-dir --upgrade pip && \
    pip3.12 install --no-cache-dir \
        sdg-hub \
        sentence-transformers \
        faiss-cpu \
        rouge-score \
        openai

# vllm needs CUDA at install time (CDI not available during podman build).
# Install it once inside the running container:
#   pip3.12 install vllm

COPY . /workspace
