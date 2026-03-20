FROM registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.3.0

USER 0

# The vLLM image is missing /tmp; dill (torch dep) crashes without it.
RUN mkdir -p /tmp /var/tmp && chmod 1777 /tmp /var/tmp

RUN dnf install -y git gcc python3.12-devel && \
    dnf clean all

RUN pip3.12 install --no-cache-dir \
        sdg-hub \
        sentence-transformers \
        faiss-cpu \
        rouge-score \
        openai

WORKDIR /workspace
COPY . /workspace

# Reset to the default non-root user
USER 1001

# Override the vLLM entrypoint so the container can run arbitrary commands
ENTRYPOINT []
CMD ["sleep", "infinity"]
