# Experiment runner image (ray tune + tensorboard).
#
#   docker build -t cjd-flows -f dockerfile .
#   docker run --gpus all -p 6006:6006 cjd-flows
#
# The PyPI torch wheels bundle the CUDA runtime; only the NVIDIA driver (via the
# NVIDIA container toolkit on the host) is needed for GPU support.
FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# uv (pinned) from the official image
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python

WORKDIR /cjd-flows

# Install the dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md LICENCE.md ./
RUN uv sync --locked --no-dev --extra experiments --no-install-project --python 3.12
COPY . .
RUN uv sync --locked --no-dev --extra experiments --python 3.12

# Expose port for TensorBoard
EXPOSE 6006

RUN chmod a+rx start.sh start_test.sh
# Use ./start.sh to run all experiment configs.
CMD ["./start_test.sh"]
