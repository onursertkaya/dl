# Refer to
# https://www.tensorflow.org/install/source#gpu
# then to
# https://hub.docker.com/r/nvidia/cuda/tags
# for required base image.
# cudnn8-devel is needed for XLA support.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
COPY tools/conf/requirements.txt /tmp

RUN apt-get update && apt-get install -y \
    python3-pip \
    && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && \
    rm -rf /var/lib/apt/lists/*
