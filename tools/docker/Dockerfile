# Refer to
# https://www.tensorflow.org/install/source#gpu
# then to
# https://hub.docker.com/r/nvidia/cuda/tags
# for required base image.

FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
COPY requirements.txt /tmp

RUN apt-get update && apt-get install -y \
    python3-pip \
    && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && \
    rm -rf /var/lib/apt/lists/*
