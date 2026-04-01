FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    ca-certificates \
    wget curl \
    && rm -rf /var/lib/apt/lists/*


# Build-time sanity check
RUN python --version && nvcc --version && gcc --version

WORKDIR /workspace