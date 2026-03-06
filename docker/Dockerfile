# syntax=docker/dockerfile:1.7.0

# Fix the base image to avoid spurious rebuilds
ARG BASE_IMAGE=nvidia/cuda:12.9.0-devel-ubuntu24.04@sha256:f418bff454b1ac74a4d9d39279f2cf664fbb9b5d5f21de104b79745ea29bf2cb

# Downloads and installs ccache for build acceleration --------------------------------
FROM alpine:3.19.1 as ccache
# We need a fairly recent ccache recent ccache with proper nvcc support.
ARG CCACHE_VERSION=4.12.2
ARG CCACHE_SHA256=sha256:630c34ec94d451b200f5b14a6a25580d6a45bc80c394b7e0b93e33556eee5d32
ADD --link --checksum=$CCACHE_SHA256 https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz .
RUN <<EOS
    tar xf ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz --strip-components=1
    chmod a+x ccache
EOS

# Base layer with Python, apt configuration, and virtual environment setup ------------
FROM ${BASE_IMAGE} as python
SHELL ["/bin/bash", "-ex", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONPYCACHEPREFIX=/var/cache/python

# Add some apt config overrides to avoid
# storing unneeded files in the image.
# It also tries to make the layers stable to avoid
# spurious downloads when we regenerate the image.
COPY --link apt /etc/apt
# Install everything we need from apt to install the dependencies
RUN --mount=type=cache,id=aptcache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=aptcache,target=/var/lib/apt,sharing=locked \
    --mount=type=tmpfs,target=/tmp \
<<EOS
    mkdir -p "$PYTHONPYCACHEPREFIX"

    apt-get update

    packages=(
        python3
        libpython3-dev
        python3-venv
        python-is-python3
        # For downloading some dependencies (plas)
        git
    )

    apt-get install "${packages[@]}"
EOS

# Sets up python's virtual environment
ENV PATH="/root/.venv/bin:$PATH"
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    python3 -m venv /root/.venv
EOS

# Installs all GSplat dependencies (including dev dependencies) -----------
FROM python as gsplat-deps

COPY --link --from=gsplatrepo setup.py deps/
COPY --link --from=gsplatrepo gsplat/version.py deps/gsplat/
RUN --mount=type=tmpfs,target=/tmp \
<<EOS

    pushd deps

    # Tests need cupy, let install faster (and smaller) by targeting a specific compute and rely on CUDA JIT.
    CUPY_NUM_BUILD_JOBS=$(nproc) \
    CUPY_NUM_NVCC_THREADS=$(nproc) \
    CUPY_NVCC_GENERATE_CODE=arch=compute_80,code=compute_80 \
    BUILD_NO_CUDA=1 \
      pip install -e .[dev]

    popd

    # To save space, we can remove the python cache files
    rm -rf $PYTHONPYCACHEPREFIX/*

    rm -rf deps
EOS

# Main image ============================================================
# It combines the Python environment, development toolchain
# and all gsplat dependencies into an image.

FROM python as main

COPY --link --from=gsplat-deps /root/.venv/ /root/.venv/
COPY --link --from=ccache ccache /usr/local/bin/ccache

# Now install everything else we need from apt
RUN --mount=type=cache,id=aptcache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=aptcache,target=/var/lib/apt,sharing=locked \
    --mount=type=tmpfs,target=/tmp \
<<EOS
    apt-get update

    packages=(
        # For JIT-build
        ninja-build
        # generally needed
        vim less
    )
    apt-get install "${packages[@]}"
EOS

# Set up ccache
ENV CCACHE_CONFIGPATH=/etc/ccache.conf
ENV PATH=/usr/local/bin:$PATH
RUN <<EOS
    cat <<'EOF' >> /root/.profile

if [[ $CI_BUILDS_DIR ]]; then
    export CCACHE_DIR=$CI_BUILDS_DIR/.ci-keep/ccache
else
    export CCACHE_DIR=/var/cache/ccache
fi
EOF
    mkdir -p /var/cache/ccache

    ccache --max-size=20G

    for compiler in c++ g++ cpp cc gcc-13 g++-13; do
        ln -sf /usr/local/bin/ccache /usr/local/bin/$compiler
    done

    # Run nvcc via ccache.
    # We can't use a symlink like we did with the host compilers or else
    # cuda sdk path detection done by setuptools won't work.
    mv /usr/local/cuda/bin/nvcc{,-real}
EOS
COPY --link cuda/nvcc_wrapper.sh /usr/local/cuda/bin/nvcc

# Local cache volume, they're not meant to be in the CI cache
RUN <<EOS
    mkdir -p /var/cache/{root,apt/lib}
    ln -sf /var/cache/root /root/.cache
    rm -rf /var/lib/apt
    ln -sf /var/cache/apt/lib /var/lib/apt
EOS

# Set up the cuda JIT cache path
RUN <<EOS
    cat <<'EOF' >> /root/.profile

if [[ $CI_BUILDS_DIR ]]; then
    export CUDA_CACHE_PATH=$CI_BUILDS_DIR/.ci-keep/cuda
else
    export CUDA_CACHE_PATH=/var/cache/cuda
fi
EOF
    mkdir -p /var/cache/cuda
EOS

# Our local, persistent cache volume
VOLUME /var/cache

WORKDIR /root/gsplat
