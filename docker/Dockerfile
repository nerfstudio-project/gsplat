# syntax=docker/dockerfile:1.7.0

# Fix the base image to avoid spurious rebuilds
ARG BASE_IMAGE=nvidia/cuda:12.9.0-devel-ubuntu24.04@sha256:f418bff454b1ac74a4d9d39279f2cf664fbb9b5d5f21de104b79745ea29bf2cb
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-ex", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

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
ARG VIRTUAL_ENV=~/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    python3 -m venv ~/.venv
EOS

# Install all gsplat dependencies
COPY --link --from=gsplatrepo setup.py deps/
COPY --link --from=gsplatrepo gsplat/version.py deps/gsplat/
RUN --mount=type=cache,id=pipcache,target=/root/.cache/pip,sharing=locked \
    --mount=type=tmpfs,target=/tmp \
<<EOS

    pushd deps

    # Tests need cupy, let install faster (and smaller) by targeting a specific compute and rely on CUDA JIT.
    CUPY_NUM_BUILD_JOBS=$(nproc) \
    CUPY_NUM_NVCC_THREADS=$(nproc) \
    CUPY_NVCC_GENERATE_CODE=arch=compute_80,code=compute_80 \
    BUILD_NO_CUDA=1 \
      pip install -e .[dev]

    popd
    rm -rf deps
EOS

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
