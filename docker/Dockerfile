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
ENV PYTHON_VENV_PATH=/usr/local/venv
ENV PATH="$PYTHON_VENV_PATH/bin:$PATH"
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    python3 -m venv "$PYTHON_VENV_PATH"
EOS

# Installs all GSplat dependencies (including dev dependencies) -----------
FROM python as gsplat-deps

# Main library dependencies
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    packages=(
        # Library deps
        "ninja==1.13.0"
        "numpy==2.2.6"
        "jaxtyping==0.3.4"
        "rich==14.3.3"
        "torch==2.9.1"
        "typing_extensions; python_version<'3.8'"
    )

    pip install "${packages[@]}"

    # To save space, we can remove the python cache files
    rm -rf $PYTHONPYCACHEPREFIX/*
EOS

# Test/dev dependencies
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    # Dev/test deps
    packages=(
        "black[jupyter]==22.3.0"
        "isort==5.10.1"
        "pylint==2.13.4"
        "pytest==7.1.3"
        "pytest-env==0.8.1"
        "pytest-xdist==2.5.0"
        "typeguard==4.5.1"
        "pyyaml==6.0.1"
        "build==1.3.0"
        "twine==1.6.0"
        "cupy==13.6.0"
        "nerfacc==0.5.3"
        "PLAS @ git+https://github.com/fraunhoferhhi/PLAS.git"
        "imageio[ffmpeg]==2.37.2"
        "torchpq==0.3.0.6"
    )

    # Tests need cupy, let install faster (and smaller) by targeting a specific compute and rely on CUDA JIT.
    CUPY_NUM_BUILD_JOBS=$(nproc) \
    CUPY_NUM_NVCC_THREADS=$(nproc) \
    CUPY_NVCC_GENERATE_CODE=arch=compute_80,code=compute_80 \
        pip install "${packages[@]}"

    # To save space, we can remove the python cache files
    rm -rf $PYTHONPYCACHEPREFIX/*
EOS

# Examples dependencies
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    # fused-ssim only builds for a pre-defined set of architectures.
    # In order to override it, and support, say, Blackwell, we need a more
    # recent commit that allows overriding this set.
    # Existing architectures:
    # sm_75: Turing/GeForce tu10x
    # sm_80: Ampere/Tesla ga100
    # sm_89: Ada/GeForce ad102
    # New architectures
    # sm_90: Hopper/Tesla gh100
    # sm_120: Blackwell/GeForce gb20x
    # sm_121: Blackwell/Tesla gb20x, gb21x
    CUDA_ARCHITECTURES='75;80;89;90;120;121' \
        pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"

    # Examples
    packages=(
        "torchvision==0.24.1"
        "git+https://github.com/brandonyzhao/pycolmap@a2a5671b0f6ab641957dd4c5a78cabcf220c28ee"
        "viser==1.0.24"
        "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
        "scikit-learn==1.8.0"
        "torchmetrics[image]==1.8.2"
        "tyro==1.0.8"
        "piexif==1.1.3"
        "tensorboard==2.20.0"
        "tensorly==0.9.0"
        "git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16"
        "splines==0.3.3"
        "ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.0"
        "opencv-python-headless==4.10.0.84"
    )

    # Needs --no-build-isolation or else fused-ssim can't be installed as
    # it needs torch in its setup.py.
    pip install --no-build-isolation "${packages[@]}"

    # To save space, we can remove the python cache files
    rm -rf $PYTHONPYCACHEPREFIX/*
EOS

# Main image ============================================================
# It combines the Python environment, development toolchain
# and all gsplat dependencies into an image.

FROM python as main

COPY --link --from=gsplat-deps "$PYTHON_VENV_PATH/" "$PYTHON_VENV_PATH/"
COPY --link --from=ccache ccache /usr/local/bin/ccache

# Now install everything else we need from apt
RUN --mount=type=cache,id=aptcache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=aptcache,target=/var/lib/apt,sharing=locked \
    --mount=type=tmpfs,target=/tmp \
<<EOS
    apt-get update

    packages=(
        # For JIT-build
        ninja-build curl unzip
        # generally needed
        vim less sudo
        # For the entrypoint script
        gosu
    )
    apt-get install "${packages[@]}"
EOS

ENV CCACHE_CONFIGPATH=/etc/ccache.conf
ENV PATH=/usr/local/bin:$PATH

COPY --link profile.d/00-persistent-cache.sh /etc/profile.d/

# Leave some small changes last
RUN <<EOS
    # Make apt cache persistent
    mkdir -p /var/cache/apt/lib
    rm -rf /var/lib/apt
    ln -sf /var/cache/apt/lib /var/lib/apt
    # World-writable so any host UID mapped into the container can use the
    # shared persistent cache (ccache, CUDA JIT, pip).  Acceptable in a
    # local dev-only image; do NOT use in production/shared environments.
    chmod 777 /var/cache

    ccache --max-size=20G

    for compiler in c++ g++ cpp cc gcc-13 g++-13; do
        ln -sf /usr/local/bin/ccache /usr/local/bin/$compiler
    done

    # Run nvcc via ccache.
    # We can't use a symlink like we did with the host compilers or else
    # cuda sdk path detection done by setuptools won't work.
    mv /usr/local/cuda/bin/nvcc{,-real}

    # Some of the profile scripts set up the config, we need
    # to do it for root
    for file in /etc/profile.d/*.sh; do
        source "$file"
    done

    # Allow sudoers do use python venv automatically
    echo "Defaults secure_path=\"$PYTHON_VENV_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" > /etc/sudoers.d/10-secure-path

    # Remove all non-system users/groups (UID/GID >= 1000, except nobody/nogroup)
    # so the entrypoint can create the host user without conflicts
    nogroup_gid="$(getent group nogroup | cut -d: -f3)"
    awk -F: -v ng="$nogroup_gid" '$3 >= 1000 && $1 != "nobody" && $4 != ng { print $1 }' /etc/passwd \
        | xargs -r -n1 userdel 2>/dev/null || true
    awk -F: '$3 >= 1000 && $1 != "nogroup" { print $1 }' /etc/group \
        | xargs -r -n1 groupdel 2>/dev/null || true
EOS

COPY --link cuda/nvcc_wrapper.sh /usr/local/cuda/bin/nvcc
COPY --chmod=755 entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash", "--login"]

# Our local, persistent cache volume
VOLUME /var/cache

# Viser is exposed via this port
# Used in examples.
EXPOSE 8080/tcp

WORKDIR /root
