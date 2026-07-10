# syntax=docker/dockerfile:1.7.0

# Fix the base image to avoid spurious rebuilds
ARG BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2

# Bumped on every image release (mirrors IMAGE_TAG in config.yaml). Used to
# bake the venv at /var/cache/venv-${IMAGE_VERSION} so a per-version cache
# volume mounted at the same sub-path is auto-populated by Docker on first
# use, and old version's venvs coexist instead of forcing --reset-cache on
# image bumps.
ARG IMAGE_VERSION

# This project's image registry, passed in at build time so its URL is not
# hard-coded here as well.
ARG DOCKER_REGISTRY

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
ARG IMAGE_VERSION
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

    # Derive the CUDA minor series (12.8) from the base image's CUDA_VERSION,
    # then reassign it to the dash form (12-8) the apt package names use, so the
    # toolkit tracks the base image without repeating the version.
    cuda_version=${CUDA_VERSION%.*}
    cuda_version=${cuda_version//./-}

    packages=(
        python3
        libpython3-dev
        python3-venv
        python-is-python3
        # For downloading some dependencies (plas)
        git
        # CUDA toolchain subset — the -base image ships only the CUDA runtime:
        # - cuda-nvcc: compiles gsplat and the source-built dependencies
        # - cuda-cudart-dev + cuda-cccl: CUDA runtime headers, cub/thrust
        # - cuda-nvrtc-dev: nvrtc as a discoverable library for torch tooling
        # - cuda-nvtx: profiling ranges referenced through torch headers
        # - libcu{blas,sparse,solver,rand}-dev: headers torch's CUDA runtime
        #   pulls in; their runtime .so siblings also serve cupy, which resolves
        #   them through the dynamic loader
        # - libcufft: cupy runtime only, no headers needed
        # - libnvjitlink: runtime dependency of the math libraries above
        # - cuda-sanitizer: the test runner's sanitize mode execs
        #   compute-sanitizer in the container
        "cuda-nvcc-${cuda_version}"
        "cuda-cudart-dev-${cuda_version}"
        "cuda-cccl-${cuda_version}"
        "cuda-nvrtc-dev-${cuda_version}"
        "cuda-nvtx-${cuda_version}"
        "libcublas-dev-${cuda_version}"
        "libcusparse-dev-${cuda_version}"
        "libcusolver-dev-${cuda_version}"
        "libcurand-dev-${cuda_version}"
        "libcufft-${cuda_version}"
        "libnvjitlink-${cuda_version}"
        "cuda-sanitizer-${cuda_version}"
    )

    apt-get install "${packages[@]}"

    # The -dev packages drag in multi-GB static archives nothing here links;
    # delete them in the same layer so the bytes never ship. nvcc's default
    # static cudart and the device runtime stay.
    # lib64 is a symlink into targets/; find does not follow it.
    find /usr/local/cuda-${CUDA_VERSION%.*}/targets/x86_64-linux/lib \
        -name '*_static*.a' \
        ! -name 'libcudart_static.a' \
        ! -name 'libcudadevrt.a' \
        -delete
EOS

# Sets up python's virtual environment
ENV PYTHON_VENV_PATH=/usr/local/venv
ENV PATH="$PYTHON_VENV_PATH/bin:$PATH"
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    # Bake the venv at /var/cache/venv-${IMAGE_VERSION}. At runtime the caller
    # mounts a per-version volume at the same sub-path; Docker auto-populates
    # a fresh volume from the image's content here, so editable pip installs
    # survive --rm and image bumps coexist (each tag gets its own volume).
    # The symlink is created first because python's venv module refuses to
    # initialize into a symlinked path, so we point it at the real target.
    ln -s /var/cache/venv-${IMAGE_VERSION} "$PYTHON_VENV_PATH"
    python3 -m venv /var/cache/venv-${IMAGE_VERSION}
EOS

# Installs all GSplat dependencies (including dev dependencies) -----------
FROM python as gsplat-deps
ARG IMAGE_VERSION

# Dependency validation: ensure Dockerfile deps stay in sync with source files.
# Aborts the build if the Dockerfile package lists diverge from
# pyproject.toml (base/dev) or examples/requirements.txt (examples).
COPY --link --chmod=755 check_deps.sh /usr/local/bin/check_gsplat_deps.sh
COPY --link --from=gsplatrepo pyproject.toml /opt/dep-check/pyproject.toml
COPY --link --from=gsplatrepo config.yaml /opt/dep-check/config.yaml
COPY --link --from=gsplatrepo gsplat/build_support/pyproject_metadata.py /opt/dep-check/gsplat/build_support/pyproject_metadata.py
COPY --link --from=gsplatrepo examples/requirements.txt /opt/dep-check/examples/requirements.txt
COPY --link --from=gsplatrepo docs/requirements.txt /opt/dep-check/docs/requirements.txt

# Main library dependencies
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    packages=(
        # Library deps
        "ninja==1.13.0"
        "numpy==2.2.6"
        "jaxtyping==0.3.4"
        "nvtx==0.2.15"
        "rich==14.3.3"
        "torch==2.9.1"
    )

    check_gsplat_deps.sh -f /opt/dep-check/pyproject.toml:dependencies -- "${packages[@]}"

    pip install "${packages[@]}"

    # Save list so the examples stage can reference it
    printf '%s\n' "${packages[@]}" > /opt/dep-check/base_packages.txt

    # To save space, we can remove the python cache files
    rm -rf $PYTHONPYCACHEPREFIX/*
EOS

# cmake — future install dependency, not yet formally declared in pyproject.toml.
RUN pip install "cmake==4.3.2" && rm -rf $PYTHONPYCACHEPREFIX/*

# Test/dev dependencies
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    clang_format_version=$(python3 /opt/dep-check/gsplat/build_support/pyproject_metadata.py \
        /opt/dep-check/pyproject.toml dev-common --pin clang-format)
    test -n "$clang_format_version"

    # Dev/test deps
    packages=(
        "packaging==26.0"
        "tomli==2.4.1"
        "black[jupyter]==22.3.0"
        "clang-format==${clang_format_version}"
        "isort==5.10.1"
        "pylint==2.13.4"
        "pytest==8.3.5"
        "pytest-check==2.8.0"
        "pytest-env==0.8.1"
        "pytest-xdist==2.5.0"
        # endonerf / dynamic-surgical test collection imports these at module top
        "Pillow==12.1.1"
        "tqdm==4.67.3"
        "typeguard==4.5.1"
        "pyyaml==6.0.1"
        "build==1.3.0"
        "twine==1.6.0"
        "cupy-cuda${CUDA_VERSION%%.*}x==13.6.0"
        "nerfacc==0.5.3"
        "imageio[ffmpeg]==2.37.2"
        "torchpq==0.3.0.6"
    )
    plas_pkg="PLAS @ git+https://github.com/fraunhoferhhi/PLAS.git"

    check_gsplat_deps.sh \
        -f /opt/dep-check/pyproject.toml:dev-cuda${CUDA_VERSION%%.*} \
        -- "${packages[@]}" "$plas_pkg"

    # The CUDA-major-specific CuPy wheel avoids compiling CuPy from source.
    pip install "${packages[@]}"

    # Install PLAS alone, then its declared dependencies read from its own
    # metadata — minus the GUI opencv-python, whose cv2 the pinned headless
    # build already provides. Deriving the list keeps it in sync with PLAS.
    pip install --no-deps "$plas_pkg"
    python3 - <<'PY' > /tmp/plas-deps.txt
from importlib.metadata import requires

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

for raw in requires("PLAS") or []:
    requirement = Requirement(raw)
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    if canonicalize_name(requirement.name) == "opencv-python":
        continue
    print(requirement)
PY
    pip install -r /tmp/plas-deps.txt

    # Save list so the examples stage can reference it
    printf '%s\n' "${packages[@]}" "$plas_pkg" > /opt/dep-check/dev_packages.txt

    # To save space, we can remove the python cache files
    rm -rf $PYTHONPYCACHEPREFIX/*
EOS

# Examples dependencies
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    # Load base + dev package lists saved by previous stages
    mapfile -t base_pkgs < /opt/dep-check/base_packages.txt
    mapfile -t dev_pkgs < /opt/dep-check/dev_packages.txt

    # Compile fused-ssim for the GPU generations supported by this image.
    # sm_80: Ampere/Tesla ga100
    # sm_89: Ada/GeForce ad102
    # sm_90: Hopper/Tesla gh100
    # sm_120: Blackwell/GeForce gb20x
    fused_ssim="git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"

    # Examples
    packages=(
        "nvidia-ncore==19.0.0"
        "torchvision==0.24.1"
        "pycolmap==3.10.0"
        "viser==1.0.24"
        "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
        "scipy==1.17.1"
        "scikit-learn==1.8.0"
        "torchmetrics[image]==1.8.2"
        "tyro==1.0.8"
        "piexif==1.1.3"
        "tensorboard==2.20.0"
        "tensorly==0.9.0"
        "git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16"
        "splines==0.3.3"
        "ppisp @ https://github.com/nv-tlabs/ppisp/archive/refs/tags/v1.0.0.tar.gz"
        "opencv-python-headless==4.10.0.84"
        "tqdm==4.67.3"
        "matplotlib==3.10.8"
        "Pillow==12.1.1"
    )

    check_gsplat_deps.sh \
        -f /opt/dep-check/pyproject.toml:dependencies \
        -f /opt/dep-check/pyproject.toml:dev-cuda${CUDA_VERSION%%.*} \
        -f /opt/dep-check/pyproject.toml:examples \
        -f /opt/dep-check/examples/requirements.txt \
        -- "${base_pkgs[@]}" "${dev_pkgs[@]}" "$fused_ssim" "${packages[@]}"

    # Build fused_ssim first (needs special env var)
    CUDA_ARCHITECTURES='80;89;90;120' \
        pip install --no-build-isolation "$fused_ssim"

    # Needs --no-build-isolation or else fused-ssim can't be installed as
    # it needs torch in its setup.py.
    pip install --no-build-isolation "${packages[@]}"

    # Save combined list for entrypoint runtime check
    printf '%s\n' "${base_pkgs[@]}" "${dev_pkgs[@]}" \
        "$fused_ssim" "${packages[@]}" > /opt/dep-check/all_packages.txt

    # To save space, we can remove the python cache files
    rm -rf $PYTHONPYCACHEPREFIX/*
EOS

# Docs dependencies
RUN --mount=type=tmpfs,target=/tmp \
<<EOS
    packages=(
        "furo==2025.12.19"
        "sphinx==9.1.0"
        "sphinx-copybutton==0.5.2"
        "sphinx-design==0.7.0"
        "sphinxcontrib-bibtex==2.7.0"
        "sphinxcontrib-video==0.4.2"
    )

    check_gsplat_deps.sh -f /opt/dep-check/docs/requirements.txt -- "${packages[@]}"

    pip install "${packages[@]}"

    rm -rf $PYTHONPYCACHEPREFIX/*

    # World-writable venv, applied in this stage so the main-image COPY carries
    # the modes without re-chmoding:
    # - any host UID mapped into the container can install into it at runtime
    # - dev-only image; do NOT use in production
    chmod -R a+w /var/cache/venv-${IMAGE_VERSION}
EOS

# Test asset ------------------------------------------------------------
# PandaSet scene 019 (240x135) test asset, served as a prebuilt registry
# image so the build pulls the layer instead of downloading and converting
# the dataset at build time.
FROM ${DOCKER_REGISTRY}/assets/pandaset:019-240x135 AS gsplat-assets

# Producer for the asset image above, kept out of the main build. The scene
# is fixed; build the pandaset-asset target and push it to the ref above to
# (re)publish the asset.
FROM gsplat-deps AS pandaset-asset-builder
RUN --mount=type=tmpfs,target=/tmp \
    --mount=type=bind,from=gsplatrepo,source=examples/prepare_pandaset.py,target=/tmp/prepare_pandaset.py \
<<EOS
    python /tmp/prepare_pandaset.py \
        --download --scene 019 --resolution 240x135 \
        --output /test_pandaset.npz
EOS

FROM scratch AS pandaset-asset
COPY --from=pandaset-asset-builder /test_pandaset.npz /test_pandaset.npz

# Main image ============================================================
# It combines the Python environment, development toolchain
# and all gsplat dependencies into an image.

FROM python as main
ARG IMAGE_VERSION

# Copy the populated venv via its real path. With --link, copying onto
# $PYTHON_VENV_PATH/ (the symlink created in the python stage) would shadow
# the symlink with a fresh dir layer, so we copy to the versioned path and
# leave the inherited symlink intact.
COPY --link --from=gsplat-deps /var/cache/venv-${IMAGE_VERSION}/ /var/cache/venv-${IMAGE_VERSION}/
COPY --link --from=gsplat-deps /opt/dep-check/ /opt/dep-check/
COPY --link --from=gsplat-assets /test_pandaset.npz /opt/gsplat-assets/test_pandaset.npz
COPY --link --from=ccache ccache /usr/local/bin/ccache
COPY --link --chmod=755 check_deps.sh /usr/local/bin/check_gsplat_deps.sh

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
        # SSH server for persistent dev container access
        openssh-server
    )
    apt-get install "${packages[@]}"
EOS

ENV CCACHE_CONFIGPATH=/etc/ccache.conf
ENV PATH=/usr/local/bin:$PATH

COPY --link profile.d/00-persistent-cache.sh /etc/profile.d/
COPY --link sshd_config.d/gsplat.conf /etc/ssh/sshd_config.d/

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

    # The venv contents already carry a+w from the gsplat-deps stage; the
    # COPY preserves those modes but creates the top-level directory itself
    # as root:755, so open just the venv root here.
    chmod a+w /var/cache/venv-${IMAGE_VERSION}

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

    # SSH server prerequisites
    ssh-keygen -A
    mkdir -p /run/sshd

    # Remove all non-system users/groups (UID/GID >= 1000, except nobody/nogroup)
    # so the entrypoint can create the host user without conflicts
    nogroup_gid="$(getent group nogroup | cut -d: -f3)"
    awk -F: -v ng="$nogroup_gid" '$3 >= 1000 && $1 != "nobody" && $4 != ng { print $1 }' /etc/passwd \
        | xargs -r -n1 userdel 2>/dev/null || true
    awk -F: '$3 >= 1000 && $1 != "nogroup" { print $1 }' /etc/group \
        | xargs -r -n1 groupdel 2>/dev/null || true

    # Persist env vars needed for dev work into SSH sessions via /etc/environment
    # (PAM reads this file for all login sessions, including SSH)
    for var in PYTHONPATH PATH PYTHON_VENV_PATH CCACHE_CONFIGPATH; do
        eval "val=\${$var:-}"
        if [ -n "$val" ]; then
            printf '%s="%s"\n' "$var" "$val" >> /etc/environment
        fi
    done
EOS

# Plain COPY (no --link) so the destination resolves through the
# /usr/local/cuda symlink (which the base image points at the actual
# versioned CUDA install dir) and the wrapper lands at the real bin
# directory next to the renamed nvcc-real. With --link the COPY layer
# creates a fresh /usr/local/cuda/ directory that shadows the symlink,
# leaving nvcc-real unreachable at runtime.
COPY cuda/nvcc_wrapper.sh /usr/local/cuda/bin/nvcc
COPY --chmod=755 entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash", "--login"]

# Our local, persistent cache volume
VOLUME /var/cache

# Viser is exposed via this port
# Used in examples.
EXPOSE 8080/tcp

# SSH server for persistent dev container access
EXPOSE 22/tcp

WORKDIR /root
