# GSplat Docker Infrastructure

This directory contains Docker infrastructure to define and build
the Docker container used for GSplat development and testing.

For details on using the container to run tests and access development
shells, see [`tests/README.md`](../tests/README.md).

[TOC]

## Dockerfile Structure

The Dockerfile uses a multi-stage build composed of multiple layers.
This layered approach optimizes build times, image size, and caching
efficiency.

### Layer Organization Criteria

When organizing Docker layers, use these criteria to determine what
belongs together and what should be separated:

**Put in the SAME layer if:**
- Components change together (e.g., `setup.py` and dependency
  installation always change together)
- Operations are part of a single logical step (e.g., updating
  package lists and installing packages)
- Combined size is reasonable and caching benefit is maintained
- Components have the same lifecycle (e.g., all build-time tools,
  or all runtime dependencies)

**Put in DIFFERENT layers if:**
- Components have different change frequencies (e.g., base OS
  packages vs. application source code)
- One is needed only at build-time while the other is needed at
  runtime (e.g., compilers vs. compiled binaries)
- Separating them improves cache utilization (e.g., system packages
  that rarely change vs. application dependencies that change often)
- One contains large intermediate artifacts that should not be in
  the final image (e.g., build artifacts vs. runtime binaries)

## Building the Image Locally

The `build_image.sh` script builds the Docker image from the Dockerfile.

### Prerequisites

The script was tested on the following OSs:
- Ubuntu 24.04

The script requires the following packages to be installed:
- `yq`: YAML processor for reading config.yaml
- `docker.io`: Docker engine
- `docker-buildx`: Docker BuildX support
- `nvidia-container-runtime`: NVIDIA GPU support for containers

The script will automatically check for these dependencies
and report if any are missing.

### Build Command

```bash
# Build the image with the current configuration
./build_image.sh
```

The build script automatically:
- Reads configuration from `config.yaml`
- Tags the image as `$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG`
- Builds using `docker buildx` with multi-stage optimization
- Uses build caching for faster subsequent builds

## Updating the Dockerfile

When making changes to the Dockerfile (updating dependencies,
modifying base images, changing CUDA versions, etc.), you must
follow the versioning workflow to maintain proper image tracking
and avoid accidental image overwrites.

### Steps to Update

1. Make your changes to the Dockerfile or related files
2. Increment `IMAGE_TAG` in `config.yaml`
3. Build locally (see "Building the Image Locally" section for details)
4. Test the image (see below)

### Test the Image

After building, test the image:

```bash
../tests/run_tests.sh
```

This runs the complete test suite and verifies that the Docker
image works correctly with your changes. See [`../tests/README.md`](../tests/README.md)
for more details on test options.

### Versioning Requirements

**Versioning Policy:**
- The `IMAGE_TAG` is a simple incrementing number (e.g., 42, 43, 44)
- Increment this number for ANY change to the Docker image:
  - Dependency updates in `setup.py`
  - Dockerfile modifications
  - Base image version changes
  - CUDA version changes
  - New system packages

## Uploading to the Registry

Once you've built and tested the image locally, you can push it
to the GitLab registry.

**⚠️ IMPORTANT**: When pushing a new Docker image to the registry,
you **MUST increment the `IMAGE_TAG` in `config.yaml`** if the
current tag is already used in a protected branch.

**⚠️ IMPORTANT**: If another developer is working on a Dockerfile update,
some coordination is required to define the container version each
one will use, thus avoiding accidental image overwrites.

### Push to Registry

```bash
# Push the built image to GitLab registry
./build_image.sh --push
```

**Safety Check**: The script will **fail if the image tag already
exists** in the registry. This prevents accidental overwrites of
images that may be in use.

### Force Overwrite (Use with Caution)

If you need to overwrite an existing image tag (e.g., for local
development or fixing a broken image):

```bash
./build_image.sh --push --force
```

⚠️ **Warning**: Only use `--force` when you're certain the existing
image is not being used in any protected branches, production
environments or by another active development branch.

## Committing Changes to GitLab

After successfully pushing the image to the registry, commit your
changes to the Git repository.

### Commit Requirements

**IMPORTANT**: Docker changes and corresponding GSplat code changes must be committed together:

- The Dockerfile/config changes **MUST** be in the same commit as the
  GSplat code changes that require them
- The GSplat code in the commit **MUST** build successfully with the new
  Docker image
- All tests **MUST** pass using the updated Docker image
- **DO NOT** create separate commits for Docker changes and corresponding GSplat code changes

### Commit Message Format

When committing, include the new Docker image version in the commit body,
and describe the changes made, together with the changes made in GSplat code, if any.

```bash
# Stage your changes
git add ../config.yaml Dockerfile ../setup.py

# Commit with descriptive message
git commit -m "Upgrade CUDA to X.Y

- Docker image version updated to 25
- Updated CUDA to version X.Y
- Added dependency Z for feature W
- Fixed issue with build caching"

# Push to GitLab
git push origin your-branch-name
```

