# GSplat Docker Infrastructure

This directory contains Docker infrastructure to define and build
the Docker container used for GSplat development and testing.

For details on using the container to run tests and access development
shells, see [`tests/README.md`](../tests/README.md).

## Dockerfile Structure

The Dockerfile uses a multi-stage build composed of multiple layers.
This layered approach optimizes build times, image size, and caching
efficiency.

### Layer Organization Criteria

Layers are organized based on these principles:

1. **Change Frequency**: Rarely-changing layers (base images,
   system packages) come before frequently-changing ones (source code)
2. **Build vs Runtime**: Separates build-time dependencies from
   runtime requirements
3. **Cache Optimization**: Groups operations that should be cached
   together
4. **Size Efficiency**: Intermediate build artifacts are kept in
   separate stages and not carried to the final image

### Build Stages

1. **`ccache` stage**: Downloads and installs ccache for build
   acceleration
2. **`python` stage**: Base layer with Python, apt configuration,
   and virtual environment setup
3. **`gsplat-deps` stage**: Installs all GSplat dependencies
   (including dev dependencies)
4. **`main` stage**: Final image combining Python environment,
   dependencies, ccache, and development tools

This structure ensures that dependency installation (the most
time-consuming step) is cached effectively and only rebuilt when
`setup.py` changes.

## Building the Image Locally

The `build_image.sh` script builds the Docker image from the Dockerfile.

### Prerequisites

The script requires the following tools to be installed:
- `yq`: YAML processor for reading config.yaml
- `docker`: Docker engine with buildx support
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

**⚠️ IMPORTANT**: If another developer is working on a Dockerfile update
theirselves, some coordination is required to define the container version each
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

### Example

```bash
# Commit the changes
git add ../config.yaml Dockerfile
git commit -m "Update GSplat Docker image to version 43

- Updated CUDA to version X.Y
- Added dependency Z for feature W
- Fixed issue with build caching"

# Push to GitLab
git push origin your-branch-name
```

