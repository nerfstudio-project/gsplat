# GSplat Docker Infrastructure

This directory contains Docker infrastructure for GSplat
development and testing.

It is designed for faster code-build-test cycles.
You update the code locally and run the tests to see the
results right away.

## 0. Prerequisites

The scripts require the following tools to be installed:
- `yq`: YAML processor for reading config.yaml
- `docker`: Docker engine with buildx support
- `nvidia-container-runtime`: NVIDIA GPU support for containers

The scripts will automatically check for these dependencies
and report if any are missing.

## 1. Running Tests

To run the GSplat test suite inside a Docker container:

```bash
./run_tests.sh
```

This script builds GSplat and runs its tests inside a Docker
container with GPU support.

The Docker image is automatically pulled from the GitLab
registry if it doesn't exist locally. The image version used
is specified in `config.yaml`.

### Basic Usage

```bash
# Show the help screen
./run_tests.sh --help

# Build all features and run all tests
./run_tests.sh

# Build only specific features
./run_tests.sh --3dgut
./run_tests.sh --3dgs --2dgs

# Run specific test files
./run_tests.sh tests/test_basic.py
./run_tests.sh --3dgut tests/test_rasterization.py

# Pass pytest arguments
./run_tests.sh -v -k "test_specific"

# Run tests in debug mode
./run_tests.sh DEBUG=1
```

### Available Flags

**Global Flags:**
- `--shell`: Enter interactive shell instead of running tests
- `--reset`: Delete the internal build cache volume
- `--help` or `-h`: Show help message

**Feature Flags:**
- `--2dgs`: Build 2DGS feature
- `--3dgs`: Build 3DGS feature
- `--3dgut`: Build 3DGUT feature
- If no feature flags are given, all features are built

The test container automatically cleans up after completion,
but the build cache volume persists for faster subsequent
builds.

Users can pass environment variables NAME=value, they will be
set inside the container.

## 2. Shell Access to Dev Container

To access an interactive shell in the development container:

```bash
./run_tests.sh --shell
```

This provides a development environment with:
- All GSplat dependencies pre-installed
- GPU access enabled
- Your local source code mounted at `/root/gsplat`
- Persistent build cache at `/root/.cache`
- Ability to manually run pytest, build features, and debug

### Example Shell Session

```bash
# Enter the container
./run_tests.sh --shell

# Inside the container, you can:
pytest -v                           # Run all tests
pytest tests/test_basic.py          # Run specific tests
pip list                            # Check installed packages
python -c "import gsplat"           # Test imports
exit                                # Leave the container
```

You can also combine `--shell` with feature flags to
pre-configure the build environment:

```bash
# Shell with only 3DGUT feature built
./run_tests.sh --shell --3dgut
```

### Running arbitrary commands

To run any command without dropping into a shell inside the container,
pass the command and its arguments after `--shell`:

```bash
# Execute a simple command, with arguments
./run_tests.sh --shell echo this is a test
# Execute a more complex bash shell pipeline.
./run_tests.sh --shell 'if command -v myprog; then myprog; else echo failure; exit 1; fi'
```

## 3. Building and Pushing the Image to GitLab

### ⚠️ IMPORTANT: Update IMAGE_TAG Before Building

**When pushing a new Docker image to the registry, you MUST
increment the `IMAGE_TAG` in `config.yaml` if the current tag
is already used in a protected branch.** For local builds and
testing, bumping the tag is optional; the push safety check
already prevents overwriting existing tags unless `--force` is used.

**Versioning:** The `IMAGE_TAG` is a simple incrementing
number. When you make changes to the Docker image (update
dependencies, modify Dockerfile, etc.), increment this number.

The build script includes a safety check: it will **fail if
the image tag already exists** in the registry (unless you
explicitly use `--force` to overwrite). This prevents
accidental overwrites of production images.

You should update the tag before pushing to:
- Maintain proper version history
- Avoid confusion about which version contains which changes
- Enable rollback to previous versions if needed
- Follow proper release management practices

### Build and Push

```bash
# Build locally only
./build_image.sh

# Build and push to GitLab registry
./build_image.sh --push

# Force overwrite if image already exists in registry
./build_image.sh --push --force
```

The script automatically:
- Reads configuration from `config.yaml`
- Tags the image as `$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG`
- Builds using `docker buildx`
- Checks if the image already exists in the registry
- Pushes to registry if `--push` flag is given
- **Fails if image already exists** unless `--force` is used

This safety check prevents accidentally overwriting existing
images in the registry.

### Complete Workflow Example

```bash
# 1. Update the image tag in config.yaml
# Change IMAGE_TAG: 42 -> 43
vim ../config.yaml

# 2. Build the image locally
./build_image.sh

# 3. Test the image
./run_tests.sh

# 4. If tests pass, push to registry
./build_image.sh --push

# Note: If the image version already exists in the registry,
# the script will fail. Use --force to overwrite:
# ./build_image.sh --push --force

# 5. Commit the config.yaml and Dockerfile changes
git add ../config.yaml Dockerfile
git commit -m "Update GSplat Docker image to version 43"
```

