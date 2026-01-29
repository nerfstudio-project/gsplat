# GSplat Test Suite

This directory contains the GSplat test suite and test running infrastructure.

The `run_tests.sh` script is designed for faster code-build-test cycles.
You update the code locally and run the tests to see the results right away.

## Running Tests

To run the GSplat test suite:

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

# Use the 2nd gpu in the system
./run_tests.sh --gpus=device=1

# Define the test parameters once in the environment variable,
# useful if it needs to be given every time.
export GSPLAT_TEST_PARAMS='--gpus=device=1 --3dgut'
./run_tests.sh

# Run a subset of tests under CUDA compute-sanitizer
./run_tests.sh --sanitize --3dgut -k 'test_shutter_relative_frame_time'
```

### Available Flags

**Global Flags:**
- `--shell`: Enter interactive shell instead of running tests
- `--reset`: Delete the internal build cache volume
- `--sanitize`: Run tests under CUDA compute-sanitizer
- `--gpus=device=<id1,id2,...>`: enable the given GPUs inside the container
- `--gpus=<count>`: enable this many GPUs inside the container
- `--help` or `-h`: Show help message
- `--debug`: Build and run everything in debug mode.
- If `--gpus` isn't given, it defaults to all GPUs.

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

The flags used can also be defined in `GSPLAT_TEST_PARAMS` environment variable
on host. The flags given in the command line have precedence, though.

## Shell Access to Dev Container

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
