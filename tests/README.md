# GSplat Test Suite

This directory contains the GSplat test suite and test running infrastructure.

The `run_tests.sh` script is designed for faster code-build-test cycles.
You update the code locally and run the tests to see the results right away.

[TOC]

## Key Features

### Fast Development Iterations

- **Accelerated Build Performance**: Persistent build cache via
  [ccache](https://ccache.dev/) enables faster development iteration by
  caching compiled objects between builds
- **Selective Feature Building**: Configure build scope to include only
  specific features (e.g., `--3dgut`, `--3dgs --2dgs`) for faster compilation
  and targeted testing
- **Mounted Source Code**: Local source code mounted at the same directory as
  on host allows immediate testing of code changes without rebuild delays,
  and debug symbols refers to the correct source files on host.

### Reproducibility

- **Consistent Test Environment**: Identical Docker container used locally
  and in CI pipelines ensures reproducible test results across all
  environments
- **Automated Container Management**: Docker images automatically pulled from
  GitLab registry based on commit-specific versions defined in `config.yaml`

### Local Debugging & Testing

- **Interactive Development Shell**: Launch an interactive shell inside the
  container (`--shell`) with GPU access, mounted source code, and all
  dependencies pre-installed; ideal for running tests under debuggers like
  `pdb`, `ipdb`, or `gdb`
- **CUDA Debugging Support**: Run tests under CUDA compute-sanitizer
  (`--sanitize`) to detect memory errors and race conditions
- **Debug Mode Support**: Enable debug builds with `--debug` flag for
  additional diagnostic information and symbol visibility
- **Verbose Output Mode**: Display intermediate build information and docker
  invocation details with `--verbose` flag for troubleshooting
- **Direct pytest Integration**: Pass pytest arguments directly to filter
  tests, adjust verbosity, or configure test execution (`-v`, `-k`, `-x`, etc.)

### Flexibility & Configuration

- **GPU Device Selection**: Specify which GPUs to use for testing
  (`--gpus=<filter>`) when multiple GPUs are available
- **Persistent Environment Configuration**: Set test parameters once via
  `GSPLAT_TEST_PARAMS` environment variable for repeated use
- **Arbitrary Command Execution**: Run any command inside the containerized
  environment without dropping into an interactive shell

## Running Tests

To run the GSplat test suite:

```bash
./run_tests.sh
```

This script builds GSplat and runs its tests inside a Docker
container with GPU support.

The Docker image is automatically pulled from the GitLab
registry if it doesn't exist locally. The image version used
is specified in `../config.yaml` and is associated with
the current commit.

### Basic Usage

To show the full list of parameters accepted, run:

```bash
./run_tests.sh --help
```

Here are some common usage examples:

```bash
# Build all features and run all tests
./run_tests.sh

# Build only specific features
./run_tests.sh --3dgut
./run_tests.sh --3dgs --2dgs

# Run specific test files
./run_tests.sh tests/test_basic.py
./run_tests.sh --3dgut tests/test_rasterization.py

# Filter tests to be executed.
./run_tests.sh -v -k "test_specific"

# Use the 2nd gpu in the system
./run_tests.sh --gpus=device=1

# Define the test parameters once in the environment variable,
# useful if it needs to be given every time.
export GSPLAT_TEST_PARAMS='--gpus=device=1 --3dgut'
./run_tests.sh

# Run a subset of tests under CUDA compute-sanitizer
./run_tests.sh --sanitize --3dgut -k 'test_shutter_relative_frame_time'

# Combine flags for detailed troubleshooting
./run_tests.sh --debug --verbose --3dgut -k 'test_specific'
```

## Shell Access to Dev Container

To access an interactive shell in the development container:

```bash
./run_tests.sh --shell
```

This provides a development environment with:
- All GSplat dependencies pre-installed
- GPU access enabled
- Your local source code mounted at the same directory inside the container.
- Persistent build cache at `/var/cache/ccache`
- Ability to manually run pytest with debuggers attached.
- Run examples and benchmark scripts.
- sudo access

The container automatically cleans up after completion, but the /var/cache
directory inside the container persists for faster subsequent builds.

Any other parameters given are interpreted as an executable to be executed
inside the container, along with its parameters.

### Example Shell Session

```bash
# Enter the container
./run_tests.sh --shell

# Check installed packages
pip list

# Test imports
python -c "import gsplat"

# Run a test under a debugger
python -m pdb -m pytest test_rasterization.py -k testname -s

# Leave the container
exit
```

You can also combine `--shell` with feature flags to
pre-configure the build environment:

```bash
# Shell with only 3DGUT feature built
./run_tests.sh --shell --3dgut
```

## Running examples and benchmarks

### Pre-requisites

Download the required dataset.

```bash
./run_tests.sh --shell python examples/datasets/download_dataset.py --dataset=mipnerf360
```

Depending on the example or benchmark script, other datasets are required.
For the list of available datasets to download, run:

```bash
./run_tests.sh --shell python examples/datasets/download_dataset.py --help
```

### Simple trainer
```bash
./run_tests.sh --shell python examples/simple_trainer.py mcmc
```

### Basic benchmark
```bash
./run_tests.sh --shell examples/benchmarks/basic.sh
```

### Visualization

Some examples (e.g. `simple_viewer.py`) launch a [viser](https://viser.studio) web viewer on
port 8080 inside the container. Use `--listen` (or its short form `-p`) to forward that port to the host:

```bash
./run_tests.sh --listen=8080 --shell python examples/simple_viewer.py
# or equivalently
./run_tests.sh -p 8080 --shell python examples/simple_viewer.py
```

Then open `http://localhost:8080` in your browser.

If the machine running Docker is **remote**, forward the port over SSH before opening the browser:

```bash
ssh -NL 8080:localhost:8080 <remote-host>
```

## Troubleshooting

### docker: Error response from daemon: driver failed programming external connectivity on endpoint ...: Bind for 0.0.0.0:8080 failed: port is already allocated.

**Problem:** Another process on the host is already listening on port 8080, so Docker cannot bind to it.

**Solution:** Choose a different host port using `--listen=HOST:CONTAINER`:

```bash
./run_tests.sh --listen=8090:8080 --shell python examples/simple_viewer.py
```

Then open `http://localhost:8090` in your browser. If using an SSH tunnel, adjust the port accordingly:

```bash
ssh -NL 8090:localhost:8090 <remote-host>
```
