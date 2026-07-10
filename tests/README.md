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
  and targeted testing. The selection applies at configure time and persists
  in the preset's build tree: passing feature flags reconfigures the tree for
  exactly those families, `--all` restores every family, and runs without
  feature flags keep the tree's current selection (a fresh tree builds all
  families).
- **Mounted Source Code**: Local source code is mounted at the same directory as
  on the host. The script incrementally rebuilds changed native sources, and
  debug symbols refer to the correct host paths. Linked Git worktrees also
  mount their external common Git directory read-only, so source-level Git
  checks behave the same way inside and outside the container.
- **Host-Visible Build Trees**: The container configures its CMake trees inside
  the repository at `build/docker/<preset>`. Because the source is
  mounted at its host path, the host sees the exact tree the container writes:
  build artifacts, `compile_commands.json`, and CTest logs are directly
  inspectable without entering the container. These directories belong to the
  container toolchain — do not configure them with a host `cmake` (host-native
  builds keep using the preset `build/<preset>` directories). Each tree records
  the Docker image tag that configured it; after an image upgrade the script
  reconfigures the tree from scratch automatically.
- **Separable Steps**: `--configure`, `--build`, and `--test` (the default)
  select how far the run goes; each step includes the previous ones, and
  incremental re-runs are cheap.
- **Shared ccache**: When the host has a ccache cache directory, it is mounted
  into the container so host and container builds share one cache budget;
  otherwise a per-host Docker volume provides the cache.

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
  (`--sanitize`) to detect memory errors
- **Debug Mode Support**: Enable debug builds with `--debug` (shorthand for
  `--preset=dev-debug`) for additional diagnostic information and symbol
  visibility
- **Verbose Output Mode**: Display intermediate build information and docker
  invocation details with `--verbose` flag for troubleshooting
- **CTest Integration**: CTest runs both the C++ and Python suites. Pass CTest
  arguments directly to select suites, labels, or verbosity.

### Flexibility & Configuration

- **GPU Device Selection**: Specify which GPUs to use for testing
  (`--gpus=<filter>`) when multiple GPUs are available
- **Persistent Environment Configuration**: Set runner flags once via
  `GSPLAT_TEST_PARAMS` for repeated use. Pass CTest or shell-command arguments
  on the command line.
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

# Only configure the build tree, or configure + build without testing
./run_tests.sh --configure
./run_tests.sh --build

# Use a specific CMake configure preset instead of dev-release
./run_tests.sh --preset=full-release --build

# Build only specific features
./run_tests.sh --3dgut
./run_tests.sh --3dgs --2dgs

# Run only the Python core suite
./run_tests.sh -R '^python_core$'
./run_tests.sh --3dgut -R '^python_core$'

# Show verbose CTest output for one registered suite
./run_tests.sh -V -R '^python_sensors$'

# Use the 2nd gpu in the system
./run_tests.sh --gpus=device=1

# Define the test parameters once in the environment variable,
# useful if it needs to be given every time.
export GSPLAT_TEST_PARAMS='--gpus=device=1 --3dgut'
./run_tests.sh

# Run one registered suite under CUDA compute-sanitizer
./run_tests.sh --sanitize --3dgut -R '^python_core$'

# Combine flags for detailed troubleshooting
./run_tests.sh --debug --verbose --3dgut -R '^python_core$'
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
- A configured and built CMake tree in `GSPLAT_BUILD_DIR`.
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

# Run a test under a debugger, using the generated pytest configuration
python -m pdb -m pytest -c "$GSPLAT_BUILD_DIR/pytest.ini" \
    tests/core/test_rasterization.py -k testname -s

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

## Persistent SSH Dev Container

For workflows that require running multiple commands inside the container without the overhead of a new container per command, you can start a persistent container with an SSH server.

This is particularly useful when:
- **AI coding agents** (Claude Code, Cursor, etc.) need to perform multi-step tasks — build, run tests, inspect output, iterate — inside the project environment. The agent connects over SSH and issues commands directly, maintaining state between steps. The container also acts as a safety boundary: the agent operates in an isolated environment with only the project mounted, limiting the blast radius of unintended operations on the host.
- **Interactive debugging sessions** where you need to keep a partially-built state or a running process alive between commands.
- **Remote development** on a headless machine where you want a long-lived environment you can reconnect to.

To start a persistent container with an SSH server:

```bash
# Start container with SSH on port 2222 (default)
./run_tests.sh --ssh

# Start container with SSH on a custom port
./run_tests.sh --ssh=2200
```

When no command is given, the container stays alive until you stop it with `Ctrl-C`.

Connect from the host:

```bash
ssh -p 2222 $USER@localhost
```

Combine with feature flags to pre-configure the build environment:

```bash
./run_tests.sh --ssh --3dgut
```

**Requirements:** `~/.ssh/authorized_keys` must exist on the host (key-based auth only; password auth is disabled). Only this file is bind-mounted into the container.

**Note:** After rebuilding the Docker image, the SSH host keys will change. Run `ssh-keygen -R '[localhost]:2222'` on the host to clear the old entry from `~/.ssh/known_hosts`.

## Running on memory-constrained machines

Some batched rasterization tests can use several GB of GPU memory at peak. On
hosts with 8 GB or less of dedicated VRAM, the PyTorch CUDA caching allocator
can spill into shared/system memory, which is orders of magnitude slower and
stalls the entire host.

The conftest installs a session-scoped CUDA-memory cap at **100% of the VRAM
that is actually free after torch's CUDA context has initialized** (queried
via `torch.cuda.mem_get_info(...)`). Sizing off *free* memory rather than
*total* memory means the cap automatically accounts for driver / cuBLAS /
cuDNN overhead (~1 GB on modern stacks) and for any other process already
using the GPU at session start. The cap turns would-be spills into the
shared/system memory pool into a clean `OutOfMemoryError`, so the host stays
responsive. Lower the fraction via the `--cuda-mem-fraction` CLI flag when
other processes on the box need headroom:

```bash
# Tighten to 80% of free (busy host with other GPU consumers):
python -m pytest -c "$GSPLAT_BUILD_DIR/pytest.ini" \
    --cuda-mem-fraction=0.80 -sv

# Tighten further to 50% of free (heavily contended GPU):
python -m pytest -c "$GSPLAT_BUILD_DIR/pytest.ini" \
    --cuda-mem-fraction=0.50 -sv
```

Two additional host-side knobs help without changing what is tested:

```bash
# 1. Use the expandable-segments allocator. Greatly reduces fragmentation
#    across long parametrize sweeps.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Run the test files separately so the allocator pool resets between them.
python -m pytest -c "$GSPLAT_BUILD_DIR/pytest.ini" \
    tests/core/test_basic.py -sv
python -m pytest -c "$GSPLAT_BUILD_DIR/pytest.ini" \
    tests/core/test_rasterization.py -sv
python -m pytest -c "$GSPLAT_BUILD_DIR/pytest.ini" \
    tests/geometry/functional -sv
```

These knobs are pure runtime hygiene — they do not deselect or shrink any
test. CI does not need them and is unaffected (CI GPUs have ample headroom
under the 100%-of-free cap).

### Per-test peak memory tracking

Per-test memory tracking is **opt-in** via the `--mem-track` pytest flag.
When omitted, default pytest output is unchanged and the tracking fixture
is a no-op (no sampler thread, no overhead). Pass the flag to enable it:

```bash
python -m pytest -c "$GSPLAT_BUILD_DIR/pytest.ini" --mem-track -sv
```

When enabled, two peaks are recorded for every test:

- **`torch_peak`** — bytes managed by torch's caching allocator at the
  moment of peak (`torch.cuda.max_memory_allocated`). Tightly attributed
  to the test's torch operations.
- **`device_peak`** — total VRAM in use as seen by the CUDA driver,
  sampled from a background thread at ~20 Hz. Includes context overhead
  and any other CUDA allocator on the same device.

At session end pytest prints a sorted summary of the heaviest tests. The values
and node ID in this format-only example are schematic:

```
================ CUDA memory peaks (top <shown> of <total> tests) ================
session max:  device=<device_max> MiB   torch=<torch_max> MiB
   device_peak    torch_peak   test
    <device> MiB    <torch> MiB   tests/...::test_name[param]
    ...
```

`--mem-track` also enables the cap-fixture's startup announcement, so the
chosen cap (and the GPU it was sized against) is logged alongside the
per-test peaks.

Knobs (CLI flags; only meaningful when `--mem-track` is on):

| Flag | Default | Effect |
| --- | --- | --- |
| `--mem-track-interval=SECONDS` | `0.05` | Sampler poll interval. Lower = catches shorter spikes, more overhead. |
| `--mem-track-top=N` | `25` | How many tests to print in the summary. |
| `--mem-track-csv=PATH` | unset | Path to dump full per-test peaks as CSV for offline analysis. |

Run
`python -m pytest -c "$GSPLAT_BUILD_DIR/pytest.ini" --help | grep -A1 'gsplat-mem'`
to see the full list with descriptions.

### Live external monitoring

For a real-time view of GPU memory while a long pytest run executes (e.g.
to spot when the host begins to crawl), run one of these in a separate
terminal:

```bash
# Compact one-line-per-second update — best for tailing.
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
           --format=csv,noheader -l 1

# Curses-style dashboard, useful for graphing trends.
nvidia-smi dmon -s mu

# Or refresh the full table every second.
watch -n 1 nvidia-smi
```

These show **dedicated VRAM only** — they don't include the Windows
"Shared GPU memory" pool. That's intentional: with the 100%-of-free cap
in place, anything that would have spilled into shared memory now OOMs
cleanly inside torch instead, and `torch_peak` from the per-test summary
will show which test hit the limit.

If you specifically need to see Windows shared-GPU-memory consumption (to
confirm nothing spilled), open Windows Task Manager → Performance → GPU,
or from PowerShell on the host:

```powershell
Get-Counter '\GPU Process Memory(*)\Shared Usage' -SampleInterval 1 -Continuous
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
