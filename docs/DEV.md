# Development

## Set up the development environment

Clone the repository and submodules with

```bash
git clone --recurse-submodules URL
```

Install an [NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
supported by PyTorch. Activate an existing Python environment, then run:

```bash
./bootstrap.sh
```

This installs the development extra. The CuPy wheel matching the local CUDA
toolkit is selected automatically. On Python 3.13, `dev` skips the
PNG-compression extra (`vc-flas` doesn't support 3.13 yet); install
`gsplat[png]` separately on 3.10-3.12 if you need it.

If bootstrap cannot detect `nvcc`, or to choose the CUDA-tagged Python
dependencies (PyTorch wheel index, CuPy package) explicitly, run:

```bash
./bootstrap.sh --cuda 12.8
```

This selects binary dependencies only; it does not choose the CUDA compiler
used by the build. Bootstrap warns when the toolkit it detects disagrees with
the requested version.

To have bootstrap create a new virtual environment, run:

```bash
./bootstrap.sh --venv /path/to/venvs/gsplat --cuda 12.8
source /path/to/venvs/gsplat/bin/activate
```

Use `./bootstrap.sh --help` to see the Python and CUDA options.

Bootstrap installs the complete development and test dependency set, including
the optional PNG compression feature. It selects CuPy from the requested or
detected dependency CUDA major, rather than from the GPU driver's maximum
supported version. A later CMake configuration verifies that the build
compiler and Torch use the same CUDA major. The `vc-flas` dependency currently
limits these environments to Python 3.10 through 3.12. The base gsplat package
continues to support Python 3.13 when the `png`, `test`, and `dev` extras are not
selected.

## Develop with a CMake build tree

The CMake presets are organized by their intended use. Each preset creates a
separate persistent build tree.

### Development presets

The development presets are intended for local C++ and CUDA work:

- Use `dev-debug` by default for everyday development.
- Use `dev-release` for optimization work, profiling, and performance
  measurement.

Both presets generate machine code only for the current GPU, avoiding the cost
of compiling the complete production architecture set and the need for PTX JIT
compilation at runtime. `dev-debug` enables runtime assertions and other
debug-only checks that make bugs easier to detect, at the cost of slightly
longer compile times. `dev-release` retains debug information.

The pinned-submodule check is enabled by default
(`GSPLAT_STRICT_SUBMODULES=ON`): configuring fails when a `third_party`
submodule is checked out at a commit other than the one the gsplat project
records. Both development presets disable it, downgrading the failure to a
warning so local work on a vendored dependency is not blocked; the portable and
production presets keep it enabled.

Changes should normally pass both `dev-debug` and `dev-release` before
submission because optimization can expose issues that are not visible in a
debug build.

### Portable presets

The portable presets create broadly compatible builds at relatively low build
cost:

- Use `debug` for a portable debug build.
- Use `release` for a portable optimized build.

Both presets compile only compute-80 PTX, without architecture-specific
cubins. A compatible CUDA driver can JIT this PTX for any supported GPU with
compute capability 8.0 or newer. Compiling a single virtual architecture keeps
these builds relatively inexpensive. The tradeoff is deferred compilation:
for each newly built binary, the CUDA driver JIT-compiles the PTX when its
kernels are first loaded, adding a one-time runtime cost that is cached for
subsequent runs.

### Production preset

The production preset creates distributable artifacts and validates the full
supported GPU architecture set:

- Use `full-release` for optimized distributable builds and release wheels.

This preset compiles every supported architecture as well as compute-80 PTX.
It takes longer to build, but verifies that every kernel compiles across the
complete architecture set and produces wheels that avoid JIT compilation on
those architectures.

### Configure, build, and test

By default, a preset creates its build tree under `<repo>/build/<preset>`. Pass
`-B` while configuring to use a different directory:

```bash
cmake --preset dev-debug -B /path/to/gsplat-build
```

After choosing a preset, configure, build, and test from its build directory.
For example, using `dev-debug` and its default directory:

```bash
cmake --preset dev-debug
cd build/dev-debug
ninja
ctest --output-on-failure
```

`cmake` configures the project and creates `build/dev-debug`. From that
directory, `ninja` compiles the C++ and CUDA code. `ctest --output-on-failure`
is the recommended test entry point because it runs both the registered C++
and Python tests and shows output from failures. To use another preset, replace
`dev-debug` with its name in the first two commands.

If multiple CUDA toolkits are installed and the default is not the intended
one, select its compiler when first configuring the build tree:

```bash
cmake --preset dev-debug \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
```

Run pytest directly only when you want the Python suite or a focused Python
test; pytest does not execute the C++ tests:

```bash
cd build/dev-debug
python -m pytest
python -m pytest ../../tests/core/test_basic.py
```

### Adding source files

New C++/CUDA implementation files must be registered explicitly in the source
lists of `gsplat/cuda/csrc/CMakeLists.txt` (and the other kernel package
`CMakeLists.txt` files); build sources are deliberately not globbed, so a file
that is not listed is silently left out of the build. The explicit lists keep
stray or work-in-progress files from being compiled in by mistake.

### CMake cache options

Pass project-specific options while configuring with `-D<name>=<value>`.
Boolean values accept CMake's usual `ON` and `OFF` spellings. The defaults below
apply to a top-level build unless noted otherwise; presets can override them.

#### Build content

| Option | Default | Purpose |
| --- | --- | --- |
| `GSPLAT_KERNEL_FAMILIES` | empty (all) | Select a comma- or semicolon-separated subset of `2DGS`, `3DGS`, `3DGUT`, `ADAM`, `RELOC`, and `LOSSES` to compile. |
| `GSPLAT_NUM_CHANNELS` | all supported widths | Select the comma- or semicolon-separated feature widths to instantiate; `NUM_CHANNELS` remains a compatibility alias. |
| `GSPLAT_FAST_MATH` | `ON` | Compile CUDA kernels with fast-math intrinsics. |
| `GSPLAT_GENERATED_DIR` | `<build>/generated` | Select the directory for generated gsplat build headers. |

#### Tests

| Option | Default | Purpose |
| --- | --- | --- |
| `GSPLAT_BUILD_TESTS` | `ON` (`OFF` as a subproject) | Build the C++ tests and install the self-contained test payload; Python source-tree tests are registered independently in top-level builds. |
| `GSPLAT_BUILD_CAMERA_WRAPPERS` | value of `GSPLAT_BUILD_TESTS` | Build the Python-exposed camera-wrapper test kernels. |
| `GSPLAT_TESTS_FORCE_CUDA` | `ON` | Require CUDA-backed pytest coverage without consulting PyTorch's availability probe. Disable it only on runners where CUDA tests should be skipped when `torch.cuda.is_available()` is false. |

#### Dependencies

| Option | Default | Purpose |
| --- | --- | --- |
| `GSPLAT_STRICT_SUBMODULES` | `ON` | Fail rather than warn when a populated `third_party` submodule differs from the commit pinned by the superproject; an absent required submodule always fails. |
| `GSPLAT_CHECK_PYTHON_DEPS` | `ON` | Check the active environment against the Python dependencies requested by the build. |
| `GSPLAT_DEVELOPMENT_MODE` | `OFF` | Require the development Python dependencies in addition to build requirements. |

#### Compile caching (ccache)

| Option | Default | Purpose |
| --- | --- | --- |
| `GSPLAT_ENABLE_CCACHE` | `ON` | Use ccache for C++, C, and CUDA compilation when it is available. |
| `GSPLAT_FORCE_CCACHE` | `OFF` | Fail configuration instead of warning when ccache is enabled but unavailable. |
| `GSPLAT_CCACHE_DIR` | empty | Override ccache's cache directory; empty preserves ccache's own resolution. |
| `GSPLAT_CCACHE_NORMALIZE_PATHS` | `ON` | Normalize source paths so compatible ccache objects can be reused across worktrees. |
| `GSPLAT_CCACHE_STATS` | `OFF` | Print cache statistics for this build after compilation. |

#### Build diagnostics

| Option | Default | Purpose |
| --- | --- | --- |
| `GSPLAT_ENABLE_BUILD_TRACES` | `OFF` | Record configure, build, and test traces; requires CMake 4.3 or newer. |

The build also honors standard CMake and toolchain cache variables such as
`CMAKE_BUILD_TYPE`, `CMAKE_CUDA_ARCHITECTURES`, `CMAKE_CUDA_COMPILER`, and
`CMAKE_CUDA_HOST_COMPILER`. `CMAKE_COMPILE_WARNING_AS_ERROR` applies the same
warnings-as-errors policy to both compilation and pytest. Run
`cmake -LAH -N <build-directory>` after configuration to inspect every cache
entry and its current value.

## Build and test wheels

Wheels are built by `pip` with scikit-build-core running a single CMake
configure and build. `cmake.args=--preset` selects the configure preset, which
stays the one source of the build configuration; `cmake.build-type` is emptied
so scikit-build-core's `Release` default cannot override the preset's
`CMAKE_BUILD_TYPE`. Clear `dist` first so every wheel there belongs to the
current build:

```bash
rm -rf dist
python -m pip wheel \
    --verbose \
    --no-build-isolation \
    --no-deps \
    --wheel-dir dist \
    --config-settings=build-dir=build/full-release \
    --config-settings=cmake.build-type= \
    --config-settings=cmake.args=--preset=full-release \
    .
```

For a portable development wheel, replace `full-release` with `debug` or
`release` in both the `build-dir` and `cmake.args` settings.

The build above includes the test payload because `GSPLAT_BUILD_TESTS` is `ON`
by default. Install its test extra to resolve the complete test environment,
including the CuPy distribution selected by the wheel's build metadata:

```bash
wheel_file=dist/WHEEL_FILENAME.whl
python -m pip install "${wheel_file}[test]"
gsplat-test
```

Wheel metadata selects CuPy from the CUDA toolkit reported by the build
environment's Torch. Configuration rejects a CUDA compiler with a different
major, so every successful wheel selects the major compiled into gsplat. The
PNG compression entry point verifies that invariant again at runtime, catching
an environment where CuPy or gsplat was replaced after installation.

The runner also supports selecting one suite and forwarding its remaining
arguments to GoogleTest or pytest:

```bash
gsplat-test cpp --gtest_filter='TorchUtils.*'
gsplat-test python -k rasterization
```

### Compiled feature widths

Set the `GSPLAT_NUM_CHANNELS` CMake cache variable to a comma- or
semicolon-separated list of positive feature widths when configuring, for
example `cmake --preset dev-release -DGSPLAT_NUM_CHANNELS=3,32`. Each entry
creates corresponding CUDA kernel specializations and increases build cost.
(`NUM_CHANNELS` is accepted as a compatibility alias.)

High-level rasterizers choose the fewest compiled widths that exactly compose
the total feature width. The largest compiled width limits one kernel launch,
not the total input width. Direct low-level `rasterize_to_pixels`,
`rasterize_to_pixels_2dgs`, and `rasterize_to_pixels_eval3d` calls still require
one exact compiled width; `rasterize_to_pixels_sparse` plans compiled-width
chunks internally.

## Protect Main Branch over Pull Request

It is recommended to commit the code into the main branch as a PR over a hard push, as the PR would protect the main branch if the code break tests but a hard push won't. Also squash the commits before merging the PR so it won't span the git history.

PR checks cover formatting, C++ and Python tests, documentation, and full CUDA
architecture compilation when build inputs change.

Because we check code formatting in CI, it is recommend to run the formatting
script before committing code; with no arguments it formats the source files
you have changed (staged or unstaged):

```bash
lint/format-code.sh
```

To format every tracked source file instead, pass `--full`.

To check formatting without modifying files, use:

```bash
lint/format-code.sh --check
```

After cloning the repository, run the bootstrap script; it installs the
pre-commit formatting hook:

```bash
./bootstrap.sh
```

The hook formats each commit automatically: it formats the staged content and
re-stages it, so the commit always lands formatted. For a partially-staged
file only the staged content is formatted; your unstaged edits are never
touched (when they overlap or sit right next to the formatting, the hook
prints a note that the
working-tree copy was left as is). A commit whose staged changes were
formatting-only is refused, since formatting reduces it to an empty commit. To
skip the hook for one commit, use `git commit --no-verify`.

Run CTest locally on the hardware you are targeting before committing. Tests
requiring unavailable hardware or external datasets skip themselves.

```bash
cd build/dev-debug
ctest --output-on-failure
```

Note that `pytest` recognizes and runs all functions named as `test_*`, so you should name the test functions in this pattern. See `test_basic.py` as an example.

## Build the Doc Locally

The documentation sources live in `docs/source`. Install the documentation
requirements in an environment containing PyTorch, then run:

```bash
python -m pip install -r docs/requirements.txt
docs/render_docs.sh build/html
```

## Clangd setup (for Neovim)

[clangd](https://clangd.llvm.org/) is a nice tool for providing completions,
type checking, and other helpful features in C++. It requires some extra effort
to get set up for CUDA development, but there are fortunately only three steps
here.

**First,** we should install a `clangd` extension for our IDE/editor.

For Neovim+lspconfig users, this is very easy, we can simply install `clangd`
via Mason and add a few setup lines in Lua:

```lua
require("lspconfig").clangd.setup{
    capabilities = capabilities
}
```

**Second,** we need to generate a `.clangd` configuration file with the current
CUDA path argument.

Make sure you're in the right environment (with CUDA installed), and then from
the root of the repository, you can run:

```sh
echo "# Autogenerated, see .clangd_template\!" > .clangd && sed -e "/^#/d" -e "s|YOUR_CUDA_PATH|$(dirname $(dirname $(which nvcc)))|" .clangd_template >> .clangd
```

**Third,** we'll need a
[`compile_commands.json`](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
file.

CMake exports this file into each top-level build directory:

```sh
cmake --preset dev-debug

# Make sure the file is not empty!
test -s build/dev-debug/compile_commands.json
```

Configure clangd or your LSP client with
`--compile-commands-dir=build/dev-debug` so it reads that database without adding a
generated file to the source root.

## Implementation details

Source builds disable pip's build isolation so CMake uses PyTorch from the
active environment. Pip still installs the selected project dependencies, but
the build requirements declared in `pyproject.toml` must already be available.

`CUDACXX` selects the CUDA compiler only when a build directory is first
configured. CMake then stores that compiler in its cache. Remove the affected
`build/<preset>` directory before configuring it with a different CUDA toolkit.

CMake stages an importable package under `build/<preset>`. This is what
allows Python tests to import the newly built extension directly from a build
tree.

The `wheel_smoke` marker selects the representative Python tests used to check
an installed wheel:

```bash
gsplat-test python -m wheel_smoke
```

For release-integrity validation, build the source distribution first and run
the wheel build from its extracted contents. This verifies that the source
distribution contains everything required for a wheel build:

```bash
rm -rf dist build/sdist-tree
mkdir -p dist build/sdist-tree
python -m build --no-isolation --sdist --outdir dist
tar \
    --extract \
    --gzip \
    --file dist/*.tar.gz \
    --directory build/sdist-tree \
    --strip-components=1
(
    cd build/sdist-tree
    python -m pip wheel \
        --verbose \
        --no-build-isolation \
        --no-deps \
        --wheel-dir dist \
        --config-settings=build-dir=build/full-release \
        --config-settings=cmake.build-type= \
        --config-settings=cmake.args=--preset=full-release \
        .
)
mv build/sdist-tree/dist/*.whl dist/
```

## Sharing ccache across worktrees

Without path normalization, ccache keeps absolute source and build paths in its
cache keys. Those paths differ between Git worktrees, so equivalent
compilations can miss the cache even when the checked-out code is substantially
the same.

Path normalization is enabled by default. Disable it when absolute path
identity is required:

```bash
cmake --preset debug -DGSPLAT_CCACHE_NORMALIZE_PATHS=OFF
```

This sets `CCACHE_BASEDIR` to the current gsplat source root and compiles with
`-ffile-prefix-map=<source-root>=.`. Source paths become relative to each
worktree's root, allowing all worktrees to reuse compatible cached objects.
Ccache's directory hashing remains enabled; the compiler mapping makes the
recorded paths location-independent instead of asking ccache to ignore them.

This optimization deliberately removes the worktree location from ccache's
identity for a compilation. Consequently:

- source paths embedded in object files or final binaries, including debug
  information and values derived from `__FILE__`, become relative rather than
  identifying the worktree that produced the final link;
- compiler diagnostics and dependency files can contain rewritten relative
  paths instead of the original absolute paths; and
- GDB resolves the normalized paths naturally when launched from the worktree
  root; otherwise add the source root with its `directory` command.

The default assumes that worktrees have equivalent relative layouts and that
their absolute location is not part of the desired output. In particular, logs
or diagnostics built from `__FILE__` do not contain an absolute checkout path.
Disable the option when absolute source provenance matters or when investigating
dependency tracking or other path-sensitive build behavior.

## Build traces

Configure with `-DGSPLAT_ENABLE_BUILD_TRACES=ON` (CMake 4.3 or newer) to
record every configure, compile, link, and test command with timings.
Each phase archives a Google Trace file into the build directory
(`configure-trace.json`, `build-trace.json`, `tests-trace.json`); open it
in a trace viewer such as [Perfetto](https://ui.perfetto.dev) or
[speedscope](https://www.speedscope.app) for a per-command parallelism
flamechart. Cache hits appear as near-zero spans; disable ccache
(`-DGSPLAT_ENABLE_CCACHE=OFF`) if you want cold-compile times. The CI build
jobs configure with the option enabled and publish their configure and build
traces. Installed-wheel test jobs invoke `gsplat-test` directly instead of
CTest, so those jobs do not create a CMake tests trace.

The pull-request checks are defined by:

- `.github/workflows/core_tests.yml`: formatting plus C++ and Python tests
  through CTest.
- `.github/workflows/cuda_architectures.yml`: full CUDA architecture compile
  validation for build-relevant changes.
- `.github/workflows/doc.yml`: documentation build.
