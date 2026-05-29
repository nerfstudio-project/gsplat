Repo-local Python packages used by `gsplat`. Each subdirectory is its own
installable package. See `design.md` for the shared architectural contract
(`functional/` + `kernels/` layering, CUDA layout, testing rules).

## Packages

| Path             | Distribution name | Import name        | Depends on                |
| ---------------- | ----------------- | ------------------ | ------------------------- |
| `libs/geometry`  | `gsplat-geometry` | `gsplat_geometry`  | `torch`                   |
| `libs/scene`     | `gsplat-scene`    | `gsplat_scene`     | `torch`                   |
| `libs/stage`     | `gsplat-stage`    | `gsplat_stage`     | `torch`, `gsplat-scene`   |

## Build and Install

Install from an environment that already has the repo's expected PyTorch/CUDA
toolchain. The libraries are editable installs, so Python-only edits are picked
up immediately by the next `python` or `pytest` process.

```bash
cd libs
./install.sh                # lists supported packages
./install.sh geometry
./install.sh scene
./install.sh stage          # install AFTER scene
```

Each call installs one package into the active Python environment. Equivalent
manual commands are:

```bash
pip install -e geometry
pip install -e scene
pip install -e stage
```

Native/CUDA extensions are built or loaded through the package-specific kernel
layers. `gsplat-scene` expects the scene CUDA extension to be available through
the active PyTorch/CUDA setup, with fallback JIT build behavior handled by
`gsplat_scene.kernels`. Re-run the relevant editable install after changing
`pyproject.toml`, adding top-level packages, switching Python environments, or
modifying native/CUDA extension sources.

To verify what's installed and where it points:

```bash
pip show gsplat-geometry gsplat-scene gsplat-stage | grep -E 'Name|Editable'
```

The `Editable project location:` line should point back into this repo.

## Developing

Installs are editable (`pip install -e`), so normal development should happen in
place under `libs/<pkg>/...`. Keep dependency order in mind when reinstalling:
`stage` depends on `scene`.

## Testing

Tests are colocated with the layer they exercise (typically alongside
`functional/` or `components/`). Run them through the active env's `pytest`:

```bash
pytest libs/geometry/functional
pytest libs/scene/components libs/scene/functional libs/scene/test_package_imports.py
pytest libs/stage/components
```

See `design.md` for the full testing contract.

## Sanity Check

After installing, this should print workspace paths for all libs:

```bash
python - <<'PY'
import gsplat_geometry, gsplat_scene, gsplat_stage
for m in (gsplat_geometry, gsplat_scene, gsplat_stage):
    print(m.__name__, "->", m.__file__)
PY
```

If any path points into `site-packages/...` instead of this repo, the package
was installed non-editable; reinstall with `./install.sh <pkg>`.
