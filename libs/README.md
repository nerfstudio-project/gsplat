Repo-local Python packages used by `gsplat`. Each subdirectory is its own
installable package. See `design.md` for the shared architectural contract
(`functional/` + `kernels/` layering, CUDA layout, testing rules).

## Packages

| Path             | Distribution name | Import name        | Depends on                |
| ---------------- | ----------------- | ------------------ | ------------------------- |
| `libs/geometry`  | `gsplat-geometry` | `gsplat_geometry`  | `torch`                   |
| `libs/scene`     | `gsplat-scene`    | `gsplat_scene`     | `torch`                   |
| `libs/stage`     | `gsplat-stage`    | `gsplat_stage`     | `torch`, `gsplat-scene`   |

## Install

Use `install.sh` from this directory:

```bash
cd libs
./install.sh                # lists supported packages
./install.sh geometry
./install.sh scene
./install.sh stage          # install AFTER scene
```

Each call runs `pip install -e <package>` against the active Python env. 

To verify what's installed and where it points:

```bash
pip show gsplat-geometry gsplat-scene gsplat-stage | grep -E 'Name|Editable'
```

The `Editable project location:` line should point back into this repo.

## Developing

Installs are editable (`pip install -e`), so edits to any `.py` file under
`libs/<pkg>/...` are picked up by the next `python` / `pytest` invocation with
no reinstall. Re-run `./install.sh <pkg>` only when you change `pyproject.toml`,
add a new top-level subpackage, switch Python envs, or modify native
extensions.

## Testing

Tests are colocated with the layer they exercise (typically alongside
`functional/` or `components/`). Run them through the active env's `pytest`:

```bash
pytest libs/geometry/functional
pytest libs/scene/components
pytest libs/stage/components
```

See `design.md` for the full testing contract.

## Sanity check

After installing, this should print three workspace paths:

```bash
python - <<'PY'
import gsplat_geometry, gsplat_scene, gsplat_stage
for m in (gsplat_geometry, gsplat_scene, gsplat_stage):
    print(m.__name__, "->", m.__file__)
PY
```

If any path points into `site-packages/...` instead of this repo, the package
was installed non-editable; reinstall with `./install.sh <pkg>`.
