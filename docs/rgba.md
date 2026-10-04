# Transparent COLMAP inputs

`examples/simple_trainer.py` supports 8-bit straight-alpha RGBA images with `--use-alpha`.
Keep transparent PNGs in the COLMAP image directory. Alpha follows the same
undistortion, crop and optional image cache as RGB. RGB images are treated as
opaque when this option is enabled. Without the option, input alpha is ignored
and existing RGB training behavior is unchanged.

```bash
python examples/simple_trainer.py default --data-dir /path/to/scene \
    --data-factor 1 --use-alpha --random-bkgd --disable-viewer
```

Training composites the target and premultiplied render over the same random
background when `--random-bkgd` is enabled, or over black otherwise. Evaluation
always uses black so metrics do not depend on a sampled background. Alpha is not
the COLMAP distortion/occlusion mask; those masks remain separate.

Downsampling preserves RGBA in newly generated PNGs. Remove or regenerate any
existing downsampled RGB cache created by an older version before using alpha.
This option currently supports the COLMAP backend, not NCore.
