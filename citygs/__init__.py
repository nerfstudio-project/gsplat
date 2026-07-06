"""CityGS: large-scale scene reconstruction on top of gsplat.

A block-based pipeline for city-scale 3D Gaussian Splatting:

    ingest -> coarse -> partition -> finetune (per block) -> merge

Stage contracts are carried by a single ``scene.json`` manifest
(:mod:`citygs.scene.manifest`). Each stage reads the manifest, writes its
artifacts to disk, and records them back into the manifest, so every stage
is independently resumable.
"""

__version__ = "0.1.0"
