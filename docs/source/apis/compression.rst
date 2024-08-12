Storage Compression
===================================

.. currentmodule:: gsplat

`gsplat` provides handy APIs for compressing and decompressing the Gaussian parameters,
which can significantly reduce the storage / streaming cost. For example, using :class:`PngCompression`,
1 million Gaussians that are stored in 236 MB can be compressed to 16.5 MB, which only 0.5dB
PSNR loss (29.18dB to 28.65dB).

The following code snippet is an example of how to use the compression approach in `gsplat`:

.. code-block:: python

    from gsplat import PngCompression

    splats: Dict[str, Tensor] = {
        "means": Tensor(N, 3), "scales": Tensor(N), "quats": Tensor(N, 4), "opacities": Tensor(N),
        "colors": Tensor(N, 25, 3), "features1": Tensor(N, 128), "features2": Tensor(N, 64),
    }

    compression_method = PngCompression()
    # run compression and save the compressed files to compress_dir
    compression_method.compress(compress_dir, params)
    # decompress the compressed files
    splats_c = compression_method.decompress(compress_dir)

Below is the APIs for the compression approaches we supported in `gsplat`:

.. autoclass:: PngCompression
    :members:
