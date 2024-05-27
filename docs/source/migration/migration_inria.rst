Migratite from `diff-gaussian-rasterization <https://github.com/graphdeco-inria/diff-gaussian-rasterization>`_
==================================================================================================================================================================

.. currentmodule:: gsplat

`gsplat` is fully compatible with the official Gaussian Splatting CUDA backend `diff-gaussian-rasterization <https://github.com/graphdeco-inria/diff-gaussian-rasterization>`_.

In `this fork <https://github.com/liruilong940607/gaussian-splatting>`_ of the official code base, we replace the rasterization backend from `diff-gaussian-rasterization` to `gsplat` with 
minimal changes (less than 100 lines), and get some improvements for free:

For example we get a 20% training speedup and a noticeable memory reduction, with slightly better performance:

+-------------------------------+---------------+--------+--------+-------+-------+
| Backend                       | Training Time | Memory | SSIM   | PSNR  | LPIPS |
+===============================+===============+========+========+=======+=======+
| `diff-gaussian-rasterization` | 482s          | 9.11GB | 0.8237 | 26.11 | 0.166 |
+-------------------------------+---------------+--------+--------+-------+-------+
| `gsplat`                      | 402s          | 8.78GB | 0.8366 | 26.18 | 0.163 |
+-------------------------------+---------------+--------+--------+-------+-------+

Note the improvements will be much more significant on larger scenes. 
On top of that, there are more functionalities supported in `gsplat`, including
**batched rasterization**, **trade-off between memory and speed**, **sparse gradient** etc.

Additionally, we also provide a wrapper function 
:func:`gsplat.rasterization_inria_wrapper` on top of the `diff-gaussian-rasterization`, that 
aligns with our API :func:`gsplat.rasterization` in `v1.0.0`.
