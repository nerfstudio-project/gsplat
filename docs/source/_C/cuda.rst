Cuda Lib
===================================

.. currentmodule:: diff_rast


Some of the important CUDA backend functions are exposed to python with the `cuda` submodule. You can import the CUDA bindings with:

.. code-block:: python

    import diff_rast.cuda as _C
    help(_C)

The following functions are currently supported:

* _C.rasterize_forward 
* _C.rasterize_backward
* _C.compute_cov2d_bounds_forward 
* _C.project_gaussians_forward
* _C.project_gaussians_backward
* _C.compute_sh_forward
* _C.compute_sh_backward
* _C.compute_cumulative_intersects 
* _C.map_gaussian_to_intersects


rasterize_forward
-----------------

.. code-block:: python

    _C.rasterize_forward(*args, **kwargs)

        PARAMETERS:
            xys: Float[Tensor, "*batch 2"],
            depths: Float[Tensor, "*batch 1"],
            radii: Float[Tensor, "*batch 1"],
            conics: Float[Tensor, "*batch 3"],
            num_tiles_hit: Int[Tensor, "*batch 1"],
            colors: Float[Tensor, "*batch channels"],
            opacity: Float[Tensor, "*batch 1"],
            img_height: int,
            img_widt: int,
            background: Float[Tensor, "channels"]

rasterize_backward
------------------

.. code-block:: python

    _C.rasterize_backward(*args, **kwargs)

        PARAMETERS:
            img_height: int,
            img_width: int,
            gaussian_ids_sorted: ,
            tile_bins: Int[Tensor, "x h 1"],
            xys: Float[Tensor, "*batch 2"],
            conics: Float[Tensor, "*batch 3"],
            colors: Float[Tensor, "*batch channels"],
            opacity: Float[Tensor, "*batch 1"],
            background: Float[Tensor, "channels"],
            final_Ts: Float[Tensor, "*batch 1"],
            final_idx: Float[Tensor, "*batch 1"],
            v_out_img: Float[Tensor, "*batch channels"]


compute_cov2d_bounds_forward
----------------------------

.. code-block:: python

    _C.compute_cov2d_bounds_forward(*args, **kwargs)

        PARAMETERS:
            cov2d: Float[Tensor, "*batch 3"]


project_gaussians_forward
-------------------------

.. code-block:: python

    _C.project_gaussians_forward(*args, **kwargs)

        PARAMETERS:
            num_points: int,
            means3d: Float[Tensor, "*batch 3"],
            scales: Float[Tensor, "*batch 3"],
            glob_scale: int,
            quats: Float[Tensor, "*batch 4"],
            viewmat: Float[Tensor, "*batch 4 4"],
            projmat: Float[Tensor, "*batch 4 4"],
            fx: float,
            fy: float,
            img_height: int,
            img_width: int,
            tile_bounds: Int[Tensor, "tiles.x tiles.y 1"],
            clip_thresh: float,


project_gaussians_backward
--------------------------

.. code-block:: python

    _C.project_gaussians_backward(*args, **kwargs)

        PARAMETERS:
            num_points int,
            means3d: Float[Tensor, "*batch 3"],
            scales: Float[Tensor, "*batch 3"],
            glob_scale: int,
            quats: Float[Tensor, "*batch 4"],
            viewmat:Float[Tensor, "*batch 4 4"],
            projmat: Float[Tensor, "*batch 4 4"],
            fx: float,
            fy: float,
            img_height: int,
            img_width: int,
            cov3d: Int[Tensor, "*batch 5"],
            radii: Int[Tensor, "*batch 1"],
            conics: Float[Tensor, "*batch 3"],
            v_xys: Float[Tensor, "*batch 2"],
            v_conics: Float[Tensor, "*batch 3"]


compute_sh_forward
------------------

.. code-block:: python

    _C.compute_sh_forward(*args, **kwargs)

        PARAMETERS:
            num_points: int, 
            degree: int, 
            viewdirs: Float[Tensor, "*batch 3"], 
            coeffs: Float[Tensor, "*batch degree channels"]


compute_sh_backward
-------------------

.. code-block:: python

    _C.compute_sh_backward(*args, **kwargs)

        PARAMETERS:
            num_points: int, 
            degree: int, 
            viewdirs: Float[Tensor, "*batch 3"], 
            v_colors: Float[Tensor, "*batch channels"]


compute_cumulative_intersects
-----------------------------

.. code-block:: python

    _C.compute_cumulative_intersects(*args, **kwargs)

        PARAMETERS:
            num_points: int, 
            num_tiles_hit: Int[Tensor, "*batch 1"]


map_gaussian_to_intersects
--------------------------

.. code-block:: python

    _C.map_gaussian_to_intersects(*args, **kwargs)

        PARAMETERS:
            num_points: int, 
            xys: Int[Tensor, "*batch 2"], 
            depths: Int[Tensor, "*batch 1"], 
            radii: Int[Tensor, "*batch 1"], 
            cum_tiles_hit: Int[Tensor, "*batch 1"], 
            tile_bounds: Int[Tensor, "tiles.x tiles.y 1"]