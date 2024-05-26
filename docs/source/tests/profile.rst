Profile
===================================

Render RGB Images
-------------------------------------

Batch size 1.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 1 --scene_grid 1 --channels 3

===========================  ========  =============  ==========  ==========  ==========
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]    FPS[bwd]
===========================  ========  =============  ==========  ==========  ==========
gsplat v1.0.0                True      True             **0.17**  243.6       126.4    
gsplat v1.0.0                True      False            **0.17**  280.9       156.7    
gsplat v1.0.0                False     False            **0.17**  **348.3**   **171.9**    
gsplat v0.1.11               n/a       n/a                  0.2   239.1       150.2    
diff-gaussian-rasterization  n/a       n/a                  0.34  282.8       55.6    
===========================  ========  =============  ==========  ==========  ==========

Batch size 8.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 8 --scene_grid 1 --channels 3

===========================  ========  =============  ==========  ============  ============
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]      FPS[bwd]
===========================  ========  =============  ==========  ============  ============
gsplat v1.0.0                True      True             **1.35**  39.8 x 8      **20.9 x 8**
gsplat v1.0.0                True      False            **1.35**  40.4 x 8      20.8 x 8
gsplat v1.0.0                False     False                1.36  **44.0 x 8**  **20.9 x 8**
gsplat v0.1.11               n/a       n/a                  1.19  30.7 x 8      18.4 x 8
diff-gaussian-rasterization  n/a       n/a                  2.67  36.3 x 8      6.8 x 8
===========================  ========  =============  ==========  ============  ============

Batch size 32.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 32 --scene_grid 1 --channels 3

===========================  ========  =============  ==========  =============  ============
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]       FPS[bwd]
===========================  ========  =============  ==========  =============  ============
gsplat v1.0.0                True      True             **5.39**  10.1 x 32      **5.0 x 32**
gsplat v1.0.0                True      False            **5.39**  10.1 x 32      **5.0 x 32**
gsplat v1.0.0                False     False                5.44  **10.6 x 32**  **5.0 x 32**
gsplat v0.1.11               n/a       n/a                  4.74  7.6 x 32       4.6 x 32
diff-gaussian-rasterization  n/a       n/a                 10.67  9.0 x 32       1.7 x 32
===========================  ========  =============  ==========  =============  ============


Render Feature Maps: 32 Channel
------------------------------------------

Batch size 1.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 1 --scene_grid 1 --channels 32

===========================  ========  =============  ==========  ==========  ==========
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]    FPS[bwd]
===========================  ========  =============  ==========  ==========  ==========
gsplat v1.0.0                True      True             **0.61**  124.5       43.6
gsplat v1.0.0                True      False            **0.61**  157.0       **44.3**
gsplat v1.0.0                False     False            **0.61**  **168.4**   44.2
gsplat v0.1.11               n/a       n/a                  0.83  18.3        6.9
diff-gaussian-rasterization  n/a       n/a                  3.66  28.9        5.0
===========================  ========  =============  ==========  ==========  ==========

Batch size 4.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 4 --scene_grid 1 --channels 32

===========================  ========  =============  ==========  ============  ============
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]      FPS[bwd]
===========================  ========  =============  ==========  ============  ============
gsplat v1.0.0                True      True             **2.45**  36.8 x 4      **10.9 x 4**
gsplat v1.0.0                True      False            **2.45**  40.4 x 4      **10.9 x 4**
gsplat v1.0.0                False     False                2.48  **42.1 x 4**  **10.9 x 4**
gsplat v0.1.11               n/a       n/a                  3.28  4.5 x 4       1.7 x 4
diff-gaussian-rasterization  n/a       n/a                 14.52  7.1 x 4       1.2 x 4
===========================  ========  =============  ==========  ============  ============

Render a Large Scene
------------------------------------------

49M Gaussians.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 1 --scene_grid 21 --channels 3

===========================  ========  =============  ==========  ==========  ==========
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]    FPS[bwd]
===========================  ========  =============  ==========  ==========  ==========
gsplat v1.0.0                True      True             **1.44**  53.7        **39.2**
gsplat v1.0.0                True      False                3.08  **62.1**    34.6
gsplat v1.0.0                False     False                5.67  59.2        37.5
gsplat v0.1.11               n/a       n/a                  9.86  23.8        21.1
diff-gaussian-rasterization  n/a       n/a                 10.84  38.3        18.8
===========================  ========  =============  ==========  ==========  ==========

107M Gaussians.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 1 --scene_grid 31 --channels 3

===========================  ========  =============  ==========  ==========  ==========
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]    FPS[bwd]
===========================  ========  =============  ==========  ==========  ==========
gsplat v1.0.0                True      True             **2.31**  45.1        **38.4**
gsplat v1.0.0                True      False                6.11  **47.3**    28.9
gsplat v1.0.0                False     False               12.17  39.3        25.8
gsplat v0.1.11               n/a       n/a                  OOM   OOM         OOM
diff-gaussian-rasterization  n/a       n/a                  OOM   OOM         OOM
===========================  ========  =============  ==========  ==========  ==========

Note: Evaluations are conducted on a NVIDIA TITAN RTX GPU. (commit 8ea2ea3). "Mem" indicates
the amount of GPU memory required by the rasterization operation (excluding the input data),
which is calculated by the diff of `torch.cuda.max_memory_allocated()` before and after the
operation.
