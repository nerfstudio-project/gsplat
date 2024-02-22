Evaluation
===================================

We evaluate our implementation of Gaussian Splatting (Splatfacto) on the Mip-NeRF 360 dataset, benchmarking it against the original Inria method (commit 2eee0e26d2d5fd00ec462df47752223952f6bf4e). We report results at 7,000 and 30,000 steps. All evaluations were executed on an NVIDIA RTX 4090 GPU.

.. list-table:: Time
   :widths: 10 10 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - 
     - Bicycle
     - Bonsai
     - Counter
     - Flowers
     - Garden
     - Kitchen
     - Stump
     - Treehill
     - Avg
   * - inria-7k
     - 3:34
     - 3:27
     - 3:05
     - 3:02
     - 3:51
     - 4:00
     - 3:25
     - 2:53
     - 3:24
   * - splfacto-7k
     - 2:36
     - 2:18
     - 2:06
     - 2:14
     - 2:49
     - 2:17
     - 2:24
     - 2:23
     - 2:23
   * - splfacto-big-7k
     - 3:41
     - 3:11
     - 2:59
     - 2:51
     - 3:51
     - 3:15
     - 2:49
     - 2:47
     - 3:10
   * - inria-30k
     - 25:07
     - 14:37
     - 15:24
     - 17:32
     - 24:03
     - 19:02
     - 20:25
     - 18:02
     - 19:17
   * - splfacto-30k
     - 18:03
     - 10:13
     - 9:07
     - 13:25
     - 14:58
     - 10:02
     - 15:15
     - 16:56
     - 13:30
   * - splfacto-big-30k
     - 31:05
     - 13:32
     - 13:27
     - 22:44
     - 26:03
     - 16:16
     - 21:58
     - 25:26
     - 21:19

.. list-table:: PSNR
   :widths: 10 10 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - 
     - Bicycle
     - Bonsai
     - Counter
     - Flowers
     - Garden
     - Kitchen
     - Stump
     - Treehill
     - Avg
   * - inria-7k
     - 24.11
     - 29.49
     - 27.16
     - 20.54
     - 26.53
     - 29.02
     - 26.74
     - 22.50
     - 25.76
   * - splatfacto-7k
     - 22.99
     - 29.45
     - 26.92
     - 20.33
     - 25.76
     - 28.48
     - 24.59
     - 21.91
     - 25.05
   * - splatfacto-big-7k
     - 23.66
     - 29.69
     - 27.01
     - 20.73
     - 26.58
     - 28.82
     - 25.69
     - 22.11
     - 25.54
   * - inria-30k
     - 25.61
     - 31.89
     - 28.96
     - 21.56
     - 27.60
     - 31.30
     - 25.89
     - 22.07
     - 26.86
   * - splatfacto-30k
     - 24.99
     - 32.14
     - 28.72
     - 21.54
     - 27.31
     - 31.18
     - 25.64
     - 22.28
     - 26.73
   * - splatfacto-big-30k
     - 25.7
     - 32.23
     - 28.95
     - 21.96
     - 27.83
     - 31.6
     - 26.7
     - 22.38
     - 27.17


.. list-table:: LPIPS
   :widths: 10 10 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - 
     - Bicycle
     - Bonsai
     - Counter
     - Flowers
     - Garden
     - Kitchen
     - Stump
     - Treehill
     - Avg
   * - inria-7k
     - 0.31
     - 0.24
     - 0.25
     - 0.42
     - 0.16
     - 0.16
     - 0.28
     - 0.42
     - 0.28
   * - splfacto-7k
     - 0.31
     - 0.16
     - 0.21
     - 0.44
     - 0.15
     - 0.14
     - 0.28
     - 0.45
     - 0.27
   * - splfacto-big-7k
     - 0.28
     - 0.16
     - 0.20
     - 0.42
     - 0.12
     - 0.13
     - 0.23
     - 0.43
     - 0.24
   * - inria-30k
     - 0.21
     - 0.21
     - 0.20
     - 0.34
     - 0.11
     - 0.13
     - 0.22
     - 0.32
     - 0.22
   * - splfacto-30k
     - 0.18
     - 0.13
     - 0.17
     - 0.34
     - 0.09
     - 0.10
     - 0.18
     - 0.32
     - 0.19
   * - splfacto-big-30k
     - 0.15
     - 0.13
     - 0.15
     - 0.31
     - 0.07
     - 0.09
     - 0.15
     - 0.28
     - 0.17

.. list-table:: SSIM
   :widths: 10 10 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - 
     - Bicycle
     - Bonsai
     - Counter
     - Flowers
     - Garden
     - Kitchen
     - Stump
     - Treehill
     - Avg
   * - inria-7k
     - 0.69
     - 0.92
     - 0.88
     - 0.53
     - 0.83
     - 0.90
     - 0.73
     - 0.59
     - 0.76
   * - splfacto-7k
     - 0.65
     - 0.92
     - 0.88
     - 0.53
     - 0.85
     - 0.90
     - 0.68
     - 0.58
     - 0.74
   * - splfacto-big-7k
     - 0.69
     - 0.92
     - 0.88
     - 0.55
     - 0.84
     - 0.90
     - 0.74
     - 0.61
     - 0.77
   * - inria-30k
     - 0.78
     - 0.94
     - 0.91
     - 0.61
     - 0.87
     - 0.92
     - 0.77
     - 0.63
     - 0.80
   * - splfacto-30k
     - 0.75
     - 0.94
     - 0.90
     - 0.60
     - 0.85
     - 0.92
     - 0.73
     - 0.63
     - 0.79
   * - splfacto-big-30k
     - 0.78
     - 0.94
     - 0.91
     - 0.63
     - 0.88
     - 0.93
     - 0.77
     - 0.64
     - 0.81
