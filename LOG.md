



#### `prune_scale3d` prunes too many GS than it should be?

```
CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default --data_factor 8 --port 8082 --opt_vert \
    --result_dir results/tet_py_init3_lr1e-3_exp \
    --strategy.prune_scale3d 1.0
```

Does not make any difference so it should not be the thing that matters.

#### `prune_opa`? (redo this)

```
CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default --data_factor 8 --port 8082 --opt_vert \
    --result_dir results/tet_py_init3_lr1e-3_exp \
    --strategy.prune_opa 0.008
```

#### Intialization should have 2x large GS because we are using init3?

hack th code:
> dist_avg = torch.sqrt(dist2_avg) * 2.0

#### t_init_s=6.0

```
CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default --data_factor 8 --port 8082 --opt_vert \
    --result_dir results/tet_cu_init6_lr1e-4 --t_init_s 6.0
```


#### centering tet

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --result_dir results/tet_center_init3_lr1e-3 --data_factor 8 --port 8082 --opt_vert --t_lr_v 1e-3 --t_init_s 3
CUDA_VISIBLE_DEVICES=1 python simple_trainer.py default --result_dir results/tet_center_init3_lr1e-2 --data_factor 8 --port 8083 --opt_vert --t_lr_v 1e-2 --t_init_s 3
CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default --result_dir results/tet_center_init3_lr1e-1 --data_factor 8 --port 8084 --opt_vert --t_lr_v 1e-1 --t_init_s 3

CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default --result_dir results/tet_norm_init3_lr1e-1 --data_factor 8 --port 8084 --opt_vert --t_lr_v 1e-1 --t_init_s 3




CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default --result_dir results/tet_norm_init6_lr1e-3 --data_factor 8 --port 8084 --opt_vert --t_lr_v 1e-3 --t_init_s 6.0
CUDA_VISIBLE_DEVICES=3 python simple_trainer.py default --result_dir results/tet_norm_init6_lr1e-2 --data_factor 8 --port 8084 --opt_vert --t_lr_v 1e-2 --t_init_s 6.0
CUDA_VISIBLE_DEVICES=4 python simple_trainer.py default --result_dir results/tet_norm_init6_lr1e-1 --data_factor 8 --port 8084 --opt_vert --t_lr_v 1e-1 --t_init_s 6.0
CUDA_VISIBLE_DEVICES=5 python simple_trainer.py default --result_dir results/tet_norm_init6_lr1e-4 --data_factor 8 --port 8084 --opt_vert --t_lr_v 1e-4 --t_init_s 6.0