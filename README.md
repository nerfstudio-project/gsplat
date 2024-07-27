# SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting
### [Project Page](https://spotlesssplats.github.io/) | [Paper](https://arxiv.org/abs/2406.20055)

## Reproduced Results
The results in the SpotLessSplats paper are reproduced on `gsplat` codebase. The results on `gsplat` are reported without enabling the appearance modeling method of `gsplat`.

SpotLessSplats on  Crab (2) dataset:
<img src="https://github.com/lilygoli/SpotLessSplats/raw/main/assets/sls_crab2.gif" height="400px" alt="Crab Ours">

Vanilla 3DGS on Crab (2) dataset:
<img src="https://github.com/lilygoli/SpotLessSplats/raw/main/assets/base_crab2.gif" height="400px" alt="Crab Base">

The reproduced results of Fig. 8 in the paper on `gsplat`:

|              |Android |Statue |Crab (2)| Yoda |Mountain|Fountain|Corner|Patio | Spot   | Patio High | Average
|--------------|--------|-------|--------|------|--------|--------|------|------|--------|------------|--------
|3DGS          |  23.23 | 21.45 | 30.03 | 29.7 | 20.02 | 21.49 | 22.34 |  16.77 | 18.93 | 17.09 | 22.105 |
|RobustFilter  |  24.34 |22.46 | 34.15 | 34.91 | 21.2 | 21.91| 25.66|17.9|23.21|20.22|24.596|
|SLS-agg       |   25.01|22.76|34.96|35.06|21.66|22.66|25.77|22.58|24.37|22.72|25.755 |
|SLS-mlp       |   25.05 |	22.85 |	35.33 |	35.4 |	21.67 |	22.51 |	25.84 |	22.68 |	25.06 |	23.12 |	25.95 |
|SLS-mlp + UBP |  24.96 | 22.65 | 35.3| 35.26| 21.31| 22.3| 26.36| 22.2|25.12|23.00|25.846 |

The original results in the paper:

![Fig 8](https://github.com/lilygoli/SpotLessSplats/raw/main/assets/sls-benchmark-paper.png)

The effect of UBP with $\kappa=10^{-14}$ on `gsplat`:
 
|              |Compression Factor |PSNR drop
|--------------|--------|-------|
|bicycle          |  2.21x | -0.32|
|garden  |  3.35x |-0.41 | 
|stump       |   2.53x|-0.23|

## Installation
Installation process is similar to the main `gsplat` branch. Make sure to `pip install -r requirements.txt` under the examples directory.

## Run Experiments
You can run experiments with the robust masking as below:

To run the SLS-mlp version run:
``` 
python spotless_trainer.py  --data_dir [data directory] --data_factor 8   --result_dir [result directory] --loss_type robust --semantics --no-cluster --train_keyword "clutter" --test_keyword "extra" 
```
To run the SLS-agg version run:
``` 
python spotless_trainer.py  --data_dir [data directory] --data_factor 8   --result_dir [result directory] --loss_type robust --semantics --cluster --train_keyword "clutter" --test_keyword "extra" 
```
To run the RobustFilter version run:
``` 
python spotless_trainer.py  --data_dir [data directory] --data_factor 8   --result_dir [result directory] --loss_type robust --no-semantics --no-cluster --train_keyword "clutter" --test_keyword "extra" 
```
To run baseline 3DGS run:
``` 
python spotless_trainer.py  --data_dir [data directory] --data_factor 8   --result_dir [result directory] --loss_type l1 --no-semantics --no-cluster --train_keyword "clutter" --test_keyword "extra" 
```
For enabling utilization-based pruning (UBP) add `--ubp` to the runs.

## Benchmarking
To run all the experiments together run:
```
./sls_benchmark.sh
```
## Preparing Datasets
The [RobustNeRF  dataset](https://storage.googleapis.com/jax3d-public/projects/robustnerf/robustnerf.tar.gz) and [NeRF On-the-go dataset](https://cvg-data.inf.ethz.ch/on-the-go.zip) are used for experiments. The Stable Diffusion features used for these scenes can be found [here](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/WOFXFT&faces-redirect=true).
 
To extract these features on your own datasets you can run the Jupyter notebook `./examples/datasets/sd_feature_extraction.ipynb`. 

We assume that the image files have prefixes determining clean (`clean`), cluttered train data (`clutter`) and clean test data (`extra`). You can process datasets like NeRF On-the-go datasets that provide a JSON file with these tags with running:
```
python ./examples/datasets/prep_data.py
```
