# Build for NRE

```
# assume the system has CUDA 12.4 already
conda create -n gsplat-build python=3.11
conda activate gsplat-build

# install torch 2.5.0
pip install torch==2.5.0 --extra-index-url https://download.pytorch.org/whl2.5.0
# check torch version and cuda version are as expected
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.version.cuda)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# set gsplat version
VERSION=`sed -n 's/^__version__ = "\(.*\)"/\1/p' gsplat/version.py`
sed -i "s/$VERSION/$VERSION+pt2.5.0cu124/" gsplat/version.py
# check version is as expected
cat gsplat/version.py

# update pip
pip install --upgrade setuptools ninja wheel twine

# build wheel
MAX_JOBS=10 python setup.py bdist_wheel --dist-dir=dist
# test wheel
cd dist && pip install *.whl && python -c "import gsplat; print('gsplat:', gsplat.__version__)" && cd ..

twine upload --repository-url https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/pypi dist/* --verbose --skip-existing --username __token__ --password $TOKEN_PASSWORD
```