#!/usr/bin/env bash
export BASE_IMAGE=ubuntu:20.04
export GSPLAT_VERSION=`cat gsplat/version.py | cut -d '"' -f 2`
export CUDA_VERSION=11-7

echo "build gsplat base image"
docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    -t nerfstudio-gsplat-base:"${GSPLAT_VERSION}" \
    -f ./docker/gsplat_base.dockerfile .

echo "build gsplat wheel"
# create a container to build wheel for only container can use CUDA
docker run -t --rm --name nerfstudio-gsplat-wheel \
    --gpus 1 -v $PWD:/gsplat \
    nerfstudio-gsplat-base:"${GSPLAT_VERSION}" \
    bash -c "cd /gsplat && python3 setup.py sdist bdist_wheel"

echo "build gsplat worker image"
docker build --build-arg BASE_IMAGE=nerfstudio-gsplat-base:"${GSPLAT_VERSION}" \
    --build-arg GSPLAT_VERSION="${GSPLAT_VERSION}" \
    -t nerfstudio-gsplat-worker:"${GSPLAT_VERSION}" \
    -f ./docker/gsplat_worker.dockerfile .