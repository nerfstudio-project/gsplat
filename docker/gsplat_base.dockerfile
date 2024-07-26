ARG BASE_IMAGE ubuntu:20.04
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y apt-utils wget software-properties-common build-essential curl
RUN add-apt-repository ppa:deadsnakes/ppa && apt update && \
    apt install -y python3.10 libpython3.10-dev python3.10-venv python-is-python3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 100
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && rm cuda-keyring_1.0-1_all.deb
ARG CUDA_VERSION=11-7
RUN apt update && apt install -y cuda-toolkit-${CUDA_VERSION}
ENV PATH ${PATH}:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
ENV CUDA_HOME /usr/local/cuda
RUN pip3 install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1