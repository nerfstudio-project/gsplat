ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN mkdir -p /workspace
ARG GSPLAT_VERSION 1.1.1
COPY dist/gsplat-${GSPLAT_VERSION}-cp310-cp310-linux_x86_64.whl /workspace
RUN pip3 install /workspace/gsplat-${GSPLAT_VERSION}-cp310-cp310-linux_x86_64.whl && \
    rm /workspace/gsplat-${GSPLAT_VERSION}-cp310-cp310-linux_x86_64.whl
WORKDIR /workspace