# cudann installed into a toolchain image of the study (fnn-bench/containers). Variants by build argument:
#   default   BASE=ghcr.io/napanto/fnn-cuda:latest   nvcc, sm_61 + sm_80 fatbin
#   hip       BASE=ghcr.io/napanto/fnn-rocm:latest   BUILD_CXX=hipcc HIP=ON HIP_ARCHS=gfx1100 (the HIPify build)
ARG BASE=ghcr.io/napanto/fnn-cuda:latest
FROM ${BASE}
ARG IMAGE_NAME=cudann:latest
ARG IMAGE_BUILT=unknown
ARG BUILD_CXX=""
ARG CUDA_ARCHS="61;80"
ARG HIP=OFF
ARG HIP_ARCHS=gfx1100
ENV CUDANN_CUDA_ARCHS=${CUDA_ARCHS} CUDANN_HIP=${HIP} CUDANN_HIP_ARCHS=${HIP_ARCHS} CMAKE_BUILD_PARALLEL_LEVEL=8 CUDAHOSTCXX=g++-13
COPY . /opt/src/cudann
RUN if [ -n "${BUILD_CXX}" ]; then export CXX="${BUILD_CXX}"; fi \
    && pip install --no-cache-dir -v /opt/src/cudann \
    && pip install --no-cache-dir "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit" \
    && python -c "import cudann; print(cudann.__version__, cudann.build_info())"
ENV FNN_IMAGE=${IMAGE_NAME} FNN_IMAGE_BUILT=${IMAGE_BUILT}
LABEL org.opencontainers.image.source=https://github.com/napanto/cudann fnn.image="${IMAGE_NAME}" fnn.image.built="${IMAGE_BUILT}"
WORKDIR /opt/src/cudann
CMD ["python"]
