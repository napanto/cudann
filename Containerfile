# cudann container: the library installed into the fnn-bench CUDA toolchain image.
#
#   podman build --memory=20g -t cudann .                                    # sm_61 + sm_80 fat binary
#   podman run --rm -it --device nvidia.com/gpu=all --security-opt=label=disable cudann pytest --device gpu
ARG BASE=ghcr.io/napanto/fnn-cuda:latest
FROM ${BASE}
ARG FNN_IMAGE=cudann:latest
ARG FNN_IMAGE_BUILT=unknown

ARG CUDANN_CUDA_ARCHS="61;80"
ENV CUDANN_CUDA_ARCHS=${CUDANN_CUDA_ARCHS} CMAKE_BUILD_PARALLEL_LEVEL=8 CUDAHOSTCXX=g++-13

COPY . /opt/src/cudann
RUN pip install --no-cache-dir -v /opt/src/cudann \
    && pip install --no-cache-dir "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit" \
    && python -c "import cudann; print(cudann.__version__, cudann.build_info())"

# identity of this image (the base keeps its own stamp in the layer history)
ENV FNN_IMAGE=${FNN_IMAGE} FNN_IMAGE_BUILT=${FNN_IMAGE_BUILT}
LABEL org.opencontainers.image.source=https://github.com/napanto/cudann fnn.image="${FNN_IMAGE}" fnn.image.built="${FNN_IMAGE_BUILT}"
WORKDIR /opt/src/cudann
CMD ["python"]
