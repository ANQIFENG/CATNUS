# Define several build arguments
ARG PYTORCH_VERSION=2.2.1
ARG CUDA_VERSION=11.8
ARG CUDNN_VERSION=8
ARG HDBET_COMMIT=ae16068
ARG PYTHON_VERSION=3.11.6
ARG DEBIAN_VERSION=bookworm
ARG ANTS_VERSION=v2.5.0
ARG RADIFOX_VERSION=1.0.5
ARG NIBABEL_VERSION=4.0.1
ARG SCIPY_VERSION=1.8.1
ARG HACA3_COMMIT=8c3f53c

# Set base images: pytorch, ants, python and debian
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime as pytorch
FROM antsx/ants:${ANTS_VERSION} as ants
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION}

# Set the maintainer label
LABEL maintainer=afeng11@jhu.edu

# Re-declare the build arguments
ARG PYTORCH_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG HDBET_COMMIT
ARG PYTHON_VERSION
ARG DEBIAN_VERSION
ARG ANTS_VERSION
ARG RADIFOX_VERSION
ARG NIBABEL_VERSION
ARG SCIPY_VERSION
ARG HACA3_COMMIT

# Set an environment variable for the Python
ENV PYTHONUSERBASE=/opt/python

# Write the build arguments to a JSON file for reference
RUN echo -e "{\n \
    \"PYTORCH_VERSION\": \"${PYTORCH_VERSION}\",\n \
    \"CUDA_VERSION\": \"${CUDA_VERSION}\",\n \
    \"CUDNN_VERSION\": \"${CUDNN_VERSION}\",\n \
    \"HDBET_COMMIT\": \"${HDBET_COMMIT}\",\n \
    \"PYTHON_VERSION\": \"${PYTHON_VERSION}\",\n \
    \"DEBIAN_VERSION\": \"${DEBIAN_VERSION}\",\n \
    \"ANTS_VERSION\": \"${ANTS_VERSION}\",\n \
    \"RADIFOX_VERSION\": \"${RADIFOX_VERSION}\",\n \
    \"NIBABEL_VERSION\": \"${NIBABEL_VERSION}\",\n \
    \"SCIPY_VERSION\": \"${SCIPY_VERSION}\",\n \
    \"HACA3_COMMIT\": \"${HACA3_COMMIT}\"\n \
}" > /opt/manifest.json

# Update the package list and install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y ca-certificates git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Copy PyTorch libraries from pytorch image
COPY --from=pytorch /opt/conda/lib/python3.10/site-packages/torch /opt/python/lib/python3.11/site-packages/torch
COPY --from=pytorch /opt/conda/lib/python3.10/site-packages/torchvision /opt/python/lib/python3.11/site-packages/torchvision

# Install necessary Python libraries
RUN pip install --no-cache-dir \
        radifox==${RADIFOX_VERSION} \
        nibabel==${NIBABEL_VERSION} \
        scipy==${SCIPY_VERSION} \
        SimpleITK \
        numpy \
        scikit-learn \
        scikit-image \
        git+https://github.com/lianruizuo/haca3@${HACA3_COMMIT}

# Set up HD-BET
RUN pip install --no-cache-dir git+https://github.com/MIC-DKFZ/HD-BET.git@${HDBET_COMMIT}
COPY setup_hdbet.py /opt
RUN python /opt/setup_hdbet.py && \
    rm -f /opt/setup_hdbet.py

# Install intensity-normalization with antspy dependency
RUN pip install --no-cache-dir "intensity-normalization[ants]"

# Copy registration atlas to /opt/atlas
COPY atlas /opt/atlas

# Copy ANTs from ants image
COPY --from=ants /opt/ants /opt/ants

# Set environment variables for executable paths and library paths
ENV PATH /opt/run:/opt/ants/bin:${PYTHONUSERBASE}/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/ants/lib:${LD_LIBRARY_PATH}

# Copy HACA3 models and target image
COPY --chmod=0755 haca3_models/*.pt /opt/run/haca3_models/
COPY --chmod=0755 haca3_target/*.nii.gz /opt/run/haca3_target/

# Copy preprocessing scripts, utils, and model weights
COPY --chmod=0755 preprocessing /opt/run/preprocessing/
COPY --chmod=0755 utils /opt/run/
COPY --chmod=0755 model_weights/*.pt /opt/run/
COPY --chmod=0755 run-catnus /opt/run/

# Set the default command to be executed when the container starts
ENTRYPOINT ["run-catnus"]
