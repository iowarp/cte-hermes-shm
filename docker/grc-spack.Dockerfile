# Install ubuntu latest
FROM ubuntu:latest
LABEL maintainer="llogan@hawk.iit.edu"
LABEL version="0.0"
LABEL description="GRC spack docker image"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update ubuntu
SHELL ["/bin/bash", "-c"]
RUN apt update && apt install

# Install some basic packages
RUN apt install -y \
    openssl libssl-dev openssh-server \
    sudo git \
    gcc g++ gfortran make binutils gpg \
    tar zip xz-utils bzip2 \
    perl m4 libncurses5-dev libxml2-dev diffutils \
    pkg-config cmake pkg-config \
    python3 python3-pip doxygen \
    lcov zlib1g-dev hdf5-tools \
    build-essential ca-certificates \
    coreutils curl environment-modules \
    gfortran git gpg lsb-release python3 \
    python3-venv unzip zip \
    bash jq gdbserver gdb gh
COPY module_load.sh ./module_load.sh

# Setup basic environment
ENV USER="root"
ENV HOME="/root"
ENV SPACK_DIR="${HOME}/spack"
ENV SPACK_VERSION="v0.22.2"

# Install Spack
RUN . /module_load.sh && \
    git clone -b ${SPACK_VERSION} https://github.com/spack/spack ${SPACK_DIR} && \
    . "${SPACK_DIR}/share/spack/setup-env.sh" && \
    spack external find

# Download GRC
RUN git clone https://github.com/grc-iit/grc-repo.git && \
    . "${SPACK_DIR}/share/spack/setup-env.sh" && \
    spack repo add grc-repo

# Update bashrc
RUN echo "source ${SPACK_DIR}/share/spack/setup-env.sh" >> ${HOME}/.bashrc && \
    echo "source module_load.sh" >> ${HOME}/.bashrc
