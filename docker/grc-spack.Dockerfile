# Install ubuntu latest
FROM ubuntu:24.04
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
    bash jq gdbserver gdb gh nano vim
COPY module_load.sh /module_load.sh

#------------------------------------------------------------
# Basic Spack Configuration
#------------------------------------------------------------

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

# Download IOWARP
RUN git clone https://github.com/iowarp/iowarp-install.git && \
    . "${SPACK_DIR}/share/spack/setup-env.sh" && \
    spack repo add iowarp-install/iowarp-spack

# Update bashrc
RUN echo "source ${SPACK_DIR}/share/spack/setup-env.sh" >> ${HOME}/.bashrc && \
    echo "source /module_load.sh" >> ${HOME}/.bashrc

#------------------------------------------------------------
# SSH Configuration
#------------------------------------------------------------

# Copy the host's SSH keys
# Docker requires COPY be relative to the current working
# directory. We cannot pass ~/.ssh/id_ed25519 unfortunately...
RUN mkdir -p ~/.ssh
RUN ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Authorize host SSH keys
RUN touch ~/.ssh/authorized_keys
RUN cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys

# Set SSH permissions
RUN chmod 700 ~/.ssh
RUN chmod 644 ~/.ssh/id_ed25519.pub
RUN chmod 600 ~/.ssh/id_ed25519
RUN chmod 600 ~/.ssh/authorized_keys

# Disable host key checking
RUN echo "Host *" >> ~/.ssh/config
RUN echo "    StrictHostKeyChecking no" >> ~/.ssh/config
RUN chmod 600 ~/.ssh/config

# Enable passwordless SSH
# Replaces #PermitEmptyPasswords no with PermitEmptyPasswords yes
RUN sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config

# Create this directory, because sshd doesn't automatically
RUN mkdir /run/sshd

# Start SSHD
ENTRYPOINT service ssh restart && bash


