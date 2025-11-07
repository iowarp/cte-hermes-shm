FROM iowarp/ppi-jarvis-cd-build:latest

COPY . /workspace

WORKDIR /workspace

# Configure with release preset and build
# Install to both /usr/local and /cte-hermes-shm for flexibility
RUN sudo chown -R $(whoami):$(whoami) /workspace && \
    mkdir -p build && \
    cmake --preset release && \
    cmake --build build -j$(nproc) && \
    sudo cmake --install build --prefix /usr/local && \
    sudo cmake --install build --prefix /cte-hermes-shm && \
    rm -rf /workspace

# Add cte-hermes-shm to Spack configuration
RUN echo "  cte-hermes-shm:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: cte-hermes-shm@main" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr/local" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml