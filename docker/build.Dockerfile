FROM iowarp/ppi-jarvis-cd-build:latest

COPY . /workspace

WORKDIR /workspace

RUN cmake --preset=release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cd build && \
    sudo make -j$(nproc) install && \
    cmake -DCMAKE_INSTALL_PREFIX=/cte-hermes-shm . && \
    sudo make -j$(nproc) install && \
    cd .. && \
    sudo rm -rf /workspace

# Add cte-hermes-shm to Spack configuration
RUN echo "  cte-hermes-shm:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: cte-hermes-shm@main" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr/local" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml