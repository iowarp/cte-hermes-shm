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
