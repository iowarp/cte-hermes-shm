FROM iowarp/ppi-jarvis-cd:latest

COPY . /workspace

WORKDIR /workspace

RUN cmake --preset=release && \
    cd build && \
    sudo make -j$(nproc) install && \
    sudo rm -rf /workspace
