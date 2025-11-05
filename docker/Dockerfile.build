FROM iowarp/ppi-jarvis-cd:latest

COPY . /workspace

WORKDIR /workspace

RUN cd build && sudo make -j$(nproc) install
RUN sudo rm -rf /workspace

