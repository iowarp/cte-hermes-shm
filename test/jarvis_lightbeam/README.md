# LightBeam Jarvis Packages

Simple Jarvis packages for deploying LightBeam server and client with round-trip timing.

## Quick Start

### 1. Setup Environment
```bash
cd /home/rpawar4/cte-hermes-shm/test/jarvis_lightbeam
export PYTHONPATH=$PYTHONPATH:$(pwd)/jarvis_lightbeam
```

### 2. Run Server
```bash
jarvis pkg configure lightbeam_server
jarvis pkg start lightbeam_server
```

### 3. Run Client (in another terminal)
```bash
cd /home/rpawar4/cte-hermes-shm/test/jarvis_lightbeam
export PYTHONPATH=$PYTHONPATH:$(pwd)/jarvis_lightbeam
jarvis pkg configure lightbeam_client
jarvis pkg start lightbeam_client
```

### 4. Stop Server
```bash
jarvis pkg stop lightbeam_server
```

## Configuration Options

**Server:**
- SERVER_IP: IP to bind to (default: 0.0.0.0)
- SERVER_PORT: Port to bind to (default: 5555)
- TRANSPORT_TYPE: tcp or libfabric (default: tcp)
- VERBOSE: Enable verbose output (default: true)

**Client:**
- SERVER_IP: Server IP (default: 127.0.0.1)
- SERVER_PORT: Server port (default: 5555)
- TRANSPORT_TYPE: tcp or libfabric (default: tcp)
- ITERATIONS: Number of benchmark iterations (default: 1000)
- MESSAGE_SIZE: Message size in bytes (default: 64)
- VERBOSE: Enable verbose output (default: true)

## Remote Deployment with Pssh

```bash
# Server on remote node
pssh -h server_hosts.txt -i "jarvis pkg start lightbeam_server"

# Client on remote node
pssh -h client_hosts.txt -i "jarvis pkg start lightbeam_client"
``` 