# LightBeam Client/Server Benchmark Tests

This directory contains separate client and server test programs for benchmarking the LightBeam transport library with round-trip timing measurements.

## Test Programs

### 1. `lightbeam_server_test` - Echo Server
Runs a standalone echo server that responds to client messages.

**Usage:**
```bash
# TCP server (default)
./bin/lightbeam_server_test

# TCP server with custom URL
./bin/lightbeam_server_test --tcp --url tcp://127.0.0.1:5555

# RDMA server
./bin/lightbeam_server_test --rdma

# RDMA server with custom URL  
./bin/lightbeam_server_test --rdma --url verbs://127.0.0.1:5556

# Help
./bin/lightbeam_server_test --help
```

### 2. `lightbeam_client_test` - Benchmark Client
Connects to a running server and performs round-trip timing measurements.

**Usage:**
```bash
# TCP client (default, 10 iterations)
./bin/lightbeam_client_test

# TCP client with more iterations
./bin/lightbeam_client_test --tcp --iterations 100

# RDMA client
./bin/lightbeam_client_test --rdma

# Custom URL and iterations
./bin/lightbeam_client_test --url tcp://127.0.0.1:5555 --iterations 50

# Help
./bin/lightbeam_client_test --help
```

## Running Benchmarks

### TCP Transport Benchmark
1. **Terminal 1 (Server):**
   ```bash
   ./bin/lightbeam_server_test --tcp
   ```

2. **Terminal 2 (Client):**
   ```bash
   ./bin/lightbeam_client_test --tcp --iterations 100
   ```

### RDMA Transport Benchmark  
1. **Terminal 1 (Server):**
   ```bash
   ./bin/lightbeam_server_test --rdma
   ```

2. **Terminal 2 (Client):**
   ```bash
   ./bin/lightbeam_client_test --rdma --iterations 100
   ```

## Expected Output

### Server Output:
```
LightBeam Echo Server Test
==========================

=== Starting Echo Server: TCP Transport ===
URL: tcp://127.0.0.1:5555
Starting server...
[ZMQ Server] Starting server on tcp://127.0.0.1:5555
[ZMQ Server] Server started successfully
Server started successfully and listening on tcp://127.0.0.1:5555
Waiting for client connections...
Press Ctrl+C to stop the server.
```

### Client Output:
```
LightBeam Client Benchmark Test
===============================

=== Client Benchmark: TCP Transport ===
URL: tcp://127.0.0.1:5555
Iterations: 10
Connecting to server...
[ZMQ Client] Connecting to tcp://127.0.0.1:5555

Starting benchmark...
Iteration | Message | Round-trip Time (μs)
----------|---------|--------------------
        1 | Benchm... |           245.67 ✓
        2 | Benchm... |           189.23 ✓
        3 | Benchm... |           201.45 ✓
        ...

=== Benchmark Results ===
Transport Type: TCP (ZeroMQ)
Total Iterations: 10
Average Round-trip Time: 198.45 μs
Min Round-trip Time: 156.78 μs  
Max Round-trip Time: 245.67 μs
Throughput: 5039.11 requests/second
```

## Key Features

### Timer Implementation
The client uses `hshm::Timer` for precise round-trip measurements:

```cpp
// Create timer as requested in the user query
hshm::Timer timer;

// Start timing the round-trip
timer.Resume();

// Send message and receive response
auto bulk = client.Expose(url, message.c_str(), message.length(), 0);
auto send_event = client.Send(bulk);
auto recv_event = client.Recv(buffer, sizeof(buffer), url);

// Stop timing
timer.Pause();

// Get round-trip time in microseconds
double rtt_usec = timer.GetUsec();
```

### Transport Comparison
- **TCP (ZeroMQ)**: Typically lower latency for small messages
- **RDMA (Libfabric)**: Better for high-throughput workloads

### Graceful Shutdown
- Server: Use Ctrl+C for graceful shutdown
- Client: Automatically disconnects after benchmark completion

## Troubleshooting

### RDMA Issues
If RDMA tests fail:
1. Ensure Libfabric is properly installed
2. Check that RDMA devices are available: `ibv_devices`
3. Try TCP transport first to verify basic functionality

### Connection Issues
- Ensure server is started before client
- Check firewall settings for the specified ports
- Verify URLs match between client and server 