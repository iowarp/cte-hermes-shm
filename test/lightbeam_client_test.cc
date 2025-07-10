#include "hermes_shm/lightbeam/lightbeam.h"
#include "hermes_shm/util/timer.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <iomanip>

void run_client_benchmark(hshm::lbm::TransportType transport_type, const std::string& url, int num_iterations = 10) {
    std::cout << "\n=== Client Benchmark: " << (transport_type == hshm::lbm::TransportType::TCP ? "TCP" : "RDMA") << " Transport ===" << std::endl;
    std::cout << "URL: " << url << std::endl;
    std::cout << "Iterations: " << num_iterations << std::endl;
    
    // Create client using Luke Logan's concrete class interface
    hshm::lbm::Client client;
    
    // Give server time to start (assuming server is already running)
    std::cout << "Waiting for server to be ready..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // Connect to server
    std::cout << "Connecting to server..." << std::endl;
    client.Connect(url, transport_type);
    
    // Give extra time for connection handshake (especially for RDMA)
    if (transport_type == hshm::lbm::TransportType::RDMA) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    std::vector<double> round_trip_times;
    round_trip_times.reserve(num_iterations);
    
    std::cout << "\nStarting benchmark..." << std::endl;
    std::cout << "Iteration | Message | Round-trip Time (μs)" << std::endl;
    std::cout << "----------|---------|--------------------" << std::endl;
    
    for (int i = 0; i < num_iterations; ++i) {
        // Prepare test message
        std::string test_message = "Benchmark message #" + std::to_string(i + 1);
        
        // Create timer as requested in the user query
        hshm::Timer timer;
        
        // Start timing the round-trip
        timer.Resume();
        
        // Expose and send message
        auto bulk = client.Expose(url, test_message.c_str(), test_message.length(), 0);
        auto send_event = client.Send(bulk);
        
        // Prepare receive buffer
        char recv_buffer[1024] = {0};
        
        // Receive response (this blocks until response is received)
        auto recv_event = client.Recv(recv_buffer, sizeof(recv_buffer), url);
        
        // Stop timing
        timer.Pause();
        
        // Process completions
        client.ProcessCompletions();
        
        // Record round-trip time
        double rtt_usec = timer.GetUsec();
        round_trip_times.push_back(rtt_usec);
        
        // Verify message was echoed back correctly
        bool success = (send_event && send_event->is_done && 
                       recv_event && recv_event->is_done &&
                       std::string(recv_buffer) == test_message);
        
        std::cout << std::setw(9) << (i + 1) 
                  << " | " << std::setw(7) << test_message.substr(0, 7) + "..."
                  << " | " << std::setw(15) << std::fixed << std::setprecision(2) << rtt_usec
                  << (success ? " ✓" : " ✗") << std::endl;
        
        if (!success) {
            std::cerr << "  Error: ";
            if (!send_event || !send_event->is_done) {
                std::cerr << "Send failed. ";
            }
            if (!recv_event || !recv_event->is_done) {
                std::cerr << "Receive failed. ";
            }
            if (std::string(recv_buffer) != test_message) {
                std::cerr << "Message mismatch (expected: '" << test_message 
                          << "', got: '" << recv_buffer << "'). ";
            }
            std::cerr << std::endl;
        }
        
        // Small delay between iterations to avoid overwhelming the server
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Calculate and display statistics
    if (!round_trip_times.empty()) {
        double sum = 0.0;
        double min_time = round_trip_times[0];
        double max_time = round_trip_times[0];
        
        for (double time : round_trip_times) {
            sum += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        double avg_time = sum / round_trip_times.size();
        
        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << "Transport Type: " << (transport_type == hshm::lbm::TransportType::TCP ? "TCP (ZeroMQ)" : "RDMA (Libfabric)") << std::endl;
        std::cout << "Total Iterations: " << num_iterations << std::endl;
        std::cout << "Average Round-trip Time: " << std::fixed << std::setprecision(2) << avg_time << " μs" << std::endl;
        std::cout << "Min Round-trip Time: " << std::fixed << std::setprecision(2) << min_time << " μs" << std::endl;
        std::cout << "Max Round-trip Time: " << std::fixed << std::setprecision(2) << max_time << " μs" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << (1000000.0 / avg_time) << " requests/second" << std::endl;
    }
    
    // Cleanup
    client.Disconnect(url);
    std::cout << "Client disconnected." << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "LightBeam Client Benchmark Test" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Parse command line arguments
    hshm::lbm::TransportType transport = hshm::lbm::TransportType::TCP;
    std::string url = "tcp://127.0.0.1:5555";
    int iterations = 10;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rdma") {
            transport = hshm::lbm::TransportType::RDMA;
            url = "verbs://127.0.0.1:5556";
        } else if (arg == "--tcp") {
            transport = hshm::lbm::TransportType::TCP;
            url = "tcp://127.0.0.1:5555";
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (arg == "--url" && i + 1 < argc) {
            url = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --tcp              Use TCP transport (default)" << std::endl;
            std::cout << "  --rdma             Use RDMA transport" << std::endl;
            std::cout << "  --url <url>        Server URL (default: tcp://127.0.0.1:5555)" << std::endl;
            std::cout << "  --iterations <n>   Number of benchmark iterations (default: 10)" << std::endl;
            std::cout << "  --help             Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Run the benchmark
    try {
        run_client_benchmark(transport, url, iterations);
        std::cout << "\nBenchmark completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 