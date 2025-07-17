#include "hermes_shm/lightbeam/lightbeam.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>
#include <cstring>
#include <string>

// Global running flag for signal handling
std::atomic<bool> running{true};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", stopping server..." << std::endl;
    running = false;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <ip_address> <protocol>" << std::endl;
    std::cout << "  ip_address: IP address to bind to (e.g., 127.0.0.1)" << std::endl;
    std::cout << "  protocol:   'zmq' (or 'tcp') or 'libfabric' (or 'rdma')" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " 127.0.0.1 zmq" << std::endl;
    std::cout << "  " << program_name << " 0.0.0.0 libfabric" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Simple LightBeam Echo Server" << std::endl;
    std::cout << "============================" << std::endl;
    
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string ip_address = argv[1];
    std::string protocol = argv[2];
    
    // Determine transport type and URL
    std::string url;
    
    if (protocol == "zmq" || protocol == "tcp") {
        url = "tcp://" + ip_address + ":5555";
    } else if (protocol == "libfabric" || protocol == "rdma") {
        url = "tcp://" + ip_address + ":5556";  // Using TCP transport for libfabric
    } else {
        std::cerr << "❌ ERROR: Invalid protocol: " << protocol << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "Protocol: " << protocol << std::endl;
    std::cout << "URL: " << url << std::endl;
    
    // Create server using the new server API
    hshm::lbm::Server server;
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Start the server (new API takes only URL)
    std::cout << "Starting server..." << std::endl;
    server.StartServer(url);
    
    if (!server.IsRunning()) {
        std::cerr << "❌ ERROR: Server failed to start" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Server started successfully on " << url << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    std::cout << "Note: This server is using protocol: " << protocol << std::endl;
    
    int loop_count = 0;
    auto last_report = std::chrono::steady_clock::now();
    
    // Server event loop - simplified for new API
    while (running && server.IsRunning()) {
        loop_count++;
        
        // Report every 10 seconds
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_report).count() >= 10.0) {
            std::cout << "📊 Server running... Loop iterations: " << loop_count << std::endl;
            last_report = now;
            loop_count = 0;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "\nStopping server..." << std::endl;
    server.Stop();
    std::cout << "Server stopped gracefully." << std::endl;
    
    return 0;
}