#include "hermes_shm/lightbeam/lightbeam.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

// Global flag for graceful shutdown
std::atomic<bool> server_running{true};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Shutting down gracefully..." << std::endl;
    server_running = false;
}

void run_echo_server(hshm::lbm::TransportType transport_type, const std::string& url) {
    std::cout << "\n=== Starting Echo Server: " << (transport_type == hshm::lbm::TransportType::TCP ? "TCP" : "RDMA") << " Transport ===" << std::endl;
    std::cout << "URL: " << url << std::endl;
    
    // Create server using Luke Logan's concrete class interface
    hshm::lbm::Server server;
    
    // Start the server
    std::cout << "Starting server..." << std::endl;
    server.StartServer(url, transport_type);
    
    if (!server.IsRunning()) {
        std::cerr << "Failed to start server!" << std::endl;
        return;
    }
    
    std::cout << "Server started successfully and listening on " << url << std::endl;
    std::cout << "Waiting for client connections..." << std::endl;
    std::cout << "Press Ctrl+C to stop the server." << std::endl;
    
    // Server message processing loop
    size_t message_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (server_running && server.IsRunning()) {
        // Process incoming messages
        server.ProcessMessages();
        
        // Small sleep to prevent busy waiting and allow other threads to run
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        // Print status update every 10 seconds
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
        if (elapsed.count() > 0 && elapsed.count() % 10 == 0) {
            static auto last_print_time = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_print_time).count() >= 10) {
                std::cout << "Server status: Running for " << elapsed.count() 
                          << " seconds, processed " << message_count << " messages" << std::endl;
                last_print_time = current_time;
            }
        }
    }
    
    // Shutdown
    std::cout << "\nShutting down server..." << std::endl;
    server.Stop();
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "=== Server Statistics ===" << std::endl;
    std::cout << "Transport Type: " << (transport_type == hshm::lbm::TransportType::TCP ? "TCP (ZeroMQ)" : "RDMA (Libfabric)") << std::endl;
    std::cout << "Total Runtime: " << total_time.count() << " seconds" << std::endl;
    std::cout << "Messages Processed: " << message_count << std::endl;
    if (total_time.count() > 0) {
        std::cout << "Average Message Rate: " << (double)message_count / total_time.count() << " messages/second" << std::endl;
    }
    std::cout << "Server stopped." << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "LightBeam Echo Server Test" << std::endl;
    std::cout << "==========================" << std::endl;
    
    // Set up signal handling for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Parse command line arguments
    hshm::lbm::TransportType transport = hshm::lbm::TransportType::TCP;
    std::string url = "tcp://127.0.0.1:5555";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rdma") {
            transport = hshm::lbm::TransportType::RDMA;
            url = "verbs://127.0.0.1:5556";
        } else if (arg == "--tcp") {
            transport = hshm::lbm::TransportType::TCP;
            url = "tcp://127.0.0.1:5555";
        } else if (arg == "--url" && i + 1 < argc) {
            url = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --tcp         Use TCP transport (default)" << std::endl;
            std::cout << "  --rdma        Use RDMA transport" << std::endl;
            std::cout << "  --url <url>   Server URL (default: tcp://127.0.0.1:5555)" << std::endl;
            std::cout << "  --help        Show this help message" << std::endl;
            std::cout << std::endl;
            std::cout << "The server runs an echo service that responds to client messages." << std::endl;
            std::cout << "Use Ctrl+C to stop the server gracefully." << std::endl;
            return 0;
        }
    }
    
    // Run the echo server
    try {
        run_echo_server(transport, url);
        std::cout << "\nServer shutdown completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 