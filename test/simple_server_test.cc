#include "hermes_shm/lightbeam/lightbeam.h"
#include "hermes_shm/lightbeam/thallium/server.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>
#include <cstring>
#include <string>
#include <cereal/types/string.hpp>

// Global running flag for signal handling
std::atomic<bool> running{true};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", stopping server..." << std::endl;
    running = false;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <ip_address> <protocol>" << std::endl;
    std::cout << "  ip_address: IP address to bind to (e.g., 127.0.0.1)" << std::endl;
    std::cout << "  protocol:   'zmq' (or 'tcp') or 'thallium'" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " 127.0.0.1 zmq" << std::endl;
    std::cout << "  " << program_name << " 0.0.0.0 thallium" << std::endl;
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
    } else if (protocol == "thallium") {
        url = "tcp://" + ip_address + ":5557";
    } else {
        std::cerr << "❌ ERROR: Invalid protocol: " << protocol << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "Protocol: " << protocol << std::endl;
    std::cout << "URL: " << url << std::endl;
    
    // Create server using the selected backend
    if (protocol == "thallium") {
        hshm::lbm::thallium::Server server;
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        try {
            std::cout << "Starting Thallium server..." << std::endl;
            server.StartServer(url);
            
            // Register echo RPC AFTER starting the server
            std::cout << "Registering echo RPC..." << std::endl;
            server.RegisterRpc("echo", [](const ::thallium::request& req, const std::string &msg) {
                std::cout << "[Thallium Server] Echo handler called with: '" << msg << "'" << std::endl;
                req.respond(msg);
            });
            
            std::cout << "✅ Server started successfully on " << url << std::endl;
            std::cout << "Press Ctrl+C to stop" << std::endl;
            std::cout << "Note: This server is using protocol: " << protocol << std::endl;
            
            // Keep the main thread alive while server runs
            while (running && server.IsRunning()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            std::cout << "Stopping Thallium server..." << std::endl;
            server.Stop();
        } catch (const std::exception &e) {
            std::cerr << "❌ ERROR: " << e.what() << std::endl;
            return 1;
        }
    } else {
        hshm::lbm::Server server;
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        std::cout << "Starting ZMQ server..." << std::endl;
        server.StartServer(url);
        
        if (!server.IsRunning()) {
            std::cerr << "❌ ERROR: Server failed to start" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Server started successfully on " << url << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        std::cout << "Note: This server is using protocol: " << protocol << std::endl;
        
        // Server event loop - simplified for new API
        while (running && server.IsRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::cout << "Stopping ZMQ server..." << std::endl;
        server.Stop();
    }
    return 0;
}