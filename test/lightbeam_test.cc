#include "hermes_shm/lightbeam/transport_factory.h"
#include "hermes_shm/lightbeam/types.h"
#include "hermes_shm/lightbeam/lightbeam.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <string>

void test_transport(hshm::lbm::TransportType transport_type, const std::string& url) {
    std::cout << "\n=== Testing " << (transport_type == hshm::lbm::TransportType::TCP ? "TCP" : "RDMA") << " Transport ===" << std::endl;
    
    // Create server and client
    auto server = hshm::lbm::Transport::CreateServer(transport_type);
    auto client = hshm::lbm::Transport::CreateClient(transport_type);
    
    if (!server || !client) {
        std::cerr << "Failed to create " << (transport_type == hshm::lbm::TransportType::TCP ? "TCP" : "RDMA") << " transport objects" << std::endl;
        return;
    }
    
    // Start server
    server->StartServer(url, transport_type);
    if (!server->IsRunning()) {
        std::cerr << "Server failed to start" << std::endl;
        return;
    }
    
    // Give server time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // For RDMA, process server messages while client is connecting
    std::thread server_thread;
    bool stop_server_processing = false;
    
    if (transport_type == hshm::lbm::TransportType::RDMA) {
        server_thread = std::thread([&server, &stop_server_processing]() {
            while (!stop_server_processing) {
                server->ProcessMessages();
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); // More frequent processing
            }
        });
        
        // Give server thread time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Connect client
    std::cout << "About to connect client..." << std::endl;
    client->Connect(url, transport_type);
    
    // For RDMA, give extra time for connection handshake
    if (transport_type == hshm::lbm::TransportType::RDMA) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    } else {
        // Give more time for connection to establish
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    // Test message exchange
    std::string test_message = "Hello from " + std::string(transport_type == hshm::lbm::TransportType::TCP ? "TCP" : "RDMA") + " client!";
    
    // Expose data for sending
    auto bulk = client->Expose(url, test_message.c_str(), test_message.length(), 0);
    
    // Send message
    auto send_event = client->Send(bulk);
    if (send_event && send_event->is_done) {
        std::cout << "Message sent successfully: " << test_message << std::endl;
    } else {
        std::cerr << "Failed to send message: " << (send_event ? send_event->error_message : "Unknown error") << std::endl;
    }
    
    // Process server messages
    for (int i = 0; i < 20; ++i) {
        server->ProcessMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Receive response
    char recv_buffer[1024];
    auto recv_event = client->Recv(recv_buffer, sizeof(recv_buffer), url);
    if (recv_event && recv_event->is_done) {
        std::cout << "Response received: " << recv_buffer << std::endl;
    } else {
        std::cerr << "Failed to receive response: " << (recv_event ? recv_event->error_message : "Unknown error") << std::endl;
    }
    
    // Process client completions
    client->ProcessCompletions();
    
    // Clean up server thread
    if (transport_type == hshm::lbm::TransportType::RDMA) {
        stop_server_processing = true;
        if (server_thread.joinable()) {
            server_thread.join();
        }
    }
    
    // Cleanup
    client->Disconnect(url);
    server->Stop();
    
    std::cout << "=== " << (transport_type == hshm::lbm::TransportType::TCP ? "TCP" : "RDMA") << " Test Complete ===" << std::endl;
}

int main() {
    std::cout << "Starting LightBeam Transport Tests" << std::endl;
    
    // Test TCP transport
    test_transport(hshm::lbm::TransportType::TCP, "tcp://127.0.0.1:5555");
    
    // Test RDMA transport
    test_transport(hshm::lbm::TransportType::RDMA, "verbs://127.0.0.1:5556");
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}