#include "hermes_shm/lightbeam/transport_factory.h"
#include "hermes_shm/lightbeam/types.h"
#include "hermes_shm/lightbeam/lightbeam.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <cassert>

class TestResult {
public:
    bool passed = false;
    std::string test_name;
    std::string error_message;
    
    TestResult(const std::string& name) : test_name(name) {}
    
    void pass() { passed = true; }
    void fail(const std::string& error) { 
        passed = false; 
        error_message = error; 
    }
    
    void print() const {
        std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name;
        if (!passed) {
            std::cout << " - " << error_message;
        }
        std::cout << std::endl;
    }
};

class LightBeamTester {
private:
    std::vector<TestResult> results;
    
public:
    void test_multiple_concurrent_operations(hshm::lbm::TransportType transport_type, const std::string& url) {
        std::string transport_name = (transport_type == hshm::lbm::TransportType::TCP ? "TCP" : "RDMA");
        TestResult result("Multiple Concurrent Operations - " + transport_name);
        
        try {
            // Create server and client
            auto server = hshm::lbm::Transport::CreateServer(transport_type);
            auto client = hshm::lbm::Transport::CreateClient(transport_type);
            
            if (!server || !client) {
                result.fail("Failed to create transport objects");
                results.push_back(result);
                return;
            }
            
            // Start server
            server->StartServer(url, transport_type);
            if (!server->IsRunning()) {
                result.fail("Server failed to start");
                results.push_back(result);
                return;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            // For RDMA, process server messages in background
            std::thread server_thread;
            bool stop_server_processing = false;
            
            if (transport_type == hshm::lbm::TransportType::RDMA) {
                server_thread = std::thread([&server, &stop_server_processing]() {
                    while (!stop_server_processing) {
                        server->ProcessMessages();
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                });
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // Connect client
            client->Connect(url, transport_type);
            std::this_thread::sleep_for(std::chrono::milliseconds(transport_type == hshm::lbm::TransportType::RDMA ? 1000 : 500));
            
            // Test multiple concurrent messages
            std::vector<std::string> test_messages = {
                "Message 1: Hello World!",
                "Message 2: Concurrent test",
                "Message 3: Binary data test"
            };
            
            std::vector<std::unique_ptr<hshm::lbm::Event>> send_events;
            std::vector<std::unique_ptr<hshm::lbm::Event>> recv_events;
            std::vector<std::unique_ptr<char[]>> recv_buffers;
            
            // Send multiple messages
            for (const auto& message : test_messages) {
                auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
                auto event = client->Send(bulk);
                send_events.push_back(std::move(event));
            }
            
            // Post multiple receives
            for (size_t i = 0; i < test_messages.size(); ++i) {
                auto buffer = std::make_unique<char[]>(1024);
                auto event = client->Recv(buffer.get(), 1024, url);
                recv_events.push_back(std::move(event));
                recv_buffers.push_back(std::move(buffer));
            }
            
            // Process operations
            auto start_time = std::chrono::steady_clock::now();
            auto timeout = std::chrono::seconds(10);
            
            while (std::chrono::steady_clock::now() - start_time < timeout) {
                if (transport_type == hshm::lbm::TransportType::TCP) {
                    for (int i = 0; i < 5; ++i) {
                        server->ProcessMessages();
                    }
                }
                
                client->ProcessCompletions(10.0);
                
                // Check if all operations completed
                bool all_sends_done = true;
                bool all_recvs_done = true;
                
                for (auto& event : send_events) {
                    if (!event->is_done) {
                        all_sends_done = false;
                        break;
                    }
                }
                
                for (auto& event : recv_events) {
                    if (!event->is_done) {
                        all_recvs_done = false;
                        break;
                    }
                }
                
                if (all_sends_done && all_recvs_done) {
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            // Verify results
            int successful_sends = 0;
            int successful_recvs = 0;
            
            for (auto& event : send_events) {
                if (event->is_done && event->error_code == 0) {
                    successful_sends++;
                }
            }
            
            for (auto& event : recv_events) {
                if (event->is_done && event->error_code == 0) {
                    successful_recvs++;
                }
            }
            
            if (successful_sends == test_messages.size() && successful_recvs == test_messages.size()) {
                result.pass();
            } else {
                result.fail("Not all operations completed successfully (sends: " + 
                           std::to_string(successful_sends) + "/" + std::to_string(test_messages.size()) +
                           ", recvs: " + std::to_string(successful_recvs) + "/" + std::to_string(test_messages.size()) + ")");
            }
            
            // Cleanup
            if (transport_type == hshm::lbm::TransportType::RDMA) {
                stop_server_processing = true;
                if (server_thread.joinable()) {
                    server_thread.join();
                }
            }
            
            client->Disconnect(url);
            server->Stop();
            
        } catch (const std::exception& e) {
            result.fail(std::string("Exception: ") + e.what());
        }
        
        results.push_back(result);
    }
    
    void test_binary_data(hshm::lbm::TransportType transport_type, const std::string& url) {
        std::string transport_name = (transport_type == hshm::lbm::TransportType::TCP ? "TCP" : "RDMA");
        TestResult result("Binary Data Support - " + transport_name);
        
        try {
            auto server = hshm::lbm::Transport::CreateServer(transport_type);
            auto client = hshm::lbm::Transport::CreateClient(transport_type);
            
            server->StartServer(url, transport_type);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            std::thread server_thread;
            bool stop_server_processing = false;
            
            if (transport_type == hshm::lbm::TransportType::RDMA) {
                server_thread = std::thread([&server, &stop_server_processing]() {
                    while (!stop_server_processing) {
                        server->ProcessMessages();
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                });
            }
            
            client->Connect(url, transport_type);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            // Create binary test data (non-string)
            constexpr size_t binary_size = 256;
            auto binary_data = std::make_unique<uint8_t[]>(binary_size);
            for (size_t i = 0; i < binary_size; ++i) {
                binary_data[i] = static_cast<uint8_t>(i % 256);
            }
            
            auto bulk = client->Expose(url, reinterpret_cast<char*>(binary_data.get()), binary_size, 0);
            auto send_event = client->Send(bulk);
            
            auto recv_buffer = std::make_unique<char[]>(binary_size + 64);
            auto recv_event = client->Recv(recv_buffer.get(), binary_size + 64, url);
            
            // Process operations
            auto start_time = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - start_time < std::chrono::seconds(5)) {
                if (transport_type == hshm::lbm::TransportType::TCP) {
                    server->ProcessMessages();
                }
                client->ProcessCompletions(10.0);
                
                if (send_event->is_done && recv_event->is_done) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Verify binary data integrity
            if (send_event->is_done && recv_event->is_done && 
                send_event->error_code == 0 && recv_event->error_code == 0 &&
                recv_event->bytes_transferred == binary_size) {
                
                bool data_match = true;
                for (size_t i = 0; i < binary_size; ++i) {
                    if (static_cast<uint8_t>(recv_buffer[i]) != static_cast<uint8_t>(i % 256)) {
                        data_match = false;
                        break;
                    }
                }
                
                if (data_match) {
                    result.pass();
                } else {
                    result.fail("Binary data corruption detected");
                }
            } else {
                result.fail("Binary data transfer failed");
            }
            
            // Cleanup
            if (transport_type == hshm::lbm::TransportType::RDMA) {
                stop_server_processing = true;
                if (server_thread.joinable()) {
                    server_thread.join();
                }
            }
            
            client->Disconnect(url);
            server->Stop();
            
        } catch (const std::exception& e) {
            result.fail(std::string("Exception: ") + e.what());
        }
        
        results.push_back(result);
    }
    
    void test_event_lifecycle(hshm::lbm::TransportType transport_type, const std::string& url) {
        std::string transport_name = (transport_type == hshm::lbm::TransportType::TCP ? "TCP" : "RDMA");
        TestResult result("Event Lifecycle Management - " + transport_name);
        
        try {
            auto server = hshm::lbm::Transport::CreateServer(transport_type);
            auto client = hshm::lbm::Transport::CreateClient(transport_type);
            
            server->StartServer(url, transport_type);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            std::thread server_thread;
            bool stop_server_processing = false;
            
            if (transport_type == hshm::lbm::TransportType::RDMA) {
                server_thread = std::thread([&server, &stop_server_processing]() {
                    while (!stop_server_processing) {
                        server->ProcessMessages();
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                });
            }
            
            client->Connect(url, transport_type);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            // Test event uniqueness
            std::string message = "Test message for event lifecycle";
            auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
            
            auto event1 = client->Send(bulk);
            auto event2 = client->Send(bulk);
            
            // Events should have unique IDs
            if (event1->event_id == event2->event_id) {
                result.fail("Events do not have unique IDs");
            } else if (event1->event_id == 0 || event2->event_id == 0) {
                result.fail("Events have invalid IDs");
            } else {
                // Test initial state
                if (event1->operation_type != hshm::lbm::OperationType::SEND ||
                    event1->transport_used != transport_type) {
                    result.fail("Event metadata incorrect");
                } else {
                    result.pass();
                }
            }
            
            // Cleanup
            if (transport_type == hshm::lbm::TransportType::RDMA) {
                stop_server_processing = true;
                if (server_thread.joinable()) {
                    server_thread.join();
                }
            }
            
            client->Disconnect(url);
            server->Stop();
            
        } catch (const std::exception& e) {
            result.fail(std::string("Exception: ") + e.what());
        }
        
        results.push_back(result);
    }
    
    void run_all_tests() {
        std::cout << "\n🧪 **Running Enhanced LightBeam Tests**\n" << std::endl;
        
        // Test TCP
        std::cout << "Testing TCP Transport:" << std::endl;
        test_multiple_concurrent_operations(hshm::lbm::TransportType::TCP, "tcp://127.0.0.1:5555");
        test_binary_data(hshm::lbm::TransportType::TCP, "tcp://127.0.0.1:5557");
        test_event_lifecycle(hshm::lbm::TransportType::TCP, "tcp://127.0.0.1:5558");
        
        std::cout << "\nTesting RDMA Transport:" << std::endl;
        test_multiple_concurrent_operations(hshm::lbm::TransportType::RDMA, "verbs://127.0.0.1:5556");
        test_binary_data(hshm::lbm::TransportType::RDMA, "verbs://127.0.0.1:5559");
        test_event_lifecycle(hshm::lbm::TransportType::RDMA, "verbs://127.0.0.1:5560");
        
        // Print results
        std::cout << "\n📊 **Test Results Summary**\n" << std::endl;
        int passed = 0;
        int total = results.size();
        
        for (const auto& result : results) {
            result.print();
            if (result.passed) passed++;
        }
        
        std::cout << "\n✅ " << passed << "/" << total << " tests passed";
        if (passed == total) {
            std::cout << " - All tests successful! 🎉";
        }
        std::cout << std::endl;
    }
};

int main() {
    std::cout << "Enhanced LightBeam Test Suite" << std::endl;
    std::cout << "=============================" << std::endl;
    
    LightBeamTester tester;
    tester.run_all_tests();
    
    return 0;
}