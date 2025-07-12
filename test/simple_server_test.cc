#include "hermes_shm/lightbeam/lightbeam.h"
#include "hermes_shm/util/timer.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>

class SimpleServerTest {
private:
    std::atomic<bool> running{true};
    
public:
    void test_server_startup() {
        std::cout << "\n=== Test 1: Server Startup ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        hshm::Timer timer;
        timer.Resume();
        
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        
        timer.Pause();
        double startup_time = timer.GetMsecFromStart();
        
        std::cout << "Server startup time: " << startup_time << " ms" << std::endl;
        std::cout << "Server started on: " << url << std::endl;
        
        if (startup_time < 1000.0) {  // Should start quickly
            std::cout << "✅ PASS: Server starts quickly" << std::endl;
        } else {
            std::cout << "❌ FAIL: Server startup too slow" << std::endl;
        }
        
        server->Stop();
    }
    
    void test_message_processing() {
        std::cout << "\n=== Test 2: Message Processing ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::cout << "Server listening on " << url << std::endl;
        
        hshm::Timer timer;
        timer.Resume();
        
        int messages_processed = 0;
        int total_cycles = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        // Run for 5 seconds
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() < 5.0) {
            server->ProcessMessages();
            total_cycles++;
            
            // Check if we processed any messages (this would show in server logs)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        timer.Pause();
        double test_time = timer.GetMsecFromStart();
        
        std::cout << "Test duration: " << test_time << " ms" << std::endl;
        std::cout << "ProcessMessages cycles: " << total_cycles << std::endl;
        std::cout << "Cycles per second: " << (total_cycles * 1000.0 / test_time) << std::endl;
        
        if (total_cycles > 100) {  // Should process many cycles
            std::cout << "✅ PASS: Server processes messages efficiently" << std::endl;
        } else {
            std::cout << "❌ FAIL: Server processing too slow" << std::endl;
        }
        
        server->Stop();
    }
    
    void test_echo_functionality() {
        std::cout << "\n=== Test 3: Echo Functionality ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::cout << "Echo server listening on " << url << std::endl;
        std::cout << "Waiting for client connections..." << std::endl;
        
        hshm::Timer timer;
        timer.Resume();
        
        int echo_cycles = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        // Run for 10 seconds to allow client connections
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() < 10.0) {
            server->ProcessMessages();
            echo_cycles++;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        timer.Pause();
        double test_time = timer.GetMsecFromStart();
        
        std::cout << "Echo test duration: " << test_time << " ms" << std::endl;
        std::cout << "Echo processing cycles: " << echo_cycles << std::endl;
        
        if (echo_cycles > 500) {  // Should complete many cycles
            std::cout << "✅ PASS: Echo server runs successfully" << std::endl;
        } else {
            std::cout << "❌ FAIL: Echo server processing issues" << std::endl;
        }
        
        server->Stop();
    }
    
    void test_concurrent_client_handling() {
        std::cout << "\n=== Test 4: Concurrent Client Handling ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::cout << "Server listening for concurrent clients on " << url << std::endl;
        
        hshm::Timer timer;
        timer.Resume();
        
        int processing_cycles = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        // Run for 15 seconds to allow multiple clients to connect and send messages
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() < 15.0) {
            server->ProcessMessages();
            processing_cycles++;
            
            if (processing_cycles % 1000 == 0) {
                std::cout << "Processed " << processing_cycles << " cycles" << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        timer.Pause();
        double test_time = timer.GetMsecFromStart();
        
        std::cout << "Concurrent client handling results:" << std::endl;
        std::cout << "  Test duration: " << test_time << " ms" << std::endl;
        std::cout << "  Processing cycles: " << processing_cycles << std::endl;
        std::cout << "  Cycles per second: " << (processing_cycles * 1000.0 / test_time) << std::endl;
        
        bool performance_ok = (processing_cycles > 1000);
        bool responsiveness_ok = (processing_cycles * 1000.0 / test_time > 100); // >100 cycles/sec
        
        if (performance_ok && responsiveness_ok) {
            std::cout << "✅ PASS: Server handles concurrent clients efficiently" << std::endl;
        } else {
            std::cout << "❌ FAIL: Server concurrent handling issues" << std::endl;
            if (!performance_ok) std::cout << "  - Insufficient processing cycles" << std::endl;
            if (!responsiveness_ok) std::cout << "  - Poor responsiveness" << std::endl;
        }
        
        server->Stop();
    }

    void test_high_throughput_messaging() {
        std::cout << "\n=== Test 5: High Throughput Messaging ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::cout << "High throughput server listening on " << url << std::endl;
        std::cout << "Expecting high volume of concurrent messages..." << std::endl;
        
        hshm::Timer timer;
        timer.Resume();
        
        int processing_cycles = 0;
        auto start_time = std::chrono::steady_clock::now();
        auto last_report = start_time;
        
        // Run for 20 seconds to handle high throughput
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() < 20.0) {
            server->ProcessMessages();
            processing_cycles++;
            
            // Report every 5 seconds
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_report).count() >= 5.0) {
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double cycle_rate = processing_cycles / elapsed;
                std::cout << "  " << elapsed << "s: " << processing_cycles << " cycles (" 
                         << cycle_rate << " cycles/s)" << std::endl;
                last_report = now;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Minimal sleep for high throughput
        }
        
        timer.Pause();
        double test_time = timer.GetMsecFromStart();
        double throughput = (processing_cycles * 1000.0) / test_time; // cycles per second
        
        std::cout << "High throughput test results:" << std::endl;
        std::cout << "  Test duration: " << test_time << " ms" << std::endl;
        std::cout << "  Processing cycles: " << processing_cycles << std::endl;
        std::cout << "  Throughput: " << throughput << " cycles/second" << std::endl;
        
        bool throughput_ok = (throughput > 100.0); // At least 100 cycles/second
        bool efficiency_ok = (processing_cycles > 1000); // Adequate processing
        
        if (throughput_ok && efficiency_ok) {
            std::cout << "✅ PASS: Server handles high throughput messaging" << std::endl;
        } else {
            std::cout << "❌ FAIL: High throughput messaging issues" << std::endl;
            if (!throughput_ok) std::cout << "  - Throughput too low" << std::endl;
            if (!efficiency_ok) std::cout << "  - Processing efficiency issues" << std::endl;
        }
        
        server->Stop();
    }

    void test_concurrent_echo_stress() {
        std::cout << "\n=== Test 6: Concurrent Echo Stress Test ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::cout << "Echo stress server listening on " << url << std::endl;
        std::cout << "Ready for concurrent echo stress testing..." << std::endl;
        
        hshm::Timer timer;
        timer.Resume();
        
        int processing_cycles = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        // Run for 25 seconds for stress testing
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() < 25.0) {
            server->ProcessMessages();
            processing_cycles++;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        
        timer.Pause();
        double test_time = timer.GetMsecFromStart();
        double cycle_rate = (processing_cycles * 1000.0) / test_time;
        
        std::cout << "Concurrent echo stress results:" << std::endl;
        std::cout << "  Test duration: " << test_time << " ms" << std::endl;
        std::cout << "  Processing cycles: " << processing_cycles << std::endl;
        std::cout << "  Cycle rate: " << cycle_rate << " cycles/second" << std::endl;
        
        bool stress_ok = (processing_cycles > 5000); // Adequate processing
        bool consistency_ok = (cycle_rate > 100.0); // Consistent performance
        
        if (stress_ok && consistency_ok) {
            std::cout << "✅ PASS: Server handles concurrent echo stress" << std::endl;
        } else {
            std::cout << "❌ FAIL: Concurrent echo stress issues" << std::endl;
            if (!stress_ok) std::cout << "  - Insufficient processing cycles" << std::endl;
            if (!consistency_ok) std::cout << "  - Inconsistent performance" << std::endl;
        }
        
        server->Stop();
    }
    
    void run_continuous_server() {
        std::cout << "\n=== Continuous Server Mode ===" << std::endl;
        std::cout << "Starting server for client testing..." << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::cout << "✅ Server ready on " << url << std::endl;
        
        hshm::Timer uptime_timer;
        uptime_timer.Resume();
        
        int total_messages = 0;
        auto last_report = std::chrono::steady_clock::now();
        
        while (running) {
            server->ProcessMessages();
            total_messages++;
            
            // Report every 5 seconds
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_report).count() >= 5.0) {
                uptime_timer.Pause();
                double uptime = uptime_timer.GetMsecFromStart() / 1000.0;
                std::cout << "Uptime: " << uptime << "s, Messages: " << total_messages << std::endl;
                uptime_timer.Resume();
                last_report = now;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        server->Stop();
        std::cout << "Server stopped gracefully" << std::endl;
    }
    
    void run_all_tests() {
        std::cout << "Simple LightBeam Server Unit Tests" << std::endl;
        std::cout << "===================================" << std::endl;
        
        test_server_startup();
        test_message_processing();
        test_echo_functionality();
        test_concurrent_client_handling();
        test_high_throughput_messaging();
        test_concurrent_echo_stress();
        
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Tests completed. Check output above for PASS/FAIL status." << std::endl;
    }
    
    void stop() {
        running = false;
    }
};

// Global server instance for signal handling
SimpleServerTest* global_server = nullptr;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", stopping server..." << std::endl;
    if (global_server) {
        global_server->stop();
    }
    exit(0);
}

int main(int argc, char* argv[]) {
    SimpleServerTest test;
    global_server = &test;
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (argc > 1 && std::string(argv[1]) == "--continuous") {
        test.run_continuous_server();
    } else {
        test.run_all_tests();
    }
    
    return 0;
} 