# HSHM Singleton Utilities Guide

## Overview

The Singleton Utilities API in Hermes Shared Memory (HSHM) provides multiple singleton patterns optimized for different use cases, including thread safety, cross-device compatibility, and performance requirements. These utilities enable global state management across complex applications and shared memory systems.

## Singleton Variants

### Basic Singleton (Thread-Safe)

```cpp
#include "hermes_shm/util/singleton.h"

class DatabaseConfig {
public:
    std::string connection_string;
    int max_connections;
    
    DatabaseConfig() {
        connection_string = "localhost:5432";
        max_connections = 100;
    }
    
    void Configure(const std::string& host, int max_conn) {
        connection_string = host;
        max_connections = max_conn;
    }
};

// Thread-safe singleton access
DatabaseConfig* config = hshm::Singleton<DatabaseConfig>::GetInstance();
config->Configure("prod-db:5432", 200);

// Multiple access from different threads
void worker_thread() {
    DatabaseConfig* cfg = hshm::Singleton<DatabaseConfig>::GetInstance();
    printf("Connecting to: %s\n", cfg->connection_string.c_str());
}
```

### Lockfree Singleton (High Performance)

```cpp
class MetricsCollector {
    std::atomic<size_t> counter_;
    
public:
    MetricsCollector() : counter_(0) {}
    
    void Increment() {
        counter_.fetch_add(1, std::memory_order_relaxed);
    }
    
    size_t GetCount() const {
        return counter_.load(std::memory_order_relaxed);
    }
};

// High-performance singleton without locking overhead
void hot_path_function() {
    auto* metrics = hshm::LockfreeSingleton<MetricsCollector>::GetInstance();
    metrics->Increment();  // Very fast, no locks
}
```

### Cross-Device Singleton

```cpp
class GPUManager {
public:
    int device_count;
    std::vector<int> available_devices;
    
    GPUManager() {
        device_count = GetGPUCount();
        InitializeDevices();
    }
    
private:
    int GetGPUCount();
    void InitializeDevices();
};

// Works on both host and GPU code
HSHM_CROSS_FUN
void initialize_cuda_context() {
    GPUManager* gpu_mgr = hshm::CrossSingleton<GPUManager>::GetInstance();
    printf("Found %d GPU devices\n", gpu_mgr->device_count);
}

// Lockfree version for GPU performance
HSHM_CROSS_FUN
void gpu_kernel_function() {
    auto* gpu_mgr = hshm::LockfreeCrossSingleton<GPUManager>::GetInstance();
    // Access without locking overhead in GPU kernels
}
```

### Global Singleton (Eager Initialization)

```cpp
class Logger {
public:
    std::ofstream log_file;
    std::mutex log_mutex;
    
    Logger() {
        log_file.open("/var/log/application.log", std::ios::app);
        printf("Logger initialized during program startup\n");
    }
    
    void Log(const std::string& message) {
        std::lock_guard<std::mutex> lock(log_mutex);
        log_file << "[" << GetTimestamp() << "] " << message << std::endl;
    }
    
private:
    std::string GetTimestamp();
};

// Initialized immediately when program starts
Logger* logger = hshm::GlobalSingleton<Logger>::GetInstance();

void application_function() {
    // Logger already exists and is ready
    hshm::GlobalSingleton<Logger>::GetInstance()->Log("Function called");
}
```

### Platform-Aware Global Singleton

```cpp
class NetworkManager {
public:
    std::string local_hostname;
    std::vector<std::string> network_interfaces;
    
    NetworkManager() {
        DiscoverNetworkInterfaces();
        printf("Network manager initialized\n");
    }
    
private:
    void DiscoverNetworkInterfaces();
};

// Automatically chooses best implementation for platform
HSHM_CROSS_FUN
void network_operation() {
    auto* net_mgr = hshm::GlobalCrossSingleton<NetworkManager>::GetInstance();
    printf("Local hostname: %s\n", net_mgr->local_hostname.c_str());
}
```

## C-Style Global Variable Singletons

### Basic Global Variables

```cpp
// Header declaration
HSHM_DEFINE_GLOBAL_VAR_H(DatabaseConfig, g_db_config);

// Source file definition  
HSHM_DEFINE_GLOBAL_VAR_CC(DatabaseConfig, g_db_config);

// Usage
void configure_database() {
    DatabaseConfig* config = HSHM_GET_GLOBAL_VAR(DatabaseConfig, g_db_config);
    config->Configure("prod:5432", 500);
}
```

### Cross-Platform Global Variables

```cpp
class SharedMemoryPool {
public:
    size_t pool_size;
    void* memory_base;
    
    SharedMemoryPool() : pool_size(0), memory_base(nullptr) {
        InitializePool();
    }
    
private:
    void InitializePool();
};

// Header - works on host and device
HSHM_DEFINE_GLOBAL_CROSS_VAR_H(SharedMemoryPool, g_memory_pool);

// Source file
HSHM_DEFINE_GLOBAL_CROSS_VAR_CC(SharedMemoryPool, g_memory_pool);

// Usage in cross-platform code
HSHM_CROSS_FUN
void allocate_from_pool(size_t size) {
    SharedMemoryPool* pool = HSHM_GET_GLOBAL_CROSS_VAR(SharedMemoryPool, g_memory_pool);
    // Allocation logic here
}
```

### Pointer-Based Global Variables

```cpp
class TaskScheduler {
public:
    std::queue<std::function<void()>> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool running;
    
    TaskScheduler() : running(true) {
        printf("Task scheduler created\n");
    }
    
    void SubmitTask(std::function<void()> task);
    void ProcessTasks();
    void Shutdown();
};

// Header - pointer version for lazy initialization
HSHM_DEFINE_GLOBAL_PTR_VAR_H(TaskScheduler, g_task_scheduler);

// Source file
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(TaskScheduler, g_task_scheduler);

// Usage - automatically creates instance on first access
void submit_work() {
    TaskScheduler* scheduler = HSHM_GET_GLOBAL_PTR_VAR(TaskScheduler, g_task_scheduler);
    
    scheduler->SubmitTask([]() {
        printf("Task executing\n");
    });
}
```

### Cross-Platform Pointer Variables

```cpp
class DeviceMemoryManager {
public:
    size_t total_memory;
    size_t available_memory;
    std::map<void*, size_t> allocations;
    
    DeviceMemoryManager() {
        QueryDeviceMemory();
    }
    
private:
    void QueryDeviceMemory();
};

// Header
HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_H(DeviceMemoryManager, g_device_memory);

// Source file  
HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_CC(DeviceMemoryManager, g_device_memory);

// Cross-platform usage
HSHM_CROSS_FUN
void* allocate_device_memory(size_t size) {
    DeviceMemoryManager* mgr = HSHM_GET_GLOBAL_CROSS_PTR_VAR(DeviceMemoryManager, g_device_memory);
    // Device-specific allocation
    return nullptr; // Implementation specific
}
```

## Best Practices

1. **Thread Safety**: Use `Singleton<T>` for thread-safe access, `LockfreeSingleton<T>` only with thread-safe types
2. **Cross-Platform Code**: Use `CrossSingleton<T>` and `GlobalCrossSingleton<T>` for code that runs on both host and device
3. **Python Compatibility**: Avoid standard singletons in code called by Python; use global variables instead
4. **Eager vs Lazy**: Use `GlobalSingleton<T>` for resources needed at startup, regular singletons for lazy initialization
5. **Resource Management**: Implement proper destructors and cleanup in singleton classes
6. **Configuration**: Use singletons for application-wide configuration and settings
7. **Performance**: Use lockfree variants in performance-critical paths with appropriate atomic types
8. **Memory Management**: Be aware that singletons live for the entire program duration
9. **Testing**: Design singleton classes to be testable by allowing dependency injection where possible
10. **Documentation**: Document singleton lifetime and thread safety guarantees for each singleton class
