#pragma once
#include <thallium.hpp>
#include <cereal/types/vector.hpp>
#include <margo.h>
#include "lightbeam.h"
#include <queue>
#include <mutex>
#include <memory>
#include <vector>
#include <algorithm>

namespace hshm::lbm {

class ThalliumClient : public Client {
public:
    explicit ThalliumClient(const std::string &url)
        : url_(url), engine_(nullptr), rpc_name_("bulk_send") {
        auto proto_end = url.find("://");
        protocol_ = (proto_end != std::string::npos) ? url.substr(0, proto_end) : url;
        // For ofi+tcp and ofi+sockets, use just the protocol, not the full URL
        if (protocol_ == "ofi+tcp" || protocol_ == "ofi+sockets") {
            engine_ = std::make_unique<thallium::engine>(protocol_, THALLIUM_CLIENT_MODE, true, 1);
        } else {
            // For other protocols like na+sm, use the full URL
            engine_ = std::make_unique<thallium::engine>(url_, THALLIUM_CLIENT_MODE, true, 1);
        }
        std::cout << "[ThalliumClient] Created with protocol: " << protocol_ << std::endl;
    }
    ~ThalliumClient() override {
        if (engine_) engine_->finalize();
    }
    Bulk Expose(const char *data, size_t data_size, int flags) override {
        Bulk bulk;
        bulk.data = const_cast<char*>(data);
        bulk.size = data_size;
        bulk.flags = flags;
        return bulk;
    }
    Event* Send(const Bulk &bulk) override {
        auto event = std::make_unique<Event>();
        try {
            std::cout << "[ThalliumClient] Defining RPC: " << rpc_name_ << std::endl;
            thallium::remote_procedure rpc = engine_->define(rpc_name_);
            
            std::cout << "[ThalliumClient] Looking up server: " << url_ << std::endl;
            thallium::endpoint server = engine_->lookup(url_);
            std::cout << "[ThalliumClient] Sending data of size: " << bulk.size << std::endl;
            std::vector<char> buf_vec(bulk.data, bulk.data + bulk.size);
            // Follow reference: disable_response() for void returns
            rpc.disable_response();
            rpc.on(server)(buf_vec);
            std::cout << "[ThalliumClient] RPC call completed successfully" << std::endl;
            event->is_done = true;
            event->bytes_transferred = bulk.size;
        } catch (const thallium::margo_exception &e) {
            std::cout << "[ThalliumClient] Margo exception: " << e.what() << std::endl;
            event->is_done = true;
            event->error_code = -1;
            event->error_message = e.what();
        } catch (const std::exception &e) {
            std::cout << "[ThalliumClient] Exception: " << e.what() << std::endl;
            std::cout << "[ThalliumClient] Exception type: " << typeid(e).name() << std::endl;
            event->is_done = true;
            event->error_code = -1;
            event->error_message = e.what();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push(std::move(event));
        return events_.back().get();
    }

private:
    std::string url_;
    std::string protocol_;
    std::string rpc_name_;
    std::unique_ptr<thallium::engine> engine_;
    std::queue<std::unique_ptr<Event>> events_;
    std::mutex mutex_;
};

class ThalliumServer : public Server {
public:
    explicit ThalliumServer(const std::string &url)
        : url_(url), engine_(nullptr), rpc_name_("bulk_send"), has_data_(false) {
        auto proto_end = url.find("://");
        protocol_ = (proto_end != std::string::npos) ? url.substr(0, proto_end) : url;
        std::cout << "[ThalliumServer] Creating server with protocol: " << protocol_ << std::endl;
        std::cout << "[ThalliumServer] Using full URL: " << url_ << std::endl;
        // For ofi+tcp and ofi+sockets, use just the protocol, not the full URL
        if (protocol_ == "ofi+tcp" || protocol_ == "ofi+sockets") {
            engine_ = std::make_unique<thallium::engine>(protocol_, THALLIUM_SERVER_MODE);
        } else {
            // For other protocols like na+sm, use the full URL
            engine_ = std::make_unique<thallium::engine>(url_, THALLIUM_SERVER_MODE);
        }
        std::cout << "[ThalliumServer] Engine created, defining RPC: " << rpc_name_ << std::endl;
        engine_->define(rpc_name_, [this](const thallium::request &req, const std::vector<char> &buf) {
            std::cout << "[ThalliumServer] RPC handler called with data size: " << buf.size() << std::endl;
            std::cout << "[ThalliumServer] Data: " << std::string(buf.begin(), buf.end()) << std::endl;
            std::lock_guard<std::mutex> lock(mutex_);
            received_data_ = buf;
            has_data_ = true;
            std::cout << "[ThalliumServer] RPC handler: stored data, has_data_ = " << has_data_ << std::endl;
            std::cout << "[ThalliumServer] RPC handler completed successfully" << std::endl;
        });
        
        // Get the server's actual address
        std::string server_addr = engine_->self();
        std::cout << "[ThalliumServer] Server address: " << server_addr << std::endl;
    }
    
    void Start() {
        // No background thread needed - Thallium handles events internally
        std::cout << "[ThalliumServer] Server ready for connections" << std::endl;
    }
    ~ThalliumServer() override {
        std::cout << "[ThalliumServer] Destructor called" << std::endl;
        if (engine_) {
            std::cout << "[ThalliumServer] Finalizing engine" << std::endl;
            engine_->finalize();
        }
    }
    Bulk Expose(char *data, size_t data_size, int flags) override {
        Bulk bulk;
        bulk.data = data;
        bulk.size = data_size;
        bulk.flags = flags;
        return bulk;
    }
    Event* Recv(const Bulk &bulk) override {
        auto event = std::make_unique<Event>();
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[ThalliumServer] Recv: checking for data, has_data_ = " << has_data_ << std::endl;
        if (has_data_) {
            size_t copy_size = std::min(bulk.size, received_data_.size());
            std::memcpy(bulk.data, received_data_.data(), copy_size);
            std::cout << "[ThalliumServer] Recv: copied " << copy_size << " bytes" << std::endl;
            std::cout << "[ThalliumServer] Recv: data = " << std::string(bulk.data, copy_size) << std::endl;
            has_data_ = false;
            event->is_done = true;
            event->bytes_transferred = copy_size;
        } else {
            std::cout << "[ThalliumServer] Recv: no data available" << std::endl;
            event->is_done = false;
        }
        events_.push(std::move(event));
        return events_.back().get();
    }
    
    std::string GetAddress() const override {
        if (engine_) {
            return engine_->self();
        }
        return "";
    }
private:
    std::string url_;
    std::string protocol_;
    std::string rpc_name_;
    std::unique_ptr<thallium::engine> engine_;
    std::vector<char> received_data_;
    bool has_data_;
    std::mutex mutex_;
    std::queue<std::unique_ptr<Event>> events_;
};

// Factory implementation for Thallium
inline std::unique_ptr<Client> TransportFactory::GetClient(const std::string &url, Transport t) {
    if (t == Transport::kThallium) {
        return std::make_unique<ThalliumClient>(url);
    }
    return nullptr;
}
inline std::unique_ptr<Server> TransportFactory::GetServer(const std::string &url, Transport t) {
    if (t == Transport::kThallium) {
        return std::make_unique<ThalliumServer>(url);
    }
    return nullptr;
}

} // namespace hshm::lbm 