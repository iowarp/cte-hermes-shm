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

class ThalliumUrl {
public:
    ThalliumUrl(const std::string &protocol, const std::string &domain, const std::string &addr, int port)
        : protocol_(protocol), domain_(domain), addr_(addr), port_(port) {}

    std::string Build() const {
        std::string url = protocol_;
        if (!url.empty()) url += "://";
        if (!domain_.empty()) {
            url += domain_ + ":";
        }
        if (!addr_.empty()) {
            url += addr_;
        }
        if (port_ > 0) {
            url += ":" + std::to_string(port_);
        }
        return url;
    }
    // For lookup, if addr_ already contains '://', return as is, else build full url
    std::string BuildForLookup() const {
        if (addr_.find("://") != std::string::npos) {
            return addr_;
        }
        return Build();
    }
private:
    std::string protocol_;
    std::string domain_;
    std::string addr_;
    int port_;
};

class ThalliumClient : public Client {
public:
    explicit ThalliumClient(const std::string &addr, const std::string &protocol = "ofi+sockets", int port = 8200, const std::string &domain = "")
        : addr_(addr), protocol_(protocol), port_(port), domain_(domain), engine_(nullptr), rpc_name_("bulk_send") {
        ThalliumUrl url_builder(protocol_, domain_, addr_, port_);
        std::string full_url = url_builder.Build();
        auto proto_end = full_url.find("://");
        std::string proto = (proto_end != std::string::npos) ? full_url.substr(0, proto_end) : full_url;
        if (proto == "ofi+tcp" || proto == "ofi+sockets") {
            engine_ = std::make_unique<thallium::engine>(proto, THALLIUM_CLIENT_MODE, true, 1);
        } else {
            engine_ = std::make_unique<thallium::engine>(full_url, THALLIUM_CLIENT_MODE, true, 1);
        }
        std::cout << "[ThalliumClient] Created with protocol: " << proto << std::endl;
        full_url_ = full_url;
        lookup_url_ = url_builder.BuildForLookup();
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
        Event* event = new Event();
        try {
            std::cout << "[ThalliumClient] Defining RPC: " << rpc_name_ << std::endl;
            thallium::remote_procedure rpc = engine_->define(rpc_name_);
            std::cout << "[ThalliumClient] Looking up server: " << lookup_url_ << std::endl;
            thallium::endpoint server = engine_->lookup(lookup_url_);
            std::cout << "[ThalliumClient] Sending data of size: " << bulk.size << std::endl;
            std::vector<char> buf_vec(bulk.data, bulk.data + bulk.size);
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
        return event;
    }

private:
    std::string addr_;
    std::string protocol_;
    int port_;
    std::string domain_;
    std::string full_url_;
    std::string lookup_url_;
    std::string rpc_name_;
    std::unique_ptr<thallium::engine> engine_;
};

class ThalliumServer : public Server {
public:
    explicit ThalliumServer(const std::string &addr, const std::string &protocol = "ofi+sockets", int port = 8200, const std::string &domain = "")
        : addr_(addr), protocol_(protocol), port_(port), domain_(domain), engine_(nullptr), rpc_name_("bulk_send") {
        try {
            ThalliumUrl url_builder(protocol_, domain_, addr_, port_);
            std::string full_url = url_builder.Build();
            auto proto_end = full_url.find("://");
            std::string proto = (proto_end != std::string::npos) ? full_url.substr(0, proto_end) : full_url;
            std::cout << "[ThalliumServer] Creating server with protocol: " << proto << std::endl;
            std::cout << "[ThalliumServer] Using full URL: " << full_url << std::endl;
            if (proto == "ofi+tcp" || proto == "ofi+sockets") {
                engine_ = std::make_unique<thallium::engine>(proto, THALLIUM_SERVER_MODE, true, 1);
            } else {
                engine_ = std::make_unique<thallium::engine>(full_url, THALLIUM_SERVER_MODE, true, 1);
            }
            std::cout << "[ThalliumServer] Engine created, defining RPC: " << rpc_name_ << std::endl;
            engine_->define(rpc_name_, [this](const thallium::request &req, const std::vector<char> &buf) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                received_queue_.push(buf);
                std::cout << "[ThalliumServer] RPC handler called with data size: " << buf.size() << std::endl;
                std::cout << "[ThalliumServer] Data: " << std::string(buf.begin(), buf.end()) << std::endl;
                std::cout << "[ThalliumServer] RPC handler: queued data, queue size = " << received_queue_.size() << std::endl;
                std::cout << "[ThalliumServer] RPC handler completed successfully" << std::endl;
            });
            std::string server_addr = engine_->self();
            std::cout << "[ThalliumServer] Server address: " << server_addr << std::endl;
            full_url_ = full_url;
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("ThalliumServer failed to initialize with addr '") + addr + "': " + e.what());
        }
    }
    
    void Start() {
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
        Event* event = new Event();
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!received_queue_.empty()) {
            const auto& data = received_queue_.front();
            size_t copy_size = std::min(bulk.size, data.size());
            std::memcpy(bulk.data, data.data(), copy_size);
            std::cout << "[ThalliumServer] Recv: copied " << copy_size << " bytes" << std::endl;
            std::cout << "[ThalliumServer] Recv: data = " << std::string(bulk.data, copy_size) << std::endl;
            received_queue_.pop();
            event->is_done = true;
            event->bytes_transferred = copy_size;
        } else {
            event->is_done = false;
        }
        return event;
    }
    
    std::string GetAddress() const override {
        if (engine_) {
            return engine_->self();
        }
        return "";
    }
private:
    std::string addr_;
    std::string protocol_;
    int port_;
    std::string domain_;
    std::string full_url_;
    std::string rpc_name_;
    std::unique_ptr<thallium::engine> engine_;
    std::queue<std::vector<char>> received_queue_;
    std::mutex queue_mutex_;
};

} // namespace hshm::lbm 