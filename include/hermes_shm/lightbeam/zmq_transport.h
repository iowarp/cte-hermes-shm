#pragma once
#if HSHM_ENABLE_ZMQ
#include <zmq.h>

#include <memory>
#include <sstream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

#include "lightbeam.h"

// Cereal serialization for Bulk
namespace cereal {
  template<class Archive>
  void serialize(Archive& ar, hshm::lbm::Bulk& bulk) {
    ar(bulk.size, bulk.flags);
  }

  template<class Archive>
  void serialize(Archive& ar, hshm::lbm::LbmMeta& meta) {
    ar(meta.send, meta.recv);
  }
}  // namespace cereal

namespace hshm::lbm {

class ZeroMqClient : public Client {
 public:
  explicit ZeroMqClient(const std::string& addr,
                        const std::string& protocol = "tcp", int port = 8192)
      : addr_(addr),
        protocol_(protocol),
        port_(port),
        ctx_(zmq_ctx_new()),
        socket_(zmq_socket(ctx_, ZMQ_PUSH)) {
    std::string full_url =
        protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    zmq_connect(socket_, full_url.c_str());
  }

  ~ZeroMqClient() override {
    zmq_close(socket_);
    zmq_ctx_destroy(ctx_);
  }

  // Base Expose implementation - accepts hipc::FullPtr
  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) override {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  // Expose from raw pointer - calls base implementation
  Bulk Expose(const char* data, size_t data_size, u32 flags) override {
    hipc::FullPtr<char> ptr(data);
    return Expose(ptr, data_size, flags);
  }

  // Expose from hipc::Pointer - calls base implementation
  Bulk Expose(const hipc::Pointer& ptr, size_t data_size, u32 flags) override {
    return Expose(hipc::FullPtr<char>(ptr), data_size, flags);
  }

  template <typename MetaT>
  int Send(MetaT& meta) {
    // Validate all send bulks have BULK_WRITE flag
    for (const auto& bulk : meta.send) {
      if (!bulk.flags.Any(BULK_WRITE)) {
        return -1;  // Error: send bulk without BULK_WRITE flag
      }
    }

    // Serialize metadata
    std::ostringstream oss(std::ios::binary);
    {
      cereal::BinaryOutputArchive ar(oss);
      ar(meta);
    }
    std::string meta_str = oss.str();

    // Send metadata - use ZMQ_SNDMORE only if there are send bulks to follow
    int flags = 0;
    if (!meta.send.empty()) {
      flags |= ZMQ_SNDMORE;
    }

    int rc = zmq_send(socket_, meta_str.data(), meta_str.size(), flags);
    if (rc == -1) {
      return zmq_errno();
    }

    // Send all bulks in send vector
    for (size_t i = 0; i < meta.send.size(); ++i) {
      flags = 0;
      if (i < meta.send.size() - 1) {
        flags |= ZMQ_SNDMORE;
      }

      rc = zmq_send(socket_, meta.send[i].data.ptr_, meta.send[i].size, flags);
      if (rc == -1) {
        return zmq_errno();
      }
    }

    return 0;  // Success
  }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  void* ctx_;
  void* socket_;
};

class ZeroMqServer : public Server {
 public:
  explicit ZeroMqServer(const std::string& addr,
                        const std::string& protocol = "tcp", int port = 8192)
      : addr_(addr),
        protocol_(protocol),
        port_(port),
        ctx_(zmq_ctx_new()),
        socket_(zmq_socket(ctx_, ZMQ_PULL)) {
    std::string full_url =
        protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    int rc = zmq_bind(socket_, full_url.c_str());
    if (rc == -1) {
      std::string err = "ZeroMqServer failed to bind to URL '" + full_url +
                        "': " + zmq_strerror(zmq_errno());
      zmq_close(socket_);
      zmq_ctx_destroy(ctx_);
      throw std::runtime_error(err);
    }
  }

  ~ZeroMqServer() override {
    zmq_close(socket_);
    zmq_ctx_destroy(ctx_);
  }

  // Base Expose implementation - accepts hipc::FullPtr
  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) override {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  // Expose from raw pointer - calls base implementation
  Bulk Expose(char* data, size_t data_size, u32 flags) override {
    hipc::FullPtr<char> ptr(data);
    return Expose(ptr, data_size, flags);
  }

  // Expose from hipc::Pointer - calls base implementation
  Bulk Expose(const hipc::Pointer& ptr, size_t data_size, u32 flags) override {
    return Expose(hipc::FullPtr<char>(ptr), data_size, flags);
  }

  template <typename MetaT>
  int RecvMetadata(MetaT& meta) {
    // Receive metadata message (non-blocking)
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int rc = zmq_msg_recv(&msg, socket_, ZMQ_DONTWAIT);

    if (rc == -1) {
      int err = zmq_errno();
      zmq_msg_close(&msg);
      return err;  // Returns EAGAIN if no message available
    }

    // Deserialize metadata
    try {
      std::string meta_str(static_cast<char*>(zmq_msg_data(&msg)),
                           zmq_msg_size(&msg));
      std::istringstream iss(meta_str, std::ios::binary);
      cereal::BinaryInputArchive ar(iss);
      ar(meta);
    } catch (const std::exception& e) {
      zmq_msg_close(&msg);
      return -1;  // Deserialization error
    }

    zmq_msg_close(&msg);
    return 0;  // Success
  }

  template <typename MetaT>
  int RecvBulks(MetaT& meta) {
    // Validate all recv bulks have BULK_EXPOSE flag
    for (const auto& bulk : meta.recv) {
      if (!bulk.flags.Any(BULK_EXPOSE)) {
        return -1;  // Error: recv bulk without BULK_EXPOSE flag
      }
    }

    // If no recv bulks, return immediately
    if (meta.recv.empty()) {
      return 0;
    }

    // Receive each bulk
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      int rc = zmq_recv(socket_, meta.recv[i].data.ptr_, meta.recv[i].size, 0);
      if (rc == -1) {
        return zmq_errno();
      }

      // Check if there are more message parts
      int more = 0;
      size_t more_size = sizeof(more);
      zmq_getsockopt(socket_, ZMQ_RCVMORE, &more, &more_size);

      // If this is the last expected bulk but more parts exist, it's an error
      if (i == meta.recv.size() - 1 && more) {
        return -1;  // More parts than expected
      }

      // If we expect more bulks but no more parts, it's incomplete
      if (i < meta.recv.size() - 1 && !more) {
        return -1;  // Fewer parts than expected
      }
    }

    return 0;  // Success
  }

  std::string GetAddress() const override { return addr_; }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  void* ctx_;
  void* socket_;
};

// --- Base Class Template Implementations ---
// These delegate to the derived class implementations
template<typename MetaT>
int Client::Send(MetaT &meta) {
  // This will be resolved through the vtable to the actual implementation
  return static_cast<ZeroMqClient*>(this)->Send(meta);
}

template<typename MetaT>
int Server::RecvMetadata(MetaT &meta) {
  return static_cast<ZeroMqServer*>(this)->RecvMetadata(meta);
}

template<typename MetaT>
int Server::RecvBulks(MetaT &meta) {
  return static_cast<ZeroMqServer*>(this)->RecvBulks(meta);
}

// --- TransportFactory Implementations ---
inline std::unique_ptr<Client> TransportFactory::GetClient(
    const std::string& addr, Transport t, const std::string& protocol, int port) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqClient>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

inline std::unique_ptr<Client> TransportFactory::GetClient(
    const std::string& addr, Transport t, const std::string& protocol, int port,
    const std::string& domain) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqClient>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

inline std::unique_ptr<Server> TransportFactory::GetServer(
    const std::string& addr, Transport t, const std::string& protocol, int port) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqServer>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

inline std::unique_ptr<Server> TransportFactory::GetServer(
    const std::string& addr, Transport t, const std::string& protocol, int port,
    const std::string& domain) {
  if (t == Transport::kZeroMq) {
    return std::make_unique<ZeroMqServer>(addr, protocol, port);
  }
  throw std::runtime_error("Unsupported transport type");
}

}  // namespace hshm::lbm

#endif  // HSHM_ENABLE_ZMQ