#pragma once
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_cm.h>
#include <cstring>
#include <queue>
#include <mutex>
#include <memory>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include "lightbeam.h"
#include "hermes_shm/util/logging.h"

namespace hshm::lbm {

inline std::string AddrToHex(const void* addr, size_t len) {
  std::ostringstream oss;
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(addr);
  for (size_t i = 0; i < len; ++i) {
    oss << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i];
  }
  return oss.str();
}

inline std::vector<uint8_t> HexToAddr(const std::string& hex) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i < hex.size(); i += 2) {
    std::string byte_string = hex.substr(i, 2);
    uint8_t byte = (uint8_t)strtol(byte_string.c_str(), nullptr, 16);
    out.push_back(byte);
  }
  return out;
}

class LibfabricClient : public Client {
 public:
  LibfabricClient(const std::string& server_addr_hex,
                  const std::string& protocol = "tcp", int port = 9222)
      : protocol_(protocol),
        port_(port),
        fabric_(nullptr),
        domain_(nullptr),
        ep_(nullptr),
        av_(nullptr),
        cq_(nullptr),
        supports_rdma_(false) {
    struct fi_info* hints = fi_allocinfo();
    hints->ep_attr->type = FI_EP_RDM;
    hints->caps = FI_MSG | FI_RMA;
    hints->mode = 0;
    hints->addr_format = FI_SOCKADDR_IN;
    hints->fabric_attr->prov_name = strdup("sockets");
    struct fi_info* info = nullptr;
    int ret = fi_getinfo(FI_VERSION(1, 11), nullptr, nullptr, 0, hints, &info);
    if (ret) throw std::runtime_error("fi_getinfo failed: " + std::to_string(ret));
    supports_rdma_ = (info->caps & FI_RMA) != 0;
    ret = fi_fabric(info->fabric_attr, &fabric_, nullptr);
    if (ret) throw std::runtime_error("fi_fabric failed");
    ret = fi_domain(fabric_, info, &domain_, nullptr);
    if (ret) throw std::runtime_error("fi_domain failed");
    ret = fi_endpoint(domain_, info, &ep_, nullptr);
    if (ret) throw std::runtime_error("fi_endpoint failed");
    struct fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.size = 10;
    ret = fi_cq_open(domain_, &cq_attr, &cq_, nullptr);
    if (ret) throw std::runtime_error("fi_cq_open failed");
    struct fi_av_attr av_attr = {};
    av_attr.type = (fi_av_type)FI_AV_MAP;
    ret = fi_av_open(domain_, &av_attr, &av_, nullptr);
    if (ret) throw std::runtime_error("fi_av_open failed");
    ret = fi_ep_bind(ep_, &cq_->fid, FI_SEND | FI_RECV);
    if (ret) throw std::runtime_error("fi_ep_bind CQ failed");
    ret = fi_ep_bind(ep_, &av_->fid, 0);
    if (ret) throw std::runtime_error("fi_ep_bind AV failed");
    ret = fi_enable(ep_);
    if (ret) throw std::runtime_error("fi_enable failed");
    // Insert server address into AV
    std::vector<uint8_t> server_addr = HexToAddr(server_addr_hex);
    fi_addr_t server_fi_addr;
    ret = fi_av_insert(av_, server_addr.data(), 1, &server_fi_addr, 0, nullptr);
    if (ret != 1) throw std::runtime_error("fi_av_insert failed: " + std::to_string(ret));
    server_fi_addr_ = server_fi_addr;
    fi_freeinfo(info);
    fi_freeinfo(hints);
  }

  ~LibfabricClient() override {
    if (ep_) fi_close(&ep_->fid);
    if (av_) fi_close(&av_->fid);
    if (cq_) fi_close(&cq_->fid);
    if (domain_) fi_close(&domain_->fid);
    if (fabric_) fi_close(&fabric_->fid);
  }

  Bulk Expose(const char* data, size_t data_size, int flags) override {
    Bulk bulk;
    bulk.data = const_cast<char*>(data);
    bulk.size = data_size;
    bulk.flags = flags;
    bulk.desc = nullptr;
    bulk.mr = nullptr;
    if (supports_rdma_) {
      struct fid_mr* mr = nullptr;
      int ret = fi_mr_reg(domain_, (void*)data, data_size,
                          FI_SEND | FI_RECV | FI_READ | FI_WRITE, 0, 0, 0, &mr,
                          nullptr);
      if (ret) throw std::runtime_error("fi_mr_reg failed: " + std::to_string(ret));
      bulk.desc = fi_mr_desc(mr);
      bulk.mr = mr;
    }
    return bulk;
  }

  Event* Send(const Bulk& bulk) override {
    if (!ep_ || !cq_ || !av_) {
      throw std::runtime_error("Null resource in fi_send");
    }
    Event* event = new Event();
    ssize_t ret = fi_send(ep_, bulk.data, bulk.size, bulk.desc, server_fi_addr_,
                          nullptr);
    if (ret < 0) {
      event->is_done = true;
      event->error_code = ret;
      event->error_message = fi_strerror(-ret);
      return event;
    }
    struct fi_cq_entry entry;
    while (true) {
      ret = fi_cq_read(cq_, &entry, 1);
      if (ret == 1) {
        event->is_done = true;
        event->bytes_transferred = bulk.size;
        break;
      } else if (ret == -FI_EAGAIN) {
        continue;
      } else {
        event->is_done = true;
        event->error_code = ret;
        event->error_message = fi_strerror(-ret);
        break;
      }
    }
    // Cleanup MR if used
    if (bulk.mr) fi_close((fid_t)bulk.mr);
    return event;
  }

 private:
  std::string protocol_;
  int port_;
  struct fid_fabric* fabric_;
  struct fid_domain* domain_;
  struct fid_ep* ep_;
  struct fid_av* av_;
  struct fid_cq* cq_;
  fi_addr_t server_fi_addr_;
  bool supports_rdma_;
};

class LibfabricServer : public Server {
 public:
  LibfabricServer(const std::string& addr, const std::string& protocol = "tcp",
                  int port = 9222)
      : addr_(addr),
        protocol_(protocol),
        port_(port),
        fabric_(nullptr),
        domain_(nullptr),
        ep_(nullptr),
        av_(nullptr),
        cq_(nullptr),
        supports_rdma_(false) {
    struct fi_info* hints = fi_allocinfo();
    hints->ep_attr->type = FI_EP_RDM;
    hints->caps = FI_MSG | FI_RMA;
    hints->mode = 0;
    hints->addr_format = FI_SOCKADDR_IN;
    hints->fabric_attr->prov_name = strdup("sockets");
    struct fi_info* info = nullptr;
    int ret = fi_getinfo(FI_VERSION(1, 11), addr.c_str(),
                         std::to_string(port).c_str(), FI_SOURCE, hints, &info);
    if (ret) throw std::runtime_error("fi_getinfo failed: " + std::to_string(ret));
    supports_rdma_ = (info->caps & FI_RMA) != 0;
    std::cout << "[LibfabricServer] supports_rdma_=" << supports_rdma_
              << std::endl;
    ret = fi_fabric(info->fabric_attr, &fabric_, nullptr);
    if (ret) throw std::runtime_error("fi_fabric failed");
    ret = fi_domain(fabric_, info, &domain_, nullptr);
    if (ret) throw std::runtime_error("fi_domain failed");
    ret = fi_endpoint(domain_, info, &ep_, nullptr);
    if (ret) throw std::runtime_error("fi_endpoint failed");
    struct fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.size = 10;
    ret = fi_cq_open(domain_, &cq_attr, &cq_, nullptr);
    if (ret) throw std::runtime_error("fi_cq_open failed");
    struct fi_av_attr av_attr = {};
    av_attr.type = (fi_av_type)FI_AV_MAP;
    ret = fi_av_open(domain_, &av_attr, &av_, nullptr);
    if (ret) throw std::runtime_error("fi_av_open failed");
    ret = fi_ep_bind(ep_, &cq_->fid, FI_SEND | FI_RECV);
    if (ret) throw std::runtime_error("fi_ep_bind CQ failed");
    ret = fi_ep_bind(ep_, &av_->fid, 0);
    if (ret) throw std::runtime_error("fi_ep_bind AV failed");
    ret = fi_enable(ep_);
    if (ret) throw std::runtime_error("fi_enable failed");
    // Get own address for client to use
    char addr_buf[128] = {0};
    size_t addrlen = sizeof(addr_buf);
    ret = fi_getname(&ep_->fid, addr_buf, &addrlen);
    if (ret) throw std::runtime_error("fi_getname failed: " + std::to_string(ret));
    addr_hex_ = AddrToHex(addr_buf, addrlen);
    fi_freeinfo(info);
    fi_freeinfo(hints);
  }

  ~LibfabricServer() override {
    if (ep_) fi_close(&ep_->fid);
    if (av_) fi_close(&av_->fid);
    if (cq_) fi_close(&cq_->fid);
    if (domain_) fi_close(&domain_->fid);
    if (fabric_) fi_close(&fabric_->fid);
  }

  Bulk Expose(char* data, size_t data_size, int flags) override {
    Bulk bulk;
    bulk.data = data;
    bulk.size = data_size;
    bulk.flags = flags;
    bulk.desc = nullptr;
    bulk.mr = nullptr;
    if (supports_rdma_) {
      struct fid_mr* mr = nullptr;
      int ret = fi_mr_reg(domain_, data, data_size,
                          FI_SEND | FI_RECV | FI_READ | FI_WRITE, 0, 0, 0, &mr,
                          nullptr);
      if (ret) throw std::runtime_error("fi_mr_reg failed: " + std::to_string(ret));
      bulk.desc = fi_mr_desc(mr);
      bulk.mr = mr;
    }
    return bulk;
  }

  Event* Recv(const Bulk& bulk) override {
    if (!ep_ || !cq_ || !av_) {
      throw std::runtime_error("Null resource in fi_recv");
    }
    Event* event = new Event();
    ssize_t ret = fi_recv(ep_, bulk.data, bulk.size, bulk.desc, FI_ADDR_UNSPEC,
                          nullptr);
    if (ret < 0) {
      event->is_done = true;
      event->error_code = ret;
      event->error_message = fi_strerror(-ret);
      return event;
    }
    struct fi_cq_entry entry;
    while (true) {
      ret = fi_cq_read(cq_, &entry, 1);
      if (ret == 1) {
        event->is_done = true;
        event->bytes_transferred = bulk.size;
        break;
      } else if (ret == -FI_EAGAIN) {
        continue;
      } else {
        event->is_done = true;
        event->error_code = ret;
        event->error_message = fi_strerror(-ret);
        break;
      }
    }
    // Cleanup MR if used
    if (bulk.mr) fi_close((fid_t)bulk.mr);
    return event;
  }

  std::string GetAddress() const override { return addr_hex_; }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  std::string addr_hex_;
  struct fid_fabric* fabric_;
  struct fid_domain* domain_;
  struct fid_ep* ep_;
  struct fid_av* av_;
  struct fid_cq* cq_;
  bool supports_rdma_;
};

}  // namespace hshm::lbm 