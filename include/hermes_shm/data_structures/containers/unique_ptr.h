//
// Created by llogan on 10/11/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_UNIQUE_PTR_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_UNIQUE_PTR_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/memory/memory_manager_.h"

namespace hshm {

namespace hipc = hshm::ipc;

template <typename T>
class unique_ptr {
 public:
  HSHM_CROSS_FUN unique_ptr() : ptr_(nullptr) {}

  HSHM_CROSS_FUN explicit unique_ptr(T* ptr) : ptr_(ptr) {}

  HSHM_CROSS_FUN ~unique_ptr() {
    hipc::Allocator *alloc = ptr_->GetAllocator();
    alloc->DelObj(ptr_);
  }

  HSHM_CROSS_FUN T* get() const {
    return ptr_;
  }

  HSHM_CROSS_FUN T* release() {
    T* tmp = ptr_;
    ptr_ = nullptr;
    return tmp;
  }

  HSHM_CROSS_FUN void reset(T* ptr) {
    hipc::Allocator *alloc = ptr_->GetAllocator();
    alloc->DelObj(ptr_);
    ptr_ = ptr;
  }

  HSHM_CROSS_FUN T& operator*() const {
    return *ptr_;
  }

  HSHM_CROSS_FUN T* operator->() const {
    return ptr_;
  }

 private:
  T* ptr_;
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_UNIQUE_PTR_H_
