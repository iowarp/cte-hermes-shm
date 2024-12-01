//
// Created by llogan on 11/29/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_KEY_SET_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_KEY_SET_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/data_structures/containers/functional.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"

namespace hshm::ipc {

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class KeySetTempl : public ShmContainer {
 public:
  hipc::fixed_spsc_queue<size_t, HSHM_CLASS_TEMPL_ARGS> keys_;
  hipc::vector<T, HSHM_CLASS_TEMPL_ARGS> set_;
  size_t heap_;
  size_t max_size_;

 public:
  KeySetTempl()
  : keys_(HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>()),
    set_(HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>()) {}

  KeySetTempl(const hipc::CtxAllocator<AllocT> &alloc)
  : keys_(alloc), set_(alloc) {}

  void Init(size_t max_size) {
    keys_.resize(max_size);
    set_.resize(max_size);
    heap_ = 0;
    max_size_ = max_size;
  }

  void resize() {
    size_t new_size = set_.size() * 2;
    keys_.resize(new_size);
    set_.resize(new_size);
    max_size_ = new_size;
  }

  void emplace(const T &entry, size_t &key) {
    pop_key(key);
    set_[key] = entry;
  }

  void peek(size_t key, T *&entry) {
    entry = &set_[key];
  }

  void pop(size_t key, T &entry) {
    entry = set_[key];
    erase(key);
  }

  void pop_key(size_t &key) {
    // We have a key cached
    if (!keys_.pop(key).IsNull()) {
      return;
    }
    // We have keys in the heap
    if (heap_ < max_size_) {
      key = heap_++;
      return;
    }
    // We need more keys
    resize();
    key = heap_++;
  }

  void erase(size_t key) {
    keys_.emplace(key);
  }
};

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using KeySet = KeySetTempl<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm::ipc

namespace hshm {

template<typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using KeySet = ipc::KeySetTempl<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_KEY_SET_H_
