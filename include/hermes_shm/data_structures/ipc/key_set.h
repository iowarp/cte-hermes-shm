//
// Created by llogan on 11/29/24.
//

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_KEY_SET_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_KEY_SET_H_

#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/data_structures/ipc/functional.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include "ring_queue.h"

namespace hshm::ipc {

/**
 * Stores a set of numeric keys and their value. Keys can be reused.
 * Programs must store the keys themselves.
 */
template <typename T, RingQueueFlag RQ_FLAGS, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class key_set_templ : public ShmContainer {
 public:
  RING_QUEUE_DEFS
  hipc::ring_queue_base<size_t, hipc::EmptyHeader, RQ_FLAGS, HSHM_CLASS_TEMPL_ARGS> keys_;
  hipc::vector<T, HSHM_CLASS_TEMPL_ARGS> set_;
  size_t heap_;
  size_t max_size_;

 public:
  key_set_templ() : keys_(), set_() {}

  key_set_templ(const hipc::CtxAllocator<AllocT> &alloc)
      : keys_(alloc), set_(alloc) {}

  void Init(size_t max_size) {
    keys_.resize(max_size);
    set_.reserve(max_size);
    heap_ = 0;
    max_size_ = max_size;
  }

  void emplace(size_t &key, const T &entry) {
    pop_key(key);
    set_[key] = entry;
  }

  void peek(size_t key, T *&entry) { entry = &set_[key]; }

  void pop(size_t key, T &entry) {
    entry = set_[key];
    erase(key);
  }

  void pop(size_t key) { erase(key); }

  void erase(size_t key) { keys_.emplace(key); }

 private:
  void resize() {
    size_t new_size = set_.capacity() * 2;
    keys_.resize(new_size);
    set_.reserve(new_size);
    max_size_ = new_size;
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
    if constexpr (!IsPushAtomic && !IsPopAtomic) {
      resize();
    } else {
      HSHM_THROW_ERROR(KEY_SET_OUT_OF_BOUNDS, "Key set is full");
    }
    key = heap_++;
  }
};

template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using spsc_key_set =
    key_set_templ<T, RING_BUFFER_FIXED_MPMC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using mpmc_key_set =
    key_set_templ<T, RING_BUFFER_FIXED_MPMC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using spsc_key_set =
    ipc::key_set_templ<T, RING_BUFFER_FIXED_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using mpmc_key_set =
    ipc::key_set_templ<T, RING_BUFFER_FIXED_MPMC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_KEY_SET_H_
