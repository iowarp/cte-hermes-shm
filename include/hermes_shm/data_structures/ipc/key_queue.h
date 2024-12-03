
#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_KEY_QUEUE_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_KEY_QUEUE_H_

#include "key_set.h"

namespace hshm::ipc {

/** 
 * A list-like queue, but with a preallocated maximum depth.
 * Uses the key set to store and track free list.
 * Assumes the type T has next_ + prior_ variables.
 */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class key_queue : public hipc::ShmContainer {
 public:
  key_set<T, HSHM_CLASS_TEMPL_ARGS> queue_;
  size_t size_, head_, tail_;
  int id_;

 public:
  void Init(int id, size_t queue_depth) {
    queue_.Init(queue_depth);
    size_ = 0;
    tail_ = 0;
    head_ = 0;
    id_ = id;
  }

  HSHM_ALWAYS_INLINE
  bool push(const T &entry) {
    size_t key;
    queue_.emplace(key, entry);
    if (size_ == 0) {
      head_ = key;
      tail_ = key;
    } else {
      T *point;
      // Tail is entry's prior
      queue_.peek(key, point);
      point->prior_ = tail_;
      // Prior's next is entry
      queue_.peek(tail_, point);
      point->next_ = key;
      // Update tail
      tail_ = key;
    }
    ++size_;
    return true;
  }

  HSHM_ALWAYS_INLINE
  void peek(T *&entry, size_t off) { queue_.peek(off, entry); }

  HSHM_ALWAYS_INLINE
  void peek(T *&entry) { queue_.peek(head_, entry); }

  HSHM_ALWAYS_INLINE
  void pop(T &entry) {
    T *point;
    peek(point);
    entry = *point;
    erase();
  }

  HSHM_ALWAYS_INLINE
  void erase() {
    T *point;
    queue_.peek(head_, point);
    size_t head = point->next_;
    queue_.erase(head_);
    head_ = head;
    --size_;
  }

  size_t size() { return size_; }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using key_queue = ipc::key_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_KEY_QUEUE_H_