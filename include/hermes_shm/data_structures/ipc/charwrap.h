#pragma once
#include "hermes_shm/types/numbers.h"

namespace hshm::ipc {

class CharWrap {
 public:
  char *data_;
  size_t size_;

 public:
  HSHM_INLINE_CROSS_FUN
  CharWrap() = default;

  HSHM_INLINE_CROSS_FUN
  ~CharWrap() = default;

  HSHM_INLINE_CROSS_FUN
  CharWrap(const char *data, size_t size)
      : data_(const_cast<char *>(data)), size_(size) {}

  HSHM_INLINE_CROSS_FUN
  CharWrap(char *data, size_t size) : data_(data), size_(size) {}

  CharWrap(const std::string &str)
      : data_(const_cast<char *>(str.data())), size_(str.size()) {}

  HSHM_INLINE_CROSS_FUN
  char *data() { return data_; }

  HSHM_INLINE_CROSS_FUN
  const char *data() const { return data_; }

  HSHM_INLINE_CROSS_FUN
  const char *c_str() const { return data_; }

  HSHM_INLINE_CROSS_FUN
  size_t size() const { return size_; }
};

}  // namespace hshm::ipc