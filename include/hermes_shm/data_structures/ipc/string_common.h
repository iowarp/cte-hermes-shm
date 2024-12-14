//
// Created by llogan on 22/10/24.
//

#ifndef STRING_COMMON_H
#define STRING_COMMON_H

#include <cstdint>
#include <cstring>
#include <string>

#include "hermes_shm/constants/macros.h"

namespace hshm {

HSHM_INLINE_CROSS_FUN static size_t strlen(const char *buf) {
  size_t length = 0;
  while (buf[length] != 0) {
    ++length;
  }
  return length;
}

HSHM_INLINE_CROSS_FUN static size_t strnlen(const char *buf, size_t max_len) {
  size_t length = 0;
  for (; length < max_len; ++length) {
    if (buf[length] == 0) {
      break;
    }
  }
  return length;
}

HSHM_INLINE_CROSS_FUN static int strncmp(const char *a, size_t len_a,
                                         const char *b, size_t len_b) {
  if (len_a != len_b) {
    return int((int64_t)len_a - (int64_t)len_b);
  }
  for (size_t i = 0; i < len_a; ++i) {
    if (a[i] != b[i]) {
      return a[i] - b[i];
    }
  }
  return 0;
}

}  // namespace hshm

#endif  // STRING_COMMON_H
