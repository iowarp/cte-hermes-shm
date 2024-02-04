//
// Created by lukemartinlogan on 2/3/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Lzo_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Lzo_H_

#include "compress.h"

namespace hshm {

class Zlib : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size,
                void *input, size_t input_size) override {
  }

  bool Decompress(void *output, size_t &output_size,
                  void *input, size_t input_size) override {
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Lzo_H_
