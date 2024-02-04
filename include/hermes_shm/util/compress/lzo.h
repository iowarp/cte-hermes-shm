//
// Created by lukemartinlogan on 2/3/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Lzo_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Lzo_H_

#include "compress.h"
#include <lzo/lzo1x.h>

namespace hshm {

class Lzo : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size,
                void *input, size_t input_size) override {
    int ret = lzo1x_1_15_compress(
        reinterpret_cast<const lzo_bytep>(input), input_size,
        reinterpret_cast<lzo_bytep>(output), &output_size, nullptr);
    return ret != 0;
  }

  bool Decompress(void *output, size_t &output_size,
                  void *input, size_t input_size) override {
    int ret = lzo1x_decompress(
        reinterpret_cast<const lzo_bytep>(input), input_size,
        reinterpret_cast<lzo_bytep>(output), &output_size, nullptr);
    return ret != 0;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Lzo_H_
