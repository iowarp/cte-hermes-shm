//
// Created by lukemartinlogan on 2/3/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Zstd_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Zstd_H_

#include "compress.h"
#include <zstd.h>

namespace hshm {

class Zstd : public Compressor {
 public:
  Zstd() = default;
  
  bool Compress(void *output, size_t &output_size,
                void *input, size_t input_size) override {
    if (ZSTD_compressBound(input_size) > output_size) {
      HILOG(kInfo, "Output buffer is potentially too small for compression");
    }
    output_size = ZSTD_compress(
        output, output_size,
        input, input_size, ZSTD_maxCLevel());
    return output_size != 0;
  }

  bool Decompress(void *output, size_t &output_size,
                  void *input, size_t input_size) override {
    output_size = ZSTD_decompress(
        output, output_size,
        input, input_size);
    return output_size != 0;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Zstd_H_
