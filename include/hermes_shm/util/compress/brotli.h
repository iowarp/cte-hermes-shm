//
// Created by lukemartinlogan on 2/3/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Brotli_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Brotli_H_

#include "compress.h"
#include <brotli/encode.h>
#include <brotli/decode.h>

namespace hshm {

class Brotli : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size,
                void *input, size_t input_size) override {
    BrotliEncoderState* state =
        BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);
    if (state == nullptr) {
      return false;
    }

    const size_t bufferSize = BrotliEncoderMaxCompressedSize(input_size);
    if (bufferSize > output_size) {
      HELOG(kError,
            "Output buffer is probably too small for Brotli compression.")
    }
    int ret = BrotliEncoderCompress(
        BROTLI_PARAM_QUALITY,
        BROTLI_OPERATION_FINISH,
        BROTLI_DEFAULT_MODE,
        input_size, reinterpret_cast<uint8_t*>(input),
        &output_size, reinterpret_cast<uint8_t*>(output));
    BrotliEncoderDestroyInstance(state);
    return ret != 0;
  }

  bool Decompress(void *output, size_t &output_size,
                  void *input, size_t input_size) override {
    BrotliDecoderState* state = BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
    if (state == nullptr) {
      return false;
    }
    int ret = BrotliDecoderDecompress(
        input_size, reinterpret_cast<const uint8_t*>(input),
        &output_size, reinterpret_cast<uint8_t*>(output));
    BrotliDecoderDestroyInstance(state);
    return ret != 0;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Brotli_H_
