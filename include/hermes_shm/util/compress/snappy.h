//
// Created by lukemartinlogan on 2/3/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Snappy_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Snappy_H_

#include "compress.h"
#include <snappy.h>
#include <snappy-sinksource.h>

namespace hshm {

class Snappy : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size,
                void *input, size_t input_size) override {
    snappy::RawCompress(
        (char*)input,
        input_size,
        (char*)output,
        &output_size);
    bool ret = snappy::IsValidCompressedBuffer((char*)output, output_size);
    return ret;
  }

  bool Decompress(void *output, size_t &output_size,
                  void *input, size_t input_size) override {
    snappy::ByteArraySource source(
        reinterpret_cast<const char*>(input), input_size);
    snappy::UncheckedByteArraySink sink(
        reinterpret_cast<char*>(output));
    output_size = snappy::UncompressAsMuchAsPossible(&source, &sink);
    return output_size != 0;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Snappy_H_
