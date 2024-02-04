//
// Created by lukemartinlogan on 2/3/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_BZIP2_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_BZIP2_H_

#include "compress.h"
#include <bzlib.h>

namespace hshm {

class Bzip2 : public Compressor {
 public:
  int level_;
  int verbosity_ = 0;
  int work_factor_ = 30;

 public:
  Bzip2() : level_(9) {}
  Bzip2(int level) : level_(level) {}

  bool Compress(void *output, size_t &output_size,
                void *input, size_t input_size) override {
    unsigned int output_size_int = output_size;
    int ret = BZ2_bzBuffToBuffCompress(
        (char*)output, &output_size_int,
        (char*)input, input_size,
        level_, verbosity_, work_factor_);
    output_size = output_size_int;
    return ret == BZ_OK;
  }

  bool Decompress(void *output, size_t &output_size,
                  void *input, size_t input_size) override {
    unsigned int output_size_int = output_size;
    int small = 0;
    int ret = BZ2_bzBuffToBuffDecompress(
        (char*)output, &output_size_int,
        (char*)input, input_size,
        small, verbosity_);
    output_size = output_size_int;
    return ret == BZ_OK;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_BZIP2_H_
