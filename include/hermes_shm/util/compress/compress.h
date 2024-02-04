//
// Created by lukemartinlogan on 2/3/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_COMPRESS_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_COMPRESS_H_

#include "hermes_shm/data_structures/data_structure.h"

namespace hshm {

class Compressor {
 public:
  Compressor() = default;
  virtual ~Compressor() = default;

  /**
   * Compress the input buffer into the output buffer
   * */
  virtual bool Compress(void *output, size_t &output_size,
                        void *input, size_t input_size) = 0;

  /**
   * Decompress the input buffer into the output buffer.
   * */
  virtual bool Decompress(void *output, size_t &output_size,
                          void *input, size_t input_size) = 0;
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_COMPRESS_H_
