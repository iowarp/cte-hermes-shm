//
// Created by lukemartinlogan on 2/3/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Lzma_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Lzma_H_

#include "compress.h"
#include <lzma.h>

namespace hshm {

class Lzma : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size,
                void *input, size_t input_size) override {
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret;

    // Initialize the LZMA encoder with preset LZMA_PRESET_DEFAULT (equivalent to -6 compression level)
    ret = lzma_easy_encoder(&strm, LZMA_PRESET_DEFAULT, LZMA_CHECK_CRC64);
    if (ret != LZMA_OK) {
      HELOG(kError, "Error initializing LZMA compression.")
      return false;
    }

    // Set input buffer and size
    strm.next_in = reinterpret_cast<const uint8_t*>(input);
    strm.avail_in = input_size;

    // Set output buffer and size
    strm.next_out = reinterpret_cast<uint8_t*>(output);
    strm.avail_out = output_size;

    // Compress the data
    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
      HELOG(kError, "Error compressing data with LZMA.")
      lzma_end(&strm);
      return false;
    }

    output_size -= strm.avail_out;

    // Finish compression
    lzma_end(&strm);
    return true;
  }

  bool Decompress(void *output, size_t &output_size,
                  void *input, size_t input_size) override {
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret;

    // Initialize the LZMA decoder
    ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);
    if (ret != LZMA_OK) {
      HELOG(kInfo, "Error initializing LZMA decompression.")
      return false;
    }

    // Set input buffer and size
    strm.next_in = reinterpret_cast<const uint8_t*>(input);
    strm.avail_in = input_size;

    // Set output buffer and size
    strm.next_out = reinterpret_cast<uint8_t*>(output);
    strm.avail_out = output_size;

    // Decompress the data
    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
      HELOG(kError, "Error decompressing data with LZMA.")
      lzma_end(&strm);
      return false;
    }

    output_size -= strm.avail_out;

    // Finish decompression
    lzma_end(&strm);
    return true;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Lzma_H_
