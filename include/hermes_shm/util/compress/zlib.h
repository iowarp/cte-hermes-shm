//
// Created by lukemartinlogan on 2/3/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Zlib_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Zlib_H_

#include "compress.h"
#include <zlib.h>

namespace hshm {

class Zlib : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size,
                void *input, size_t input_size) override {
    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    if (deflateInit(&stream, Z_DEFAULT_COMPRESSION) != Z_OK) {
      HELOG(kError, "Error initializing zlib compression.");
      return false;
    }

    stream.avail_in = input_size;
    stream.next_in = reinterpret_cast<Bytef*>(reinterpret_cast<char*>(input));

    stream.avail_out = output_size;
    stream.next_out = reinterpret_cast<Bytef*>(output);

    if (deflate(&stream, Z_FINISH) != Z_STREAM_END) {
      std::cerr << "Error compressing data with zlib." << std::endl;
      deflateEnd(&stream);
      return false;
    }

    output_size = stream.total_out;
    deflateEnd(&stream);
    return true;
  }

  bool Decompress(void *output, size_t &output_size,
                  void *input, size_t input_size) override {
    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    if (inflateInit(&stream) != Z_OK) {
      HELOG(kError, "Error initializing zlib decompression.");
      return false;
    }

    stream.avail_in = input_size;
    stream.next_in = reinterpret_cast<Bytef*>(input);

    stream.avail_out = output_size;
    stream.next_out = reinterpret_cast<Bytef*>(output);

    if (inflate(&stream, Z_FINISH) != Z_STREAM_END) {
      HELOG(kError, "Error decompressing data with zlib.")
      inflateEnd(&stream);
      return false;
    }

    output_size = stream.total_out;
    inflateEnd(&stream);
    return true;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_COMPRESS_Zlib_H_
