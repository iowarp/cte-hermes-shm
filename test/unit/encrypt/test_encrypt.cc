/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
* Distributed under BSD 3-Clause license.                                   *
* Copyright by The HDF Group.                                               *
* Copyright by the Illinois Institute of Technology.                        *
* All rights reserved.                                                      *
*                                                                           *
* This file is part of Hermes. The full Hermes copyright notice, including  *
* terms governing use, modification, and redistribution, is contained in    *
* the COPYING file, which can be found at the top directory. If you do not  *
* have access to the file, you may request a copy from help@hdfgroup.org.   *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "basic_test.h"
#include <utility>
#include <hermes_shm/util/encrypt/encrypt.h>

TEST_CASE("TestEncrypt") {
  PAGE_DIVIDE("AES") {
    AES aes;
    aes.GenerateKey("passwd");
    size_t encoded_size = 8192 + 256, decoded_size = 8192;
    std::vector<char> data(8192, 0), encoded(8192 + 256, 1), decoded(8192 + 256, 2);
    aes.CreateInitialVector();
    aes.Encrypt(encoded.data(), encoded_size, data.data(), data.size());
    aes.Decrypt(decoded.data(), decoded_size, encoded.data(), encoded_size);
    decoded.resize(decoded_size);
    REQUIRE(data == decoded);
  }
}
