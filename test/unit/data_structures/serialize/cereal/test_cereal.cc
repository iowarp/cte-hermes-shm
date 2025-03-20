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

#include <cereal/types/atomic.hpp>

#include "basic_test.h"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "hermes_shm/data_structures/all.h"
#include "hermes_shm/data_structures/ipc/string.h"
#include "test_init.h"

TEST_CASE("SerializePod") {
  std::stringstream ss;
  {
    int x = 225;
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    int x;
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x == 225);
  }
}

TEST_CASE("SerializeVector") {
  std::stringstream ss;
  {
    std::vector<int> x{1, 2, 3, 4, 5};
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    std::vector<int> x;
    std::vector<int> y{1, 2, 3, 4, 5};
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x == y);
  }
}

TEST_CASE("SerializeHipcVec0") {
  std::stringstream ss;
  {
    auto x = hipc::vector<int>();
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    hipc::vector<int> x;
    std::vector<int> y;
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x.vec() == y);
  }
}

TEST_CASE("SerializeHipcVec") {
  std::stringstream ss;
  {
    auto x = hipc::vector<int>();
    x.reserve(5);
    for (int i = 0; i < 5; ++i) {
      x.emplace_back(i);
    }
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    hipc::vector<int> x;
    std::vector<int> y{0, 1, 2, 3, 4};
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x.vec() == y);
  }
}

TEST_CASE("SerializeHipcVecString") {
  std::stringstream ss;
  {
    auto x = hipc::vector<std::string>();
    x.reserve(5);
    for (int i = 0; i < 5; ++i) {
      x.emplace_back(std::to_string(i));
    }
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    hipc::vector<std::string> x;
    std::vector<std::string> y{"0", "1", "2", "3", "4"};
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x.vec() == y);
  }
}

TEST_CASE("SerializeHipcShmArchive") {
  std::stringstream ss;
  {
    hipc::ShmArchive<hipc::vector<int>> x;
    HSHM_MAKE_AR0(x, HSHM_DEFAULT_ALLOC);
    x->reserve(5);
    for (int i = 0; i < 5; ++i) {
      x->emplace_back(i);
    }
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    hipc::ShmArchive<hipc::vector<int>> x;
    HSHM_MAKE_AR0(x, HSHM_DEFAULT_ALLOC);
    std::vector<int> y{0, 1, 2, 3, 4};
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x->vec() == y);
  }
}

TEST_CASE("SerializeAtomic") {
  std::stringstream ss;
  {
    std::atomic<int> x(225);
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    std::atomic<int> x(225);
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x == 225);
  }
}
TEST_CASE("SerializeBitfield") {
  std::stringstream ss;
  {
    hshm::ibitfield x;
    x.SetBits(0x8);
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    hshm::ibitfield x;
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x.bits_ == 0x8);
  }
}
