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
#include "test_init.h"
#include "hermes_shm/data_structures/ipc/string.h"
#include "hermes_shm/data_structures/all.h"
#include "cereal/types/vector.hpp"
#include "cereal/types/string.hpp"
#include <cereal/types/atomic.hpp>

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
    auto x = hipc::make_uptr<hipc::vector<int>>();
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    hipc::uptr<hipc::vector<int>> x;
    std::vector<int> y;
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x->vec() == y);
  }
}

TEST_CASE("SerializeHipcVec") {
  std::stringstream ss;
  {
    auto x = hipc::make_uptr<hipc::vector<int>>();
    x->reserve(5);
    for (int i = 0; i < 5; ++i) {
      x->emplace_back(i);
    }
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    hipc::uptr<hipc::vector<int>> x;
    std::vector<int> y{0, 1, 2, 3, 4};
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x->vec() == y);
  }
}

TEST_CASE("SerializeHipcVecString") {
  std::stringstream ss;
  {
    auto x = hipc::make_uptr<hipc::vector<std::string>>();
    x->reserve(5);
    for (int i = 0; i < 5; ++i) {
      x->emplace_back(std::to_string(i));
    }
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    hipc::uptr<hipc::vector<std::string>> x;
    std::vector<std::string> y{"0", "1", "2", "3", "4"};
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x->vec() == y);
  }
}

TEST_CASE("SerializeHipcShmArchive") {
  std::stringstream ss;
  {
    hipc::ShmArchive<hipc::vector<int>> x;
    HSHM_MAKE_AR0(x, HERMES_MEMORY_MANAGER->GetDefaultAllocator());
    x->reserve(5);
    for (int i = 0; i < 5; ++i) {
      x->emplace_back(i);
    }
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
  }
  {
    hipc::ShmArchive<hipc::vector<int>> x;
    HSHM_MAKE_AR0(x, HERMES_MEMORY_MANAGER->GetDefaultAllocator());
    std::vector<int> y{0, 1, 2, 3, 4};
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x->vec() == y);
  }
}


TEST_CASE("SerializePodArray") {
  std::stringstream ss;
  hipc::CtxAllocator<Allocator> alloc(HERMES_MEMORY_MANAGER->GetDefaultAllocator());
  {
    hipc::pod_array<int, 2> x;
    x.construct(alloc, 5);
    for (int i = 0; i < 5; ++i) {
      x[i] = i;
    }
    cereal::BinaryOutputArchive ar(ss);
    ar << x;
    x.destroy();
  }
  {
    hipc::pod_array<int, 2> x;
    std::vector<int> y{0, 1, 2, 3, 4};
    cereal::BinaryInputArchive ar(ss);
    ar >> x;
    REQUIRE(x.size_ == (int)y.size());
    for (int i = 0; i < x.size_; ++i) {
      REQUIRE(x[i] == y[i]);
    }
    x.destroy();
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
