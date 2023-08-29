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
#include "hermes_shm/data_structures/ipc/pod_array.h"
#include "vector.h"

TEST_CASE("PodArray") {
  Allocator *alloc = HERMES_MEMORY_MANAGER->GetDefaultAllocator();

  PAGE_DIVIDE("resize") {
    hipc::pod_array<int, 2> vec;
    vec.resize(alloc, 3);
    REQUIRE(vec.size_ == 3);
    REQUIRE(vec.get() != vec.cache_);
    vec[0] = 25;
    vec[1] = 26;
    REQUIRE(vec[0] == 25);
    REQUIRE(vec[1] == 26);
  }

  PAGE_DIVIDE("Get") {
    hipc::pod_array<int, 2> vec;
    vec.resize(alloc, 1);
    REQUIRE(vec.get() == vec.cache_);
    vec[0] = 25;
    REQUIRE(vec[0] == 25);
  }
}
