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
#include "hermes_shm/data_structures/string.h"

using hermes_shm::ipc::string;

void TestString() {
  Allocator *alloc = alloc_g;

  PAGE_DIVIDE("Test basic construction from string") {
    hipc::string text(alloc, "hello1");
    REQUIRE(text == "hello1");
    REQUIRE(text != "h");
    REQUIRE(text != "asdfklaf");
    return;
  }

  PAGE_DIVIDE("Test the mutability of the string") {
    hipc::string text(alloc, 6);
    memcpy(text.data_mutable(), "hello4", strlen("hello4"));
    REQUIRE(text == "hello4");
  }

  PAGE_DIVIDE("Test copy assign") {
    hipc::string text1(alloc, "hello");
    hipc::string text2(alloc);
    (text2) = (text1);
    REQUIRE(text1 == "hello");
    REQUIRE(text1.header_ != text2.header_);
  }

  PAGE_DIVIDE("Test move assign") {
    hipc::string text1(alloc, "hello");
    hipc::string text2(alloc);
    (text2) = std::move(text1);
    REQUIRE(text2 == "hello");
    REQUIRE(text1.header_ != text2.header_);
  }
}

TEST_CASE("String") {
  Allocator *alloc = alloc_g;
  REQUIRE(IS_SHM_ARCHIVEABLE(string));
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  TestString();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
