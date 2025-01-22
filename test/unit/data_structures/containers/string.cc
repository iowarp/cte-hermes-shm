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

using hshm::ipc::string;

void TestString() {
  auto *alloc = HSHM_DEFAULT_ALLOC;

  PAGE_DIVIDE("Test construction from const char*") {
    hipc::string text(alloc, "hello1");
    REQUIRE(text == "hello1");
    REQUIRE(text != "h");
    REQUIRE(text != "asdfklaf");
  }

  PAGE_DIVIDE("Test assignment operator") {
    hipc::string text = hipc::string("hello1");
    REQUIRE(text == "hello1");
    REQUIRE(text != "h");
    REQUIRE(text != "asdfklaf");
  }

  PAGE_DIVIDE("Test construction from std::string") {
    hipc::string text(alloc, std::string("hello1"));
    REQUIRE(text == "hello1");
    REQUIRE(text != "h");
    REQUIRE(text != "asdfklaf");
  }

  PAGE_DIVIDE("Test the mutability of the string") {
    hipc::string text(alloc, 6);
    memcpy(text.data(), "hello4", strlen("hello4"));
    REQUIRE(text == "hello4");
  }

  PAGE_DIVIDE("Test copy assign from hipc::string") {
    hipc::string text1(alloc, "hello");
    hipc::string text2(alloc);
    text2 = text1;
    REQUIRE(text1 == "hello");
  }

  PAGE_DIVIDE("Test copy assign from std::string") {
    hipc::string text1(alloc, "hello");
    text1 = std::string("hello2");
    REQUIRE(text1 == "hello2");
  }

  PAGE_DIVIDE("Test move assign from hipc::string") {
    hipc::string text1(alloc, "hello");
    hipc::string text2(alloc);
    text2 = std::move(text1);
    REQUIRE(text2 == "hello");
  }

  PAGE_DIVIDE("Move from a string. Re-assign moved string.") {
    hipc::string text1(alloc, "hello");
    hipc::string text2(alloc);
    text2 = std::move(text1);
    text1 = "hello2";
    REQUIRE(text2 == "hello");
    REQUIRE(text1 == "hello2");
  }
}

TEST_CASE("StringConv") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(IS_SHM_ARCHIVEABLE(string));
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  {
    hipc::string text("hello");
    hshm::string x(text);
    REQUIRE(text == x);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("String") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(IS_SHM_ARCHIVEABLE(string));
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  TestString();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
