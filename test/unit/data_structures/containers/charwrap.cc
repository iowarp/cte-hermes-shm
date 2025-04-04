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

void TestCharbuf() {
  auto *alloc = HSHM_DEFAULT_ALLOC;

  PAGE_DIVIDE("Construct from allocator") {
    hshm::charwrap data(256);
    memset(data.data(), 0, 256);
    REQUIRE(data.size() == 256);
    REQUIRE(data.GetAllocator() == alloc);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Construct from malloc") {
    char *ptr = (char*)malloc(256);
    hshm::charwrap data(ptr, 256);
    memset(data.data(), 0, 256);
    REQUIRE(data.size() == 256);
    free(ptr);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Resize null charbuf to higher value") {
    hshm::charwrap data;
    data.resize(256);
    REQUIRE(data.size() == 256);
    REQUIRE(data.GetAllocator() == alloc);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Resize null charbuf to 0 value") {
    hshm::charwrap data;
    data.resize(0);
    REQUIRE(data.size() == 0);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Resize destructable charbuf to 0 value") {
    hshm::charwrap data(8192);
    data.resize(0);
    REQUIRE(data.size() == 0);
    REQUIRE(data.GetAllocator() == alloc);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Resize destructable charbuf to lower value") {
    hshm::charwrap data(8192);
    data.resize(256);
    REQUIRE(data.size() == 256);
    REQUIRE(data.GetAllocator() == alloc);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Resize destructable charbuf to higher value") {
    hshm::charwrap data(256);
    data.resize(8192);
    REQUIRE(data.size() == 8192);
    REQUIRE(data.GetAllocator() == alloc);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Resize indestructable charbuf to higher value") {
    char *ptr = (char*)malloc(256);
    hshm::charwrap data(ptr, 256);
    data.resize(8192);
    REQUIRE(data.size() == 8192);
    free(ptr);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Resize indestructable charbuf to lower value") {
    char *ptr = (char*)malloc(8192);
    hshm::charwrap data(ptr, 8192);
    data.resize(256);
    REQUIRE(data.size() == 256);
    free(ptr);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Move construct from destructable") {
    hshm::charwrap data1(8192);
    hshm::charwrap data2(std::move(data1));
    REQUIRE(data2.size() == 8192);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Move construct from indestructable") {
    char *ptr1 = (char*)malloc(8192);
    hshm::charwrap data1(ptr1, 8192);
    hshm::charwrap data2(std::move(data1));
    REQUIRE(data2.size() == 8192);
    free(ptr1);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Move assign between two destructables") {
    hshm::charwrap data1(8192);
    hshm::charwrap data2(512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Move assign between two indestructables") {
    char *ptr1 = (char*)malloc(8192);
    hshm::charwrap data1(ptr1, 8192);
    char *ptr2 = (char*)malloc(512);
    hshm::charwrap data2(ptr2, 512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
    free(ptr1);
    free(ptr2);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Move assign indestructable -> destructable") {
    hshm::charwrap data1(8192);
    char *ptr2 = (char*)malloc(512);
    hshm::charwrap data2(ptr2, 512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
    free(ptr2);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Move assign destructable -> indestructable") {
    char *ptr1 = (char*)malloc(8192);
    hshm::charwrap data1(ptr1, 8192);
    hshm::charwrap data2(512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
    free(ptr1);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Move assign to null") {
    hshm::charwrap data1;
    hshm::charwrap data2(512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Copy construct from destructable") {
    hshm::charwrap data1(8192);
    hshm::charwrap data2(data1);
    REQUIRE(data1.size() == 8192);
    REQUIRE(data2.size() == 8192);
    REQUIRE(data1 == data2);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Copy construct from indestructable") {
    char *ptr1 = (char*)malloc(8192);
    hshm::charwrap data1(ptr1, 8192);
    hshm::charwrap data2(data1);
    REQUIRE(data1.size() == 8192);
    REQUIRE(data2.size() == 8192);
    REQUIRE(data1 == data2);
    free(ptr1);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Copy assign between two destructables") {
    hshm::charwrap data1(8192);
    hshm::charwrap data2(512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    REQUIRE(data1 == data2);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Copy assign between two indestructables") {
    char *ptr1 = (char*)malloc(8192);
    hshm::charwrap data1(ptr1, 8192);
    char *ptr2 = (char*)malloc(512);
    hshm::charwrap data2(ptr2, 512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    REQUIRE(data1 == data2);
    free(ptr1);
    free(ptr2);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Copy assign indestructable -> destructable") {
    hshm::charwrap data1(8192);
    char *ptr2 = (char*)malloc(512);
    hshm::charwrap data2(ptr2, 512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    free(ptr2);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Copy assign destructable -> indestructable") {
    char *ptr1 = (char*)malloc(8192);
    hshm::charwrap data1(ptr1, 8192);
    hshm::charwrap data2(512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    free(ptr1);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  PAGE_DIVIDE("Copy assign to null") {
    char *ptr1 = (char*)malloc(8192);
    hshm::charwrap data1;
    hshm::charwrap data2(512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    free(ptr1);
  }
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("Charbuf") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  TestCharbuf();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
