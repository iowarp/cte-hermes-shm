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
#include "hermes_shm/data_structures/ipc/chararr.h"
#include "hermes_shm/data_structures/internal/template/hipc_container_template_ex.h"

using hshm::ipc::string;

void Testchararr() {
  auto *alloc = HSHM_DEFAULT_ALLOC;

  PAGE_DIVIDE("Construct from malloc") {
    char *ptr = (char*)malloc(256);
    hshm::chararr data(ptr, 256);
    memset(data.data(), 0, 256);
    REQUIRE(data.size() == 256);
    free(ptr);
  }

  PAGE_DIVIDE("Resize null chararr to higher value") {
    hshm::chararr data;
    data.resize(256);
    REQUIRE(data.size() == 256);
  }

  PAGE_DIVIDE("Resize null chararr to 0 value") {
    hshm::chararr data;
    data.resize(0);
    REQUIRE(data.size() == 0);
  }

  PAGE_DIVIDE("Resize destructable chararr to 0 value") {
    hshm::chararr data(4095);
    data.resize(0);
    REQUIRE(data.size() == 0);
  }

  PAGE_DIVIDE("Resize destructable chararr to lower value") {
    hshm::chararr data(4095);
    data.resize(256);
    REQUIRE(data.size() == 256);
  }

  PAGE_DIVIDE("Resize destructable chararr to higher value") {
    hshm::chararr data(256);
    data.resize(4095);
    REQUIRE(data.size() == 4095);
  }

  PAGE_DIVIDE("Resize indestructable chararr to higher value") {
    char *ptr = (char*)malloc(256);
    hshm::chararr data(ptr, 256);
    data.resize(4095);
    REQUIRE(data.size() == 4095);
    free(ptr);
  }

  PAGE_DIVIDE("Resize indestructable chararr to lower value") {
    char *ptr = (char*)malloc(4095);
    hshm::chararr data(ptr, 4095);
    data.resize(256);
    REQUIRE(data.size() == 256);
    free(ptr);
  }

  PAGE_DIVIDE("Move construct from destructable") {
    hshm::chararr data1(4095);
    hshm::chararr data2(std::move(data1));
    REQUIRE(data2.size() == 4095);
  }

  PAGE_DIVIDE("Move construct from indestructable") {
    char *ptr1 = (char*)malloc(4095);
    hshm::chararr data1(ptr1, 4095);
    hshm::chararr data2(std::move(data1));
    REQUIRE(data2.size() == 4095);
    free(ptr1);
  }

  PAGE_DIVIDE("Move assign between two destructables") {
    hshm::chararr data1(4095);
    hshm::chararr data2(512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
  }

  PAGE_DIVIDE("Move assign between two indestructables") {
    char *ptr1 = (char*)malloc(4095);
    hshm::chararr data1(ptr1, 4095);
    char *ptr2 = (char*)malloc(512);
    hshm::chararr data2(ptr2, 512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
    free(ptr1);
    free(ptr2);
  }

  PAGE_DIVIDE("Move assign indestructable -> destructable") {
    hshm::chararr data1(4095);
    char *ptr2 = (char*)malloc(512);
    hshm::chararr data2(ptr2, 512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
    free(ptr2);
  }

  PAGE_DIVIDE("Move assign destructable -> indestructable") {
    char *ptr1 = (char*)malloc(4095);
    hshm::chararr data1(ptr1, 4095);
    hshm::chararr data2(512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
    free(ptr1);
  }

  PAGE_DIVIDE("Move assign to null") {
    hshm::chararr data1;
    hshm::chararr data2(512);
    data1 = std::move(data2);
    REQUIRE(data1.size() == 512);
  }

  PAGE_DIVIDE("Copy construct from destructable") {
    hshm::chararr data1(4095);
    hshm::chararr data2(data1);
    REQUIRE(data1.size() == 4095);
    REQUIRE(data2.size() == 4095);
    REQUIRE(data1 == data2);
  }

  PAGE_DIVIDE("Copy construct from indestructable") {
    char *ptr1 = (char*)malloc(4095);
    hshm::chararr data1(ptr1, 4095);
    hshm::chararr data2(data1);
    REQUIRE(data1.size() == 4095);
    REQUIRE(data2.size() == 4095);
    REQUIRE(data1 == data2);
    free(ptr1);
  }

  PAGE_DIVIDE("Copy assign between two destructables") {
    hshm::chararr data1(4095);
    hshm::chararr data2(512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    REQUIRE(data1 == data2);
  }

  PAGE_DIVIDE("Copy assign between two indestructables") {
    char *ptr1 = (char*)malloc(4095);
    hshm::chararr data1(ptr1, 4095);
    char *ptr2 = (char*)malloc(512);
    hshm::chararr data2(ptr2, 512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    REQUIRE(data1 == data2);
    free(ptr1);
    free(ptr2);
  }

  PAGE_DIVIDE("Copy assign indestructable -> destructable") {
    hshm::chararr data1(4095);
    char *ptr2 = (char*)malloc(512);
    hshm::chararr data2(ptr2, 512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    free(ptr2);
  }

  PAGE_DIVIDE("Copy assign destructable -> indestructable") {
    char *ptr1 = (char*)malloc(4095);
    hshm::chararr data1(ptr1, 4095);
    hshm::chararr data2(512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    free(ptr1);
  }

  PAGE_DIVIDE("Copy assign to null") {
    char *ptr1 = (char*)malloc(4095);
    hshm::chararr data1;
    hshm::chararr data2(512);
    data1 = data2;
    REQUIRE(data2.size() == 512);
    REQUIRE(data1.size() == 512);
    free(ptr1);
  }
}

TEST_CASE("chararr") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Testchararr();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
