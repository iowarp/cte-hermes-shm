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
#include "hermes_shm/data_structures/ipc/pair.h"
#include "hermes_shm/data_structures/ipc/string.h"

template<typename FirstT, typename SecondT>
void PairTest() {
  hipc::CtxAllocator<HSHM_DEFAULT_ALLOC_T> alloc(HSHM_DEFAULT_ALLOC);

  // Construct test
  PAGE_DIVIDE("Construct") {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::pair<FirstT, SecondT> data(first, second);
    REQUIRE(data.GetFirst() == first);
    REQUIRE(data.GetSecond() == second);
  }

  // SHM Construct test
  PAGE_DIVIDE("SHM Construct") {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::pair<FirstT, SecondT> data(
      alloc, first, second);
    REQUIRE(data.GetFirst() == first);
    REQUIRE(data.GetSecond() == second);
  }

  // Copy constructor test
  PAGE_DIVIDE("Copy constructor") {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::pair<FirstT, SecondT> data(
      alloc, first, second);
    hipc::pair<FirstT, SecondT> cpy(data);
    REQUIRE(cpy.GetFirst() == first);
    REQUIRE(cpy.GetSecond() == second);
  }

  // SHM Copy constructor test
  PAGE_DIVIDE("SHM Copy constructor") {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::pair<FirstT, SecondT> data(
      alloc, first, second);
    hipc::pair<FirstT, SecondT> cpy(
      alloc, data);
    REQUIRE(cpy.GetFirst() == first);
    REQUIRE(cpy.GetSecond() == second);
  }

  // Copy assignment test
  PAGE_DIVIDE("Copy assignment operator") {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::pair<FirstT, SecondT> data(
      alloc, first, second);
    hipc::pair<FirstT, SecondT> cpy(
      alloc);
    cpy = data;
    REQUIRE(cpy.GetFirst() == first);
    REQUIRE(cpy.GetSecond() == second);
  }

  // Move constructor test
  PAGE_DIVIDE("Move constructor") {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::pair<FirstT, SecondT> data(
      alloc, first, second);
    hipc::pair<FirstT, SecondT> cpy(std::move(data));
    REQUIRE(cpy.GetFirst() == first);
    REQUIRE(cpy.GetSecond() == second);
  }

  // SHM Move constructor test
  PAGE_DIVIDE("SHM Move constructor") {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::pair<FirstT, SecondT> data(
      alloc, first, second);
    hipc::pair<FirstT, SecondT> cpy(
      alloc, std::move(data));
    REQUIRE(cpy.GetFirst() == first);
    REQUIRE(cpy.GetSecond() == second);
  }

  // Move assignment test
  PAGE_DIVIDE("Move assignment operator") {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::pair<FirstT, SecondT> data(
      alloc, first, second);
    hipc::pair<FirstT, SecondT> cpy(
      alloc);
    cpy = std::move(data);
    REQUIRE(cpy.GetFirst() == first);
    REQUIRE(cpy.GetSecond() == second);
  }
}

TEST_CASE("PairOfIntInt") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  PairTest<int, int>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("PairOfIntString") {
  auto *alloc = HSHM_DEFAULT_ALLOC;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  PairTest<int, hipc::string>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
