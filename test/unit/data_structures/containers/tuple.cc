//
// Created by lukemartinlogan on 1/15/23.
//

#include "basic_test.h"
#include "test_init.h"
#include "hermes_shm/data_structures/string.h"

template<typename FirstT, typename SecondT>
void TupleTest() {
  Allocator *alloc = alloc_g;

  // Construct test
  /*{
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::ShmHeader<hipc::ShmStruct<FirstT, SecondT>> hdr;
    hipc::ShmStruct<FirstT, SecondT>
      data(hdr, alloc,
           hermes::make_argpack(first),
           hermes::make_argpack(second));
    REQUIRE(data.template Get<0>() == first);
    REQUIRE(data.template Get<1>() == second);
  }

  // Copy test
  {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::tuple<FirstT, SecondT> data(alloc, first, second);
    hipc::tuple<FirstT, SecondT> cpy(data);
    REQUIRE(cpy.template Get<0>() == first);
    REQUIRE(cpy.template Get<1>() == second);
  }

  // Move test
  {
    CREATE_SET_VAR_TO_INT_OR_STRING(FirstT, first, 124);
    CREATE_SET_VAR_TO_INT_OR_STRING(SecondT, second, 130);
    hipc::tuple<FirstT, SecondT> data(alloc, first, second);
    hipc::tuple<FirstT, SecondT> cpy(std::move(data));
    REQUIRE(cpy.template Get<0>() == first);
    REQUIRE(cpy.template Get<1>() == second);
  }*/
}

#include <functional>

int y() {
  return 0;
}


class Y {
 public:
  Y() = default;
  Y(int x) {}
  Y(int x, int w) {}
};

TEST_CASE("TupleOfIntInt") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  TupleTest<int, int>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TupleOfIntString") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  // TupleTest<int, hipc::string>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}