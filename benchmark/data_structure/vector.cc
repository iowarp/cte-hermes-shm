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

// Boost interprocess
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

// Std
#include <string>
#include <vector>

// hermes
#include "hermes_shm/data_structures/ipc/string.h"
#include <hermes_shm/data_structures/ipc/vector.h>

namespace bipc = boost::interprocess;

#ifdef USE_BOOST_STRING
#define _USE_BOOST_STRING() bipc::string, bipc::string*
#else
#define _USE_BOOST_STRING()
#endif

template<typename T>
struct TestAllocator

template<typename T>
struct StringOrInt {
  typedef hshm::type_switch<T, int,
                            std::string, std::string,
                            hshm::string, hipc::uptr<hshm::string>,
                            bipc::string, bipc::string*> internal_t;
  internal_t internal_;

  /** Convert from int to internal_t */
  static void FromInt(int num) {
    if constexpr(std::is_same_v<T, int>) {
      internal_ = num;
    } else if constexpr(std::is_same_v<T, std::string>) {
      internal_ = std::to_string(num);
    } else if constexpr(std::is_same_v<T, hshm::string>) {
      internal_ = hshm::make_uptr<hshm::string>(std::to_string(num));
    } else if constexpr(std::is_same_v<T, bipc::string>) {
      internal_ =
    }
  }

  /** Get the internal type */
  T& Get() {
  }

  /** Convert internal_t to int */
  static int ToInt() {
  }
};

/**
 * A series of performance tests for vectors
 * OUTPUT:
 * [test_name] [vec_type] [internal_type] [time_ms]
 * */
template<typename T, typename VecT,
  typename VecTPtr=SHM_X_OR_Y(VecT, hipc::mptr<VecT>, VecT*)>
class VectorTest {
 public:
  std::string vec_type_;
  std::string internal_type_;
  VecT *vec_;
  VecTPtr vec_ptr_;

  /** Test case constructor */
  VectorTest() {
    if constexpr(std::is_same_v<std::vector<T>, VecT>) {
      vec_type_ = "std::vector";
    } else if constexpr(std::is_same_v<hipc::vector<T>, VecT>) {
      vec_type_ = "hipc::vector";
    } else if constexpr(std::is_same_v<bipc::vector<T>, VecT>) {
      vec_type_ = "bipc::vector";
    } else {
      std::cout << "INVALID: none of the vector tests matched" << std::endl;
      return;
    }

    if constexpr(std::is_same_v<T, hipc::string>) {
      internal_type_ = "hipc::string";
    } else if constexpr(std::is_same_v<T, std::string>) {
      internal_type_ = "std::string";
    } else if constexpr(std::is_same_v<T, bipc::ipc_string>) {
      internal_type_ = "bipc::string";
    } else if constexpr(std::is_same_v<T, int>) {
      internal_type_ = "int";
    }
  }

  /** Run the tests */
  void Test() {
    AllocateTest(1000000);
    ResizeTest(1000000);
    ReserveEmplaceTest(1000000);
    GetTest(1000000);
    ForwardIteratorTest(1000000);
    CopyTest(1000000);
    MoveTest(1000000);
  }

  /** Test performance of vector allocation */
  void AllocateTest(size_t count) {
    Timer t;
    t.Resume();
    for (size_t i = 0; i < count; ++i) {
      Allocate();
    }
    t.Pause();

    TestOutput("Allocate", t);
  }

  /** Test the performance of a resize */
  void ResizeTest(size_t count) {
    Timer t;
    CREATE_SET_VAR_TO_INT_OR_STRING(T, var, 124);

    Allocate();
    t.Resume();
    vec_->resize(count);
    t.Pause();

    TestOutput("FixedResize", t);
  }

  /** Emplace after reserving enough space */
  void ReserveEmplaceTest(size_t count) {
    Timer t;
    CREATE_SET_VAR_TO_INT_OR_STRING(T, var, 124);

    Allocate();
    t.Resume();
    vec_->reserve(count);
    for (size_t i = 0; i < count; ++i) {
      vec_->emplace_back(var);
    }
    t.Pause();

    TestOutput("FixedEmplace", t);
  }

  /** Get performance */
  void GetTest(size_t count) {
    Timer t;
    CREATE_SET_VAR_TO_INT_OR_STRING(T, var, 124);

    Allocate();
    vec_->reserve(count);
    for (size_t i = 0; i < count; ++i) {
      vec_->emplace_back(var);
    }

    t.Resume();
    for (size_t i = 0; i < count; ++i) {
      if constexpr(IS_SHM_ARCHIVEABLE(VecT)) {
        volatile hipc::Ref<T> x = (*vec_)[i];
        (void) x;
      } else {
        volatile auto &x = (*vec_)[i];
        (void) x;
      }
    }
    t.Pause();

    TestOutput("FixedGet", t);
  }

  /** Iterator performance */
  void ForwardIteratorTest(size_t count) {
    Timer t;
    CREATE_SET_VAR_TO_INT_OR_STRING(T, var, 124);

    Allocate();
    vec_->reserve(count);
    for (size_t i = 0; i < count; ++i) {
      vec_->emplace_back(var);
    }

    t.Resume();
    volatile int i = 0;
    for (size_t j = 0; j < 5; ++j) {
      if constexpr(IS_SHM_ARCHIVEABLE(VecT)) {
        for (volatile auto x : *vec_) {
          (void) x;
          ++i;
        }
      } else {
        for (volatile auto &x : *vec_) {
          (void) x;
          ++i;
        }
      }
    }
    t.Pause();

    TestOutput("ForwardIterator", t);
  }

  /** Copy performance */
  void CopyTest(size_t count) {
    Timer t;
    CREATE_SET_VAR_TO_INT_OR_STRING(T, var, 124);

    Allocate();
    vec_->reserve(count);
    for (size_t i = 0; i < count; ++i) {
      vec_->emplace_back(var);
    }

    t.Resume();
    if constexpr(IS_SHM_ARCHIVEABLE(VecT)) {
      volatile auto vec2 = hipc::make_uptr<VecT>(*vec_);
    } else {
      volatile VecT vec2(*vec_);
    }
    t.Pause();

    TestOutput("Copy", t);
  }

  /** Move performance */
  void MoveTest(size_t count) {
    Timer t;
    CREATE_SET_VAR_TO_INT_OR_STRING(T, var, 124);

    Allocate();
    vec_->reserve(count);
    for (size_t i = 0; i < count; ++i) {
      vec_->emplace_back(var);
    }

    t.Resume();
    if constexpr(IS_SHM_ARCHIVEABLE(VecT)) {
      volatile auto vec2 = hipc::make_uptr<VecT>(std::move(*vec_));
    } else {
      volatile VecT vec2(*vec_);
    }
    t.Pause();

    TestOutput("Move", t);
  }

 private:
  /** Output as CSV */
  void TestOutput(const std::string &test_name, Timer &t) {
    vec_->clear();
    HIPRINT("{}, {}, {}, {}",
            test_name, vec_type_, internal_type_, t.GetMsec())
  }

  /** Allocate an arbitrary vector for the test cases */
  void Allocate() {
    if constexpr(std::is_same_v<VecT, hipc::vector<T>>) {
      vec_ptr_ = hipc::make_mptr<VecT>();
    } else if constexpr(std::is_same_v<VecT, std::vector<T>>) {
      vec_ptr_ = new std::vector<T>();
    } else if constexpr (std::is_same_v<VecT, bipc::vector<T>>) {
      vec_ptr_ = AllocateBoostIpc();
    }
  }

  /** Destroy the vector */
  void Destroy() {
    if constexpr(std::is_same_v<VecT, hipc::vector<T>>) {
      vec_ptr_.shm_destroy();
    } else if constexpr(std::is_same_v<VecT, std::vector<T>>) {
      delete vec_ptr_;
    } else if constexpr (std::is_same_v<VecT, boost::container::vector<T>>) {
      delete vec_ptr_;
    } else if constexpr (std::is_same_v<VecT, bipc::vector<T>>) {
      FreeBoostIpc();
    }
  }

  /** Allocate a bipc::vector */
  VecT* AllocateBoostIpc() {
    bipc::void_allocator &alloc_inst = *alloc_inst_g;
    bipc::managed_shared_memory &segment = *segment_g;
    VecT *vec = segment.construct<VecT>("BoostVector")(alloc_inst);
    return vec;
  }

  /** Free a bipc::vector */
  void FreeBoostIpc() {
    bipc::void_allocator &alloc_inst = *alloc_inst_g;
    bipc::managed_shared_memory &segment = *segment_g;
    segment.destroy<VecT>("BoostVector");
  }
};

void FullVectorTest() {
  // std::vector tests
  VectorTest<int, std::vector<int>>().Test();
  VectorTest<std::string, std::vector<std::string>>().Test();
  VectorTest<hipc::string, std::vector<hipc::string>>().Test();

  // boost::ipc::vector tests
  VectorTest<int, bipc::vector<int>>().Test();
  VectorTest<std::string, bipc::vector<std::string>>().Test();
  VectorTest<bipc::ipc_string,
    bipc::vector<bipc::ipc_string>>().Test();

  // hipc::vector tests
  VectorTest<int, hipc::vector<int>>().Test();
  VectorTest<std::string, hipc::vector<std::string>>().Test();
  VectorTest<hipc::string, hipc::vector<hipc::string>>().Test();
}

TEST_CASE("VectorBenchmark") {
  FullVectorTest();
}
