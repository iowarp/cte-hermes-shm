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
#include <boost/interprocess/containers/vector.hpp>

// Std
#include <string>
#include <vector>

// hermes
#include "hermes_shm/data_structures/ipc/string.h"
#include <hermes_shm/data_structures/ipc/vector.h>

template<typename T>
using bipc_vector = bipc::vector<T, typename BoostAllocator<T>::alloc_t>;

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
  void *ptr_;

  /**====================================
   * Test Runner
   * ===================================*/

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
    } else if constexpr(std::is_same_v<T, bipc_string>) {
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

  /**====================================
   * Tests
   * ===================================*/

  /** Test performance of vector allocation */
  void AllocateTest(size_t count) {
    Timer t;
    t.Resume();
    for (size_t i = 0; i < count; ++i) {
      Allocate();
    }
    t.Pause();

    TestOutput("Allocate", t);
    Destroy();
  }

  /** Test the performance of a resize */
  void ResizeTest(size_t count) {
    Timer t;
    StringOrInt<T> var(124);

    Allocate();
    t.Resume();
    // vec_->resize(count);
    t.Pause();

    TestOutput("FixedResize", t);
    Destroy();
  }

  /** Emplace after reserving enough space */
  void ReserveEmplaceTest(size_t count) {
    Timer t;
    StringOrInt<T> var(124);

    Allocate();
    t.Resume();
    vec_->reserve(count);
    Emplace(count);
    t.Pause();

    TestOutput("FixedEmplace", t);
    Destroy();
  }

  /** Get performance */
  void GetTest(size_t count) {
    Timer t;
    StringOrInt<T> var(124);

    Allocate();
    vec_->reserve(count);
    Emplace(count);

    t.Resume();
    for (size_t i = 0; i < count; ++i) {
      Get(i);
    }
    t.Pause();

    TestOutput("FixedGet", t);
    Destroy();
  }

  /** Iterator performance */
  void ForwardIteratorTest(size_t count) {
    Timer t;
    StringOrInt<T> var(124);

    Allocate();
    vec_->reserve(count);
    Emplace(count);

    t.Resume();
    int i = 0;
    if constexpr(IS_SHM_ARCHIVEABLE(VecT)) {
      for (auto x : *vec_) {
        USE(*x);
        ++i;
      }
    } else {
      for (auto &x : *vec_) {
        USE(x);
        ++i;
      }
    }
    t.Pause();

    TestOutput("ForwardIterator", t);
    Destroy();
  }

  /** Copy performance */
  void CopyTest(size_t count) {
    Timer t;
    StringOrInt<T> var(124);

    Allocate();
    vec_->reserve(count);
    Emplace(count);

    t.Resume();
    if constexpr(IS_SHM_ARCHIVEABLE(VecT)) {
      auto vec2 = hipc::make_uptr<VecT>(*vec_);
    } else {
      VecT vec2(*vec_);
    }
    t.Pause();

    TestOutput("Copy", t);
    Destroy();
  }

  /** Move performance */
  void MoveTest(size_t count) {
    Timer t;

    Allocate();
    vec_->reserve(count);
    Emplace(count);

    t.Resume();
    if constexpr(IS_SHM_ARCHIVEABLE(VecT)) {
      auto vec2 = hipc::make_uptr<VecT>(std::move(*vec_));
      USE(vec2)
    } else {
      VecT vec2(*vec_);
      USE(vec2)
    }
    t.Pause();

    TestOutput("Move", t);
    Destroy();
  }

 private:
  /**====================================
   * Helpers
   * ===================================*/

  /** Output as CSV */
  void TestOutput(const std::string &test_name, Timer &t) {
    HIPRINT("{}, {}, {}, {}",
            test_name, vec_type_, internal_type_, t.GetMsec())
  }

  /** Get element at position i */
  void Get(size_t i) {
    if constexpr(std::is_same_v<VecT, std::vector<T>>) {
      T &x = (*vec_)[i];
      USE(x);
    } else if constexpr(std::is_same_v<VecT, bipc_vector<T>>) {
      T &x = (*vec_)[i];
      USE(x);
    } else if constexpr(std::is_same_v<VecT, hipc::vector<T>>) {
      hipc::Ref<T> x = (*vec_)[i];
      USE(*x);
    }
  }

  /** Emplace elements into the vector */
  void Emplace(size_t count) {
    StringOrInt<T> var(124);
    for (size_t i = 0; i < count; ++i) {
      if constexpr(std::is_same_v<VecT, std::vector<T>>) {
        vec_->emplace_back(var.Get());
      } else if constexpr(std::is_same_v<VecT, bipc_vector<T>>) {
        // vec_->emplace_back(var.Get());
      } else if constexpr(std::is_same_v<VecT, hipc::vector<T>>) {
        vec_->emplace_back(var.Get());
      }
    }
  }

  /** Allocate an arbitrary vector for the test cases */
  void Allocate() {
    if constexpr(std::is_same_v<VecT, hipc::vector<T>>) {
      vec_ptr_ = hipc::make_mptr<VecT>();
    } else if constexpr(std::is_same_v<VecT, std::vector<T>>) {
      vec_ptr_ = new std::vector<T>();
    } else if constexpr (std::is_same_v<VecT, bipc::vector<T>>) {
      vec_ptr_ = BOOST_SEGMENT->construct<VecT>("BoostVector")(
        BOOST_ALLOCATOR((std::pair<int, T>)));
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
      BOOST_SEGMENT->destroy<VecT>("BoostVector");
    }
  }
};

void FullVectorTest() {
  // std::vector tests
  VectorTest<int, std::vector<size_t>>().Test();
  VectorTest<std::string, std::vector<std::string>>().Test();

  // boost::ipc::vector tests
  VectorTest<int, bipc_vector<size_t>>().Test();
  VectorTest<std::string, bipc_vector<std::string>>().Test();
  VectorTest<bipc_string, bipc_vector<bipc_string>>().Test();

  // hipc::vector tests
  VectorTest<int, hipc::vector<size_t>>().Test();
  VectorTest<std::string, hipc::vector<std::string>>().Test();
  VectorTest<hipc::string, hipc::vector<hipc::string>>().Test();
}

TEST_CASE("VectorBenchmark") {
  FullVectorTest();
}
