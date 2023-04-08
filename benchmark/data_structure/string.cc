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

#include <string>
#include "hermes_shm/data_structures/ipc/string.h"

template<typename T>
class StringTestSuite {
 public:
  std::string str_type_;

  /**====================================
   * Test Cases
   * ===================================*/

  /** Constructor */
  StringTestSuite() {
    if constexpr(std::is_same_v<std::string, T>) {
      str_type_ = "std::string";
    } else if constexpr(std::is_same_v<hipc::string, T>) {
      str_type_ = "hipc::string";
    }
  }

  /** Construct + destruct in a loop */
  void ConstructDestructTest(size_t count, int length) {
    std::string data(length, 1);

    Timer t;
    t.Resume();
    for (size_t i = 0; i < count; ++i) {
      if constexpr(std::is_same_v<std::string, T>) {
        volatile T hello(data);
      } else if constexpr(std::is_same_v<hipc::string, T>) {
        volatile auto hello = hipc::make_uptr<hipc::string>(data);
      }
    }
    t.Pause();

    TestOutput("ConstructDestructTest", t, length);
  }

  /**====================================
   * Test Output
   * ===================================*/

  /** Output test results */
  void TestOutput(const std::string &test_name, Timer &t, size_t length) {
    HIPRINT("{},{},{},{}\n",
            test_name, str_type_, length, t.GetMsec())
  }
};

template<typename T>
void StringTest() {
  size_t count = 1000;
  StringTestSuite<T>().ConstructDestructTest(count, 16);
  StringTestSuite<T>().ConstructDestructTest(count, 256);
}

void FullStringTest() {
  StringTest<std::string>();
  StringTest<hipc::string>();
}

TEST_CASE("StringBenchmark") {
  FullStringTest();
}
