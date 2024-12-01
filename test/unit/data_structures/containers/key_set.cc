//
// Created by llogan on 11/29/24.
//

#include "basic_test.h"
#include "test_init.h"
#include "list.h"

#include "hermes_shm/data_structures/data_structure.h"

TEST_CASE("KeySet") {
  hshm::KeySet<size_t> count;
  count.Init(32);
  std::vector<size_t> keys;

  for (int i = 0; i < 64; ++i) {
    size_t entry = i;
    count.emplace(keys[i], entry);
  }

  for (int i = 0; i < 64; ++i) {
    size_t entry;
    count.pop(keys[i], entry);
    REQUIRE(entry == i);
  }
}