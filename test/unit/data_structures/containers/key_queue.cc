//
// Created by llogan on 11/29/24.
//

#include "basic_test.h"
#include "test_init.h"
#include "list.h"

#include "hermes_shm/data_structures/data_structure.h"

struct Entry {
  size_t next_;
  size_t prior_;
  size_t value_;
};

TEST_CASE("key_queue") {
  hshm::key_queue<Entry> queue;
  queue.Init(0, 32);

  for (int i = 0; i < 64; ++i) {
    Entry entry;
    entry.value_ = i;
    queue.push(entry);
  }

  for (int i = 0; i < 64; ++i) {
    Entry entry;
    queue.pop(entry);
    REQUIRE(entry.value_ == i);
  }
}