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
#include "hermes_shm/data_structures/data_structure.h"

TEST_CASE("SerializePod") {
  Allocator *alloc = HERMES_MEMORY_MANAGER->GetDefaultAllocator();
  int a = 1;
  double b = 2;
  float c = 3;
  size_t size = sizeof(int) + sizeof(double) + sizeof(float);
  REQUIRE(hipc::ShmSerializer::shm_buf_size(a, b, c) == size);
  hipc::ShmSerializer istream(alloc, a, b, c);
  char *buf = istream.buf_;

  hipc::ShmDeserializer ostream;
  auto a2 = ostream.deserialize<int>(alloc, buf);
  REQUIRE(a2 == a);
  auto b2 = ostream.deserialize<double>(alloc, buf);
  REQUIRE(b2 == b);
  auto c2 = ostream.deserialize<float>(alloc, buf);
  REQUIRE(c2 == c);
}

TEST_CASE("SerializeString") {
  Allocator *alloc = HERMES_MEMORY_MANAGER->GetDefaultAllocator();

  auto a = hipc::make_uptr<hipc::string>(alloc, "h1");
  auto b = hipc::make_uptr<hipc::string>(alloc, "h2");
  int c;
  size_t size = 2 * sizeof(hipc::OffsetPointer) + sizeof(int);
  REQUIRE(hipc::ShmSerializer::shm_buf_size(*a, *b, c) == size);
  hipc::ShmSerializer istream(alloc, *a, *b, c);
  char *buf = istream.buf_;

  hipc::ShmDeserializer ostream;
  hipc::mptr<hipc::string> a2;
  ostream.deserialize<hipc::string>(alloc, buf, a2);
  REQUIRE(*a2 == *a);
  hipc::mptr<hipc::string> b2;
  ostream.deserialize<hipc::string>(alloc, buf, b2);
  REQUIRE(*b2 == *b);
  int c2 = ostream.deserialize<int>(alloc, buf);
  REQUIRE(c2 == c);
}

