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
  hipc::ShmSerializer istream;
  Allocator *alloc = HERMES_MEMORY_MANAGER->GetDefaultAllocator();
  int a = 1;
  double b = 2;
  float c = 3;
  int size = sizeof(int) + sizeof(double) + sizeof(float) + sizeof(allocator_id_t);
  REQUIRE(istream.shm_buf_size(alloc->GetId(), a, b, c) == size);
  char *buf = istream.serialize(alloc, a, b, c);

  hipc::ShmSerializer ostream;
  Allocator *alloc2 = ostream.deserialize(buf);
  REQUIRE(alloc == alloc2);
  auto a2 = ostream.deserialize<int>(alloc, buf);
  REQUIRE(a2 == a);
  auto b2 = ostream.deserialize<double>(alloc, buf);
  REQUIRE(b2 == b);
  auto c2 = ostream.deserialize<float>(alloc, buf);
  REQUIRE(c2 == c);
}

