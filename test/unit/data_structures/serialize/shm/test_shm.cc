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

// Class with external serialize
struct ClassWithExternalSerialize {
  int z_;
};
namespace hshm::ipc {
template<typename Ar>
void serialize(Ar &ar, ClassWithExternalSerialize &obj) {
  ar(obj.z_);
}
}

// Class with external load/save
struct ClassWithExternalLoadSave {
  int z_;
};
namespace hshm::ipc {
template<typename Ar>
void save(Ar &ar, const ClassWithExternalLoadSave &obj) {
  ar(obj.z_);
}
template<typename Ar>
void load(Ar &ar, ClassWithExternalLoadSave &obj) {
  ar(obj.z_);
}
}

// Class with serialize
class ClassWithSerialize {
 public:
  int z_;

 public:
  template<typename Ar>
  void serialize(Ar& ar) {
    ar(z_);
  }
};

// Class with load/save
class ClassWithLoadSave {
 public:
  int z_;

 public:
  template<typename Ar>
  void save(Ar& ar) const {
    ar << z_;
  }

  template<typename Ar>
  void load(Ar& ar) {
    ar >> z_;
  }
};

TEST_CASE("SerializeExists") {
  std::string buf;
  buf.resize(8192);
  static_assert(hipc::has_load_fun_v<hipc::LocalSerialize<std::string>, ClassWithExternalLoadSave>);

  PAGE_DIVIDE("Arithmetic serialize, shift operator") {
    hipc::LocalSerialize srl(buf);
    int y = 25;
    int z = 30;
    srl << y;
    srl << z;
  }
  PAGE_DIVIDE("Arithmetic deserialize, shift operator") {
    hipc::LocalDeserialize srl(buf);
    int y;
    int z;
    srl >> y;
    srl >> z;
    REQUIRE(y == 25);
    REQUIRE(z == 30);
  }
  PAGE_DIVIDE("Arithmetic serialize, paren operator") {
    hipc::LocalSerialize srl(buf);
    int y = 27;
    int z = 41;
    srl(y, z);
  }
  PAGE_DIVIDE("Arithmetic deserialize, paren operator") {
    hipc::LocalDeserialize srl(buf);
    int y;
    int z;
    srl(y, z);
    REQUIRE(y == 27);
    REQUIRE(z == 41);
  }
  PAGE_DIVIDE("External serialize") {
    hipc::LocalSerialize srl(buf);
    ClassWithExternalSerialize y;
    y.z_ = 12;
    srl(y);
  }
  PAGE_DIVIDE("External deserialize") {
    hipc::LocalDeserialize srl(buf);
    ClassWithExternalSerialize y;
    srl(y);
    REQUIRE(y.z_ == 12);
  }
  PAGE_DIVIDE("External save") {
    hipc::LocalSerialize srl(buf);
    ClassWithExternalLoadSave y;
    y.z_ = 13;
    srl(y);
  }
  PAGE_DIVIDE("External load") {
    hipc::LocalDeserialize srl(buf);
    ClassWithExternalLoadSave y;
    srl(y);
    REQUIRE(y.z_ == 13);
  }
  PAGE_DIVIDE("Internal serialize") {
    hipc::LocalSerialize srl(buf);
    ClassWithSerialize y;
    y.z_ = 14;
    srl(y);
  }
  PAGE_DIVIDE("Internal deserialize") {
    hipc::LocalDeserialize srl(buf);
    ClassWithSerialize y;
    srl(y);
    REQUIRE(y.z_ == 14);
  }
  PAGE_DIVIDE("Internal save") {
    hipc::LocalSerialize srl(buf);
    ClassWithLoadSave y;
    y.z_ = 15;
    srl(y);
  }
  PAGE_DIVIDE("Internal load") {
    hipc::LocalDeserialize srl(buf);
    ClassWithLoadSave y;
    srl(y);
    REQUIRE(y.z_ == 15);
  }
}
