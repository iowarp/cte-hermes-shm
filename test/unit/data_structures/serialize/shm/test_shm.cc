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
#include "hermes_shm/data_structures/all.h"
#include "hermes_shm/data_structures/ipc/string.h"
#include "test_init.h"

// Class with external serialize
struct ClassWithExternalSerialize {
  int z_;
};
namespace hshm::ipc {
template <typename Ar>
void serialize(Ar &ar, ClassWithExternalSerialize &obj) {
  ar(obj.z_);
}
}  // namespace hshm::ipc

// Class with external load/save
struct ClassWithExternalLoadSave {
  int z_;
};
namespace hshm::ipc {
template <typename Ar>
void save(Ar &ar, const ClassWithExternalLoadSave &obj) {
  ar(obj.z_);
}
template <typename Ar>
void load(Ar &ar, ClassWithExternalLoadSave &obj) {
  ar(obj.z_);
}
}  // namespace hshm::ipc

// Class with serialize
class ClassWithSerialize {
 public:
  int z_;

 public:
  template <typename Ar>
  void serialize(Ar &ar) {
    ar(z_);
  }
};

// Class with load/save
template <typename T>
class ClassWithLoadSave {
 public:
  T z_;

 public:
  template <typename Ar>
  HSHM_CROSS_FUN void save(Ar &ar) const {
    ar << z_;
  }

  template <typename Ar>
  HSHM_CROSS_FUN void load(Ar &ar) {
    ar >> z_;
  }
};

TEST_CASE("SerializeExists") {
  std::string buf;
  buf.resize(8192);
  STATIC_ASSERT((hipc::has_load_fun_v<hipc::LocalSerialize<std::string>,
                                      ClassWithExternalLoadSave>),
                "", void);

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
    ClassWithLoadSave<int> y;
    y.z_ = 15;
    srl(y);
  }
  PAGE_DIVIDE("Internal load") {
    hipc::LocalDeserialize srl(buf);
    ClassWithLoadSave<int> y;
    srl(y);
    REQUIRE(y.z_ == 15);
  }
}

/** Serialize data structures */
TEST_CASE("SerializeHshm") {
  std::string buf;
  buf.resize(8192);

  /** std::string */
  PAGE_DIVIDE("std::string") {
    hipc::LocalSerialize srl(buf);
    std::string h("hello");
    srl(h);
  }
  PAGE_DIVIDE("std::string") {
    hipc::LocalDeserialize srl(buf);
    std::string h;
    srl(h);
    REQUIRE(h == "hello");
  }
  /** std::vector */
  PAGE_DIVIDE("std::vector") {
    hipc::LocalSerialize srl(buf);
    std::vector<int> h(5);
    for (int i = 0; i < 5; ++i) {
      h[i] = i;
    }
    srl(h);
  }
  PAGE_DIVIDE("std::vector") {
    hipc::LocalDeserialize srl(buf);
    std::vector<int> h(5);
    srl(h);
    for (int i = 0; i < 5; ++i) {
      REQUIRE(h[i] == i);
    }
  }
  /** std::unordered_map */
  PAGE_DIVIDE("std::unordered_map") {
    hipc::LocalSerialize srl(buf);
    std::unordered_map<int, int> h;
    h[10] = 2;
    srl(h);
  }
  PAGE_DIVIDE("std::unordered_map") {
    hipc::LocalDeserialize srl(buf);
    std::unordered_map<int, int> h;
    srl(h);
    REQUIRE(h[10] == 2);
  }
  /** hipc::charbuf */
  PAGE_DIVIDE("hipc::charbuf") {
    hipc::LocalSerialize srl(buf);
    hipc::charbuf h("hello");
    srl(h);
  }
  PAGE_DIVIDE("hipc::charbuf") {
    hipc::LocalDeserialize srl(buf);
    hipc::charbuf h;
    srl(h);
    REQUIRE(h == "hello");
  }
}
