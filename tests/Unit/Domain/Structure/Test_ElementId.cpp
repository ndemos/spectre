// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <pup.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
template <size_t Dim>
struct DirectionTester : public ElementId<Dim> {
  DirectionTester(const Direction<Dim>& dir, const ElementId<Dim>& id)
      : ElementId<Dim>(dir, id) {}
  using ElementId<Dim>::direction;
};

template <size_t VolumeDim>
void test_placement_new_and_hashing_impl(
    const size_t block1, const std::array<SegmentId, VolumeDim>& segments1,
    const size_t grid1, const size_t block2,
    const std::array<SegmentId, VolumeDim>& segments2, const size_t grid2) {
  using Hash = std::hash<ElementId<VolumeDim>>;

  const ElementId<VolumeDim> id1(block1, segments1, grid1);
  const ElementId<VolumeDim> id2(block2, segments2, grid2);

  ElementId<VolumeDim> test_id1{};
  ElementId<VolumeDim> test_id2{};
  // Check for nondeterminacy due to previous memory state.
  std::memset(&test_id1, 0, sizeof(test_id1));
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif  // defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
  std::memset(&test_id2, 255, sizeof(test_id2));
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 7)
  new (&test_id1) ElementId<VolumeDim>(id1);
  new (&test_id2) ElementId<VolumeDim>(id2);

  CHECK((test_id1 == test_id2) == (id1 == id2));
  CHECK((test_id1 != test_id2) == (id1 != id2));
  CHECK((Hash{}(test_id1) == Hash{}(test_id2)) == (id1 == id2));

  for (const auto& dir : Direction<VolumeDim>::all_directions()) {
    const DirectionTester<VolumeDim> dir_test_id1{dir, test_id1};
    CHECK(static_cast<const ElementId<VolumeDim>&>(dir_test_id1) == test_id1);
    CHECK(Hash{}(dir_test_id1) == Hash{}(test_id1));
  }
}

void test_placement_new_and_hashing() {
  const std::array<size_t, 3> blocks{{0, 1, 4}};
  const std::array<SegmentId, 4> segments{{{0, 0}, {1, 0}, {1, 1}, {8, 4}}};
  const std::array<size_t, 3> grids{{0, 1, 3}};

  for (const auto& block1 : blocks) {
    for (const auto& block2 : blocks) {
      for (const auto& grid1 : grids) {
        for (const auto& grid2 : grids) {
          for (const auto& segment10 : segments) {
            for (const auto& segment20 : segments) {
              test_placement_new_and_hashing_impl<1>(
                  block1, {{segment10}}, grid1, block2, {{segment20}}, grid2);
              for (const auto& segment11 : segments) {
                for (const auto& segment21 : segments) {
                  test_placement_new_and_hashing_impl<2>(
                      block1, {{segment10, segment11}}, grid1, block2,
                      {{segment20, segment21}}, grid2);
                  for (const auto& segment12 : segments) {
                    for (const auto& segment22 : segments) {
                      test_placement_new_and_hashing_impl<3>(
                          block1, {{segment10, segment11, segment12}}, grid1,
                          block2, {{segment20, segment21, segment22}}, grid2);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void test_element_id() {
  // Test retrieval functions:
  auto segment_ids = std::array<SegmentId, 3>(
      {{SegmentId(2, 3), SegmentId(1, 0), SegmentId(1, 1)}});
  ElementId<3> block_2_3d(2, segment_ids);
  CHECK(block_2_3d.block_id() == 2);
  CHECK(block_2_3d.segment_ids() == segment_ids);
  CHECK(block_2_3d.segment_id(0) == segment_ids[0]);
  CHECK(block_2_3d.segment_id(1) == segment_ids[1]);
  CHECK(block_2_3d.segment_id(2) == segment_ids[2]);
  CHECK(block_2_3d.grid_index() == 0);
  CHECK(block_2_3d.refinement_levels() == std::array{2_st, 1_st, 1_st});
  CHECK(ElementId<2>{31, {{{6, 3}, {9, 67}}}}.refinement_levels() ==
        std::array{6_st, 9_st});
  CHECK(ElementId<1>{4, {{{4, 7}}}}.refinement_levels() == std::array{4_st});

  // Test parent and child operations:
  const auto check_parent_and_child = [](const ElementId<3>& id) {
    for (size_t dim = 0; dim < 3; dim++) {
      CHECK(id == id.id_of_child(dim, Side::Lower).id_of_parent(dim));
      CHECK(id == id.id_of_child(dim, Side::Upper).id_of_parent(dim));
      if (0 == id.segment_id(dim).index() % 2) {
        CHECK(id == id.id_of_parent(dim).id_of_child(dim, Side::Lower));
      } else {
        CHECK(id == id.id_of_parent(dim).id_of_child(dim, Side::Upper));
      }
    }
  };
  check_parent_and_child(block_2_3d);
  check_parent_and_child({3, segment_ids, 4});

  // Test equality operator:
  ElementId<3> element_one(1);
  ElementId<3> element_two(1);
  ElementId<3> element_three(2);
  ElementId<3> element_four(4);
  ElementId<3> element_five(4, 0);
  ElementId<3> element_six(4, 1);
  CHECK(element_one == element_two);
  CHECK(element_two != element_three);
  CHECK(element_two != element_four);
  CHECK(element_three != block_2_3d);
  CHECK(element_five == element_four);
  CHECK(element_six != element_four);

  // Test comparison operator
  check_cmp(ElementId<1>{1, {{{1, 0}}}, 0}, ElementId<1>{1, {{{1, 1}}}, 0});
  check_cmp(ElementId<1>{0, {{{1, 0}}}, 0}, ElementId<1>{1, {{{1, 0}}}, 0});
  check_cmp(ElementId<1>{1, {{{0, 0}}}, 0}, ElementId<1>{1, {{{1, 0}}}, 0});
  check_cmp(ElementId<1>{1, {{{1, 0}}}, 0}, ElementId<1>{1, {{{1, 0}}}, 1});
  check_cmp(ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 0}, {2, 2}}}, 0});
  check_cmp(ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 1}, {2, 1}}}, 0});
  check_cmp(ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 1}, {1, 1}}}, 0});
  check_cmp(ElementId<2>{1, {{{0, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 0}, {1, 1}}}, 0});
  check_cmp(ElementId<2>{1, {{{1, 1}, {1, 1}}}, 0},
            ElementId<2>{1, {{{1, 1}, {2, 1}}}, 0});
  check_cmp(ElementId<2>{0, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0});
  check_cmp(ElementId<2>{1, {{{1, 0}, {2, 1}}}, 0},
            ElementId<2>{1, {{{1, 0}, {2, 1}}}, 1});
  check_cmp(ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 0}, {2, 2}, {1, 0}}}, 0});
  check_cmp(ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 1}, {2, 1}, {1, 0}}}, 0});
  check_cmp(ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 1}}}, 0});
  check_cmp(ElementId<3>{0, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0});
  check_cmp(ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 0},
            ElementId<3>{1, {{{1, 0}, {2, 1}, {1, 0}}}, 1});

  // Test pup operations:
  test_serialization(element_one);
  test_serialization(element_six);

  // Test output operator:
  CHECK(get_output(block_2_3d) == "[B2,(L2I3,L1I0,L1I1)]");
  CHECK(get_output(element_six) == "[B4,(L0I0,L0I0,L0I0),G1]");
  CHECK(ElementId<3>{"[B2,(L2I3,L1I0,L1I1)]"} == block_2_3d);
  CHECK(ElementId<3>{"[B4,(L0I0,L0I0,L0I0),G1]"} == element_six);
  CHECK(ElementId<1>{"[B1,(L2I3)]"} == ElementId<1>{1, {{{2, 3}}}});
  CHECK(ElementId<1>{"[B1,(L2I3),G2]"} == ElementId<1>{1, {{{2, 3}}}, 2});
  CHECK(ElementId<1>{"[B10,(L2I3)]"} == ElementId<1>{10, {{{2, 3}}}});
  CHECK(ElementId<2>{"[B2,(L1I1,L2I0)]"} ==
        ElementId<2>{2, {{{1, 1}, {2, 0}}}});
  CHECK(ElementId<2>{"[B2,(L1I1,L2I0),G12]"} ==
        ElementId<2>{2, {{{1, 1}, {2, 0}}}, 12});
  CHECK(ElementId<2>{"[B52,(L12I133,L6I38)]"} ==
        ElementId<2>{52, {{{12, 133}, {6, 38}}}});
  CHECK_THROWS_WITH(ElementId<1>("somegrid"),
                    Catch::Matchers::ContainsSubstring("Invalid grid name"));
  CHECK_THROWS_WITH(ElementId<2>("[B0,(L1I0)]"),
                    Catch::Matchers::ContainsSubstring("Invalid grid name"));
  CHECK_THROWS_WITH(ElementId<2>("[B0]"),
                    Catch::Matchers::ContainsSubstring("Invalid grid name"));
  CHECK_THROWS_WITH(ElementId<3>("L1I0,L2I1,L2I0"),
                    Catch::Matchers::ContainsSubstring("Invalid grid name"));

  CHECK(ElementId<3>::external_boundary_id().block_id() ==
        two_to_the(ElementId<3>::block_id_bits) - 1);
  CHECK(ElementId<3>::external_boundary_id().grid_index() == 0);

  const ElementId<3> element1(0);
  const ElementId<3> element2{0, 1};
  const ElementId<3> element3(1);
  const ElementId<3> element4{1, 1};
  const ElementId<3> element5{0, {{{1, 0}, {2, 0}, {1, 0}}}};
  const ElementId<3> element6{0, {{{1, 0}, {2, 0}, {1, 0}}}, 1};
  const ElementId<3> element7{0, {{{1, 0}, {2, 1}, {1, 0}}}};
  const ElementId<3> element8{0, {{{1, 0}, {2, 1}, {1, 0}}}, 1};
  const ElementId<3> element9{1, {{{1, 0}, {2, 0}, {1, 0}}}};
  const ElementId<3> element10{1, {{{1, 0}, {2, 0}, {1, 0}}}, 1};
  const ElementId<3> element11{1, {{{1, 0}, {2, 1}, {1, 0}}}};
  const ElementId<3> element12{1, {{{1, 0}, {2, 1}, {1, 0}}}, 1};

  CHECK(is_zeroth_element(element1));
  CHECK_FALSE(is_zeroth_element(element1, {1}));
  CHECK(is_zeroth_element(element2));
  CHECK_FALSE(is_zeroth_element(element2, {0}));
  CHECK(is_zeroth_element(element2, {1}));
  CHECK(is_zeroth_element(element5));
  CHECK_FALSE(is_zeroth_element(element5, {1}));
  CHECK(is_zeroth_element(element6));
  CHECK_FALSE(is_zeroth_element(element6, {0}));
  CHECK(is_zeroth_element(element6, {1}));
  // Do this just so we don't duplicate the same two checks over and over for
  // all elements that aren't the zeroth element
  const std::vector<ElementId<3>> not_zeroth_elements{
      {element3, element4, element7, element8, element9, element10, element11,
       element12}};
  for (const auto& element : not_zeroth_elements) {
    CHECK_FALSE(is_zeroth_element(element));
    CHECK_FALSE(is_zeroth_element(element, {1}));
  }

  CHECK(ElementId<1>{0, {{{0, 0}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<1>{0, {{{1, 0}}}}.number_of_block_boundaries() == 1);
  CHECK(ElementId<1>{0, {{{1, 1}}}}.number_of_block_boundaries() == 1);
  CHECK(ElementId<1>{0, {{{2, 0}}}}.number_of_block_boundaries() == 1);
  CHECK(ElementId<1>{0, {{{2, 1}}}}.number_of_block_boundaries() == 0);
  CHECK(ElementId<1>{0, {{{2, 3}}}}.number_of_block_boundaries() == 1);

  CHECK(ElementId<2>{0, {{{0, 0}, {0, 0}}}}.number_of_block_boundaries() == 4);
  CHECK(ElementId<2>{0, {{{1, 0}, {0, 0}}}}.number_of_block_boundaries() == 3);
  CHECK(ElementId<2>{0, {{{0, 0}, {1, 0}}}}.number_of_block_boundaries() == 3);
  CHECK(ElementId<2>{0, {{{1, 0}, {1, 0}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{1, 1}, {1, 0}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{1, 0}, {1, 1}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{2, 0}, {1, 0}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{2, 3}, {1, 0}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{1, 0}, {2, 0}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{1, 0}, {2, 3}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{2, 1}, {1, 0}}}}.number_of_block_boundaries() == 1);
  CHECK(ElementId<2>{0, {{{1, 0}, {2, 2}}}}.number_of_block_boundaries() == 1);
  CHECK(ElementId<2>{0, {{{2, 0}, {2, 0}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{2, 0}, {2, 2}}}}.number_of_block_boundaries() == 1);
  CHECK(ElementId<2>{0, {{{2, 2}, {2, 0}}}}.number_of_block_boundaries() == 1);
  CHECK(ElementId<2>{0, {{{2, 3}, {2, 0}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{2, 0}, {2, 3}}}}.number_of_block_boundaries() == 2);
  CHECK(ElementId<2>{0, {{{2, 3}, {2, 3}}}}.number_of_block_boundaries() == 2);

  CHECK(ElementId<3>{0, {{{0, 0}, {0, 0}, {0, 0}}}}
            .number_of_block_boundaries() == 6);
  CHECK(ElementId<3>{0, {{{1, 0}, {0, 0}, {0, 0}}}}
            .number_of_block_boundaries() == 5);
  CHECK(ElementId<3>{0, {{{0, 0}, {1, 0}, {0, 0}}}}
            .number_of_block_boundaries() == 5);
  CHECK(ElementId<3>{0, {{{0, 0}, {0, 0}, {1, 0}}}}
            .number_of_block_boundaries() == 5);
  CHECK(ElementId<3>{0, {{{1, 1}, {0, 0}, {0, 0}}}}
            .number_of_block_boundaries() == 5);
  CHECK(ElementId<3>{0, {{{0, 0}, {1, 1}, {0, 0}}}}
            .number_of_block_boundaries() == 5);
  CHECK(ElementId<3>{0, {{{0, 0}, {0, 0}, {1, 1}}}}
            .number_of_block_boundaries() == 5);
  CHECK(ElementId<3>{0, {{{1, 0}, {1, 0}, {0, 0}}}}
            .number_of_block_boundaries() == 4);
  CHECK(ElementId<3>{0, {{{1, 0}, {0, 0}, {1, 0}}}}
            .number_of_block_boundaries() == 4);
  CHECK(ElementId<3>{0, {{{1, 0}, {1, 0}, {1, 0}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{1, 1}, {1, 0}, {1, 0}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{1, 0}, {1, 1}, {1, 0}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{1, 0}, {1, 0}, {1, 1}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{2, 0}, {1, 0}, {1, 0}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{1, 0}, {2, 0}, {1, 0}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{1, 0}, {1, 0}, {2, 0}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{2, 3}, {1, 0}, {1, 0}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{1, 0}, {2, 3}, {1, 0}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{1, 0}, {1, 0}, {2, 3}}}}
            .number_of_block_boundaries() == 3);
  CHECK(ElementId<3>{0, {{{2, 1}, {1, 0}, {1, 0}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{1, 0}, {2, 1}, {1, 0}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{1, 0}, {1, 0}, {2, 1}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{2, 1}, {2, 0}, {1, 0}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{2, 1}, {1, 0}, {2, 0}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{2, 0}, {2, 1}, {1, 0}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{1, 0}, {2, 1}, {2, 0}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{2, 0}, {1, 0}, {2, 1}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{1, 0}, {2, 0}, {2, 1}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{2, 0}, {2, 0}, {2, 1}}}}
            .number_of_block_boundaries() == 2);
  CHECK(ElementId<3>{0, {{{2, 1}, {2, 0}, {2, 1}}}}
            .number_of_block_boundaries() == 1);
  CHECK(ElementId<3>{0, {{{2, 0}, {2, 1}, {2, 1}}}}
            .number_of_block_boundaries() == 1);
  CHECK(ElementId<3>{0, {{{2, 1}, {2, 1}, {2, 0}}}}
            .number_of_block_boundaries() == 1);
  CHECK(ElementId<3>{0, {{{2, 1}, {2, 1}, {2, 1}}}}
            .number_of_block_boundaries() == 0);

  CHECK(overlapping(element1, element1));
  CHECK(overlapping(element1, element2));
  CHECK(not overlapping(element1, element3));
  CHECK(overlapping(element1, element5));
  CHECK(not overlapping(element5, element7));
}

template <size_t VolumeDim>
void test_serialization() {
  constexpr size_t volume_dim = VolumeDim;

  // Generate random element IDs so we test the full range of possible values
  MAKE_GENERATOR(gen);
  std::uniform_int_distribution<size_t> dist_block_id(
      0, two_to_the(ElementId<VolumeDim>::block_id_bits) - 1);
  std::uniform_int_distribution<size_t> dist_grid_index(
      0, two_to_the(ElementId<VolumeDim>::grid_index_bits) - 1);
  std::uniform_int_distribution<size_t> dist_refinement(
      0, ElementId<VolumeDim>::max_refinement_level);
  const auto random_segment_id = [&gen, &dist_refinement]() -> SegmentId {
    const size_t refinement = dist_refinement(gen);
    std::uniform_int_distribution<size_t> dist_index(
        0, two_to_the(refinement) - 1);
    return {refinement, dist_index(gen)};
  };
  const auto random_segment_ids =
      [&random_segment_id]() -> std::array<SegmentId, VolumeDim> {
    if constexpr (VolumeDim == 1) {
      return {{random_segment_id()}};
    } else if constexpr (VolumeDim == 2) {
      return {{random_segment_id(), random_segment_id()}};
    } else {
      return {{random_segment_id(), random_segment_id(), random_segment_id()}};
    }
  };

  const ElementId<volume_dim> unused_id(0);
  CHECK(size_of_object_in_bytes(unused_id) == 8);
  for (size_t i = 0; i < 100; ++i) {
    ElementId<volume_dim> element_id{dist_block_id(gen), random_segment_ids(),
                                     dist_grid_index(gen)};
    CAPTURE(element_id);

    // Test serialization
    const auto serialized_id = serialize_and_deserialize(element_id);
    CHECK(serialized_id == element_id);

    // The following checks that ElementId can be used as a Charm array
    // index
    Parallel::ArrayIndex<ElementId<volume_dim>> array_index(element_id);
    CHECK(element_id == array_index.get_index());
    // now check pupping the ArrayIndex works...
    const auto serialized_array_index =
        serialize<Parallel::ArrayIndex<ElementId<volume_dim>>>(array_index);
    PUP::fromMem reader(serialized_array_index.data());
    Parallel::ArrayIndex<ElementId<volume_dim>> deserialized_array_index(
        unused_id);
    reader | deserialized_array_index;
    CHECK(array_index == deserialized_array_index);
    CHECK(element_id == deserialized_array_index.get_index());

    // Check roundtrip to string representation and back
    CHECK(ElementId<VolumeDim>(get_output(element_id)) == element_id);
  }
}

// to_short_id() returns the 45 segment ID bits: the compact xi in bits 0-14,
// eta in bits 15-29 and zeta in bits 30-44, where a compact segment ID is
// 2^level + index. The expected values below are computed from the compact
// segment IDs, independently of how the implementation extracts them.
void test_to_short_id() {
  static_assert(ElementId<3>::max_refinement_level == 14 and
                ElementId<3>::compact_segment_id_xi_lo_bits == 13);
  // Same segments, different block id or grid index -> same short id
  const ElementId<3> elem_b0{0, {{{2, 3}, {1, 0}, {1, 1}}}};
  const ElementId<3> elem_b5{5, {{{2, 3}, {1, 0}, {1, 1}}}};
  CHECK(elem_b0.to_short_id() == elem_b5.to_short_id());
  const ElementId<3> elem_g0{0, {{{2, 3}, {1, 0}, {1, 1}}}, 0};
  const ElementId<3> elem_g3{0, {{{2, 3}, {1, 0}, {1, 1}}}, 3};
  CHECK(elem_g0.to_short_id() == elem_g3.to_short_id());

  // xi L14 I10923: compact 2^14 + 10923 = 27307 = 11|0101010101011, with bits
  // in both the high part (11) and the low part (2731) of the split xi
  // bitfield. eta L3 I5: compact 2^3 + 5 = 13. zeta L14 I1: compact
  // 2^14 + 1 = 16385.
  const ElementId<3> id{0, {{{14, 10923}, {3, 5}, {14, 1}}}};
  const size_t short_id = id.to_short_id();
  CHECK((short_id bitand 0x7FFF_st) == 27307_st);
  CHECK(((short_id >> 15) bitand 0x7FFF_st) == 13_st);
  CHECK((short_id >> 30) == 16385_st);
  // 27307 + 13 * 2^15 + 16385 * 2^30
  CHECK(short_id == 17593260239531_st);

  // A direction other than self gives the same short id; a shift by 16 would
  // put direction bits 1-3 into bits 0-2
  const DirectionTester<3> id_upper_xi{Direction<3>::upper_xi(), id};
  CHECK(id_upper_xi.direction() == Direction<3>::upper_xi());
  CHECK(id_upper_xi.to_short_id() == short_id);
  const DirectionTester<3> id_lower_zeta{Direction<3>::lower_zeta(), id};
  CHECK(id_lower_zeta.direction() == Direction<3>::lower_zeta());
  CHECK(id_lower_zeta.to_short_id() == short_id);

  // The xi and eta above are odd, so they cannot detect an implementation
  // that sets bit 0 of the short id, or one that leaks direction bit 0 (bit 15
  // of the first word) into bit 15 of the short id (bit 0 of eta). xi L14
  // I10922: compact 2^14 + 10922 = 27306 = 11|0101010101010, even, with bits
  // in both parts of the split xi bitfield. eta L3 I4: compact 2^3 + 4 = 12,
  // even. zeta L14 I1: 16385 as above.
  const ElementId<3> id_even{0, {{{14, 10922}, {3, 4}, {14, 1}}}};
  const size_t short_id_even = id_even.to_short_id();
  CHECK((short_id_even bitand 0x7FFF_st) == 27306_st);
  CHECK(((short_id_even >> 15) bitand 0x7FFF_st) == 12_st);
  CHECK((short_id_even >> 30) == 16385_st);
  // 27306 + 12 * 2^15 + 16385 * 2^30
  CHECK(short_id_even == 17593260206762_st);
  // Direction bits are axis bitor side, so an eta direction sets direction
  // bit 0: upper eta is 0b01 bitor 0b1000 = 0b1001, whereas self, upper xi
  // and lower zeta are 0b1100, 0b1000 and 0b0110
  const DirectionTester<3> id_upper_eta{Direction<3>::upper_eta(), id_even};
  CHECK(id_upper_eta.direction() == Direction<3>::upper_eta());
  CHECK(id_upper_eta.to_short_id() == short_id_even);

  // Each compact value is a single power of two, so only bits 14, 18 and 44
  // of the short id are set, and an implementation that ORs a constant into
  // any other bit fails, e.g. `| 2`, which the ids above miss (each of their
  // xi has bit 1 set). xi L14 I0: compact 2^14 = 16384 = 10|0000000000000.
  // eta L3 I0: compact 2^3 = 8. zeta L14 I0: compact 2^14 = 16384.
  const ElementId<3> id_pow2{0, {{{14, 0}, {3, 0}, {14, 0}}}};
  const size_t short_id_pow2 = id_pow2.to_short_id();
  CHECK((short_id_pow2 bitand 0x7FFF_st) == 16384_st);
  CHECK(((short_id_pow2 >> 15) bitand 0x7FFF_st) == 8_st);
  CHECK((short_id_pow2 >> 30) == 16384_st);
  // 16384 + 8 * 2^15 + 16384 * 2^30
  CHECK(short_id_pow2 == 17592186322944_st);

  // xi below level 14, so that bit 14 is 0: xi L13 I0: compact 2^13 = 8192 =
  // 01|0000000000000. eta L0 I0 and zeta L0 I0: compact 2^0 = 1. Only bits
  // 13, 15 and 30 of the short id are set, none of them set above.
  const ElementId<3> id_pow2_low{0, {{{13, 0}, {0, 0}, {0, 0}}}};
  const size_t short_id_pow2_low = id_pow2_low.to_short_id();
  CHECK((short_id_pow2_low bitand 0x7FFF_st) == 8192_st);
  CHECK(((short_id_pow2_low >> 15) bitand 0x7FFF_st) == 1_st);
  CHECK((short_id_pow2_low >> 30) == 1_st);
  // 8192 + 1 * 2^15 + 1 * 2^30
  CHECK(short_id_pow2_low == 1073782784_st);

  // Upper eta (0b1001) and lower zeta (0b0110) between them carry all four
  // direction bits, and the two ids above share no set bit of the short id,
  // so a direction bit leaking into any bit of the short id changes it for
  // at least one of them (bits 2 and 3, which self 0b1100 also carries,
  // already change the value checks above)
  const auto check_direction = [](const Direction<3>& dir,
                                  const ElementId<3>& self_id) {
    const DirectionTester<3> id_with_dir{dir, self_id};
    CHECK(id_with_dir.direction() == dir);
    CHECK(id_with_dir.to_short_id() == self_id.to_short_id());
  };
  check_direction(Direction<3>::upper_eta(), id_pow2);
  check_direction(Direction<3>::lower_zeta(), id_pow2);
  check_direction(Direction<3>::upper_eta(), id_pow2_low);
  check_direction(Direction<3>::lower_zeta(), id_pow2_low);

  // Every single source bit of the raw word on its own, on the segments of
  // the two ids above in turn: each block id bit, each grid index bit, and
  // each of xi, eta and zeta replaced by level 0-14 with index 0 (one bit of
  // the compact segment id). The two bases share no set bit of the short id,
  // so a source bit copied into any other bit of the short id changes it for
  // at least one of them. Each short id is checked against the compact
  // values 2^level + index of the segments the id was built from, as
  // xi + eta * 2^15 + zeta * 2^30.
  const auto check_value = [](const size_t block_id,
                              const std::array<SegmentId, 3>& segments,
                              const size_t grid_index) {
    const auto compact = [&segments](const size_t dim) {
      return two_to_the(gsl::at(segments, dim).refinement_level()) +
             gsl::at(segments, dim).index();
    };
    const size_t swept_id =
        ElementId<3>{block_id, segments, grid_index}.to_short_id();
    CHECK((swept_id bitand 0x7FFF_st) == compact(0));
    CHECK(((swept_id >> 15) bitand 0x7FFF_st) == compact(1));
    CHECK((swept_id >> 30) == compact(2));
    CHECK(swept_id == compact(0) + (compact(1) << 15) + (compact(2) << 30));
  };
  for (const auto& base :
       make_array(std::array<SegmentId, 3>{{{14, 0}, {3, 0}, {14, 0}}},
                  std::array<SegmentId, 3>{{{13, 0}, {0, 0}, {0, 0}}})) {
    for (size_t k = 0; k < ElementId<3>::block_id_bits; ++k) {
      check_value(two_to_the(k), base, 0);
    }
    for (size_t k = 0; k < ElementId<3>::grid_index_bits; ++k) {
      check_value(0, base, two_to_the(k));
    }
    for (size_t dim = 0; dim < 3; ++dim) {
      for (size_t level = 0; level <= ElementId<3>::max_refinement_level;
           ++level) {
        std::array<SegmentId, 3> segments = base;
        gsl::at(segments, dim) = SegmentId{level, 0};
        check_value(0, segments, 0);
      }
    }
  }

  // Largest block id and grid index, every segment L14 I16383 (compact
  // 2^15 - 1): all 45 bits set, 2^45 - 1
  const SegmentId finest_last{14, 16383};
  const ElementId<3> extreme{two_to_the(ElementId<3>::block_id_bits) - 1,
                             {{finest_last, finest_last, finest_last}},
                             two_to_the(ElementId<3>::grid_index_bits) - 1};
  CHECK(extreme.to_short_id() == 35184372088831_st);

  // 1D and 2D: the unused segment fields are zero. 1D: 2^15 - 1. 2D: the xi
  // and eta above, 27307 + 13 * 2^15
  CHECK(ElementId<1>{0, {{{14, 16383}}}}.to_short_id() == 32767_st);
  CHECK(ElementId<2>{0, {{{14, 10923}, {3, 5}}}}.to_short_id() == 453291_st);
}

// Deterministic check of the extreme representable values, which the random
// test above reaches only by chance: the largest block id and grid index, and
// the finest refinement level in every dimension. The xi segment indices are
// chosen so that their bits fall into the low part, the high part, or both
// parts of the split xi bitfield.
template <size_t VolumeDim>
void test_extreme_values() {
  const size_t max_block_id =
      two_to_the(ElementId<VolumeDim>::block_id_bits) - 1;
  const size_t max_grid_index =
      two_to_the(ElementId<VolumeDim>::grid_index_bits) - 1;
  const size_t max_level = ElementId<VolumeDim>::max_refinement_level;
  const size_t xi_lo_bits = ElementId<VolumeDim>::compact_segment_id_xi_lo_bits;
  const SegmentId finest_last{max_level, two_to_the(max_level) - 1};
  for (const size_t xi_index :
       {size_t{0}, two_to_the(xi_lo_bits) - 1, two_to_the(xi_lo_bits),
        two_to_the(max_level) - 1}) {
    CAPTURE(xi_index);
    auto segment_ids = make_array<VolumeDim>(finest_last);
    gsl::at(segment_ids, 0) = SegmentId{max_level, xi_index};
    const ElementId<VolumeDim> element_id{max_block_id, segment_ids,
                                          max_grid_index};
    CHECK(element_id.block_id() == max_block_id);
    CHECK(element_id.grid_index() == max_grid_index);
    CHECK(element_id.segment_ids() == segment_ids);
    for (size_t d = 0; d < VolumeDim; ++d) {
      CHECK(gsl::at(element_id.refinement_levels(), d) == max_level);
    }
    CHECK(element_id.id_of_parent(0).segment_id(0) ==
          SegmentId{max_level - 1, xi_index / 2});
    CHECK(serialize_and_deserialize(element_id) == element_id);
    CHECK(ElementId<VolumeDim>(get_output(element_id)) == element_id);
  }
}

// The carry across the split of the xi bitfield on the write path of
// id_of_child (test_extreme_values covers only id_of_parent). The xi segment
// on level 12 with index 2^12 - 1 has the compact value 2^13 - 1, entirely in
// the low part; its upper child, level 13 with index 2^13 - 1, has the compact
// value 2^14 - 1, whose leading bit is the lowest bit of the high part. The
// other segments are on the finest level with the largest index.
template <size_t VolumeDim>
void test_child_across_xi_split() {
  const size_t max_block_id =
      two_to_the(ElementId<VolumeDim>::block_id_bits) - 1;
  const size_t max_grid_index =
      two_to_the(ElementId<VolumeDim>::grid_index_bits) - 1;
  const size_t max_level = ElementId<VolumeDim>::max_refinement_level;
  const size_t xi_lo_bits = ElementId<VolumeDim>::compact_segment_id_xi_lo_bits;
  const SegmentId finest_last{max_level, two_to_the(max_level) - 1};
  auto segment_ids = make_array<VolumeDim>(finest_last);
  gsl::at(segment_ids, 0) =
      SegmentId{xi_lo_bits - 1, two_to_the(xi_lo_bits - 1) - 1};
  const ElementId<VolumeDim> parent{max_block_id, segment_ids, max_grid_index};
  for (const auto& [side, child_xi_index] :
       {std::pair{Side::Lower, two_to_the(xi_lo_bits) - 2},
        std::pair{Side::Upper, two_to_the(xi_lo_bits) - 1}}) {
    CAPTURE(side);
    const ElementId<VolumeDim> child = parent.id_of_child(0, side);
    auto expected_segment_ids = segment_ids;
    gsl::at(expected_segment_ids, 0) = SegmentId{xi_lo_bits, child_xi_index};
    CHECK(child.segment_ids() == expected_segment_ids);
    CHECK(child.block_id() == max_block_id);
    CHECK(child.grid_index() == max_grid_index);
    CHECK(child.id_of_parent(0) == parent);
  }
  if constexpr (VolumeDim == 1) {
    // Catches a carry that is set whatever the top bit of the low part (the
    // parent above has the low part all ones). Parent 01|0...0 (level 13,
    // index 0) has carry 0: children 10|0...0 and 10|0...01
    const ElementId<1> zero_lo_parent{0, {{{xi_lo_bits, 0}}}};
    CHECK(zero_lo_parent.id_of_child(0, Side::Lower).segment_id(0) ==
          SegmentId{xi_lo_bits + 1, 0});
    CHECK(zero_lo_parent.id_of_child(0, Side::Upper).segment_id(0) ==
          SegmentId{xi_lo_bits + 1, 1});
    // Catches a carry from `lo != 0` or `lo & 1` instead of the top bit of
    // the low part. Parent 01|0...01 (level 13, index 1) has carry 0:
    // children 10|0...010 and 10|0...011 (either mutant gives 11|0...01x)
    const ElementId<1> low_bit_parent{0, {{{xi_lo_bits, 1}}}};
    CHECK(low_bit_parent.id_of_child(0, Side::Lower).segment_id(0) ==
          SegmentId{xi_lo_bits + 1, 2});
    CHECK(low_bit_parent.id_of_child(0, Side::Upper).segment_id(0) ==
          SegmentId{xi_lo_bits + 1, 3});
  }
}

// Sites that read the split xi field, with xi bits in the high part. Compact
// xi values are written as high part | low part (2 | 13 bits).
void test_xi_high_part_sites() {
  static_assert(ElementId<1>::compact_segment_id_xi_lo_bits == 13 and
                ElementId<1>::max_refinement_level == 14);
  using Hash = std::hash<ElementId<1>>;
  const ElementId<1> l13_first{0, {{{13, 0}}}};     // 01|0...0
  const ElementId<1> l13_last{0, {{{13, 8191}}}};   // 01|1...1
  const ElementId<1> l14_first{0, {{{14, 0}}}};     // 10|0...0
  const ElementId<1> l14_last{0, {{{14, 16383}}}};  // 11|1...1

  // (a) operator< compares the high part first: 01|1...1 < 10|0...0
  CHECK(l13_last < l14_first);
  CHECK_FALSE(l14_first < l13_last);
  // Catches comparing only the high part. Equal high parts, 01|0...0 <
  // 01|1...1, are ordered by the low part
  CHECK(l13_first < l13_last);
  CHECK_FALSE(l13_last < l13_first);

  // (b) number_of_block_boundaries and is_zeroth_element through
  // compact_segment_id_xi(): 10|0...0 is on the lower and 11|1...1 on the
  // upper boundary; 10|1...1 (low part all ones) is on neither; 11|0...0
  // (low part zero) is not the zeroth element
  CHECK(l14_first.number_of_block_boundaries() == 1);
  CHECK(l14_last.number_of_block_boundaries() == 1);
  CHECK(ElementId<1>{0, {{{14, 8191}}}}.number_of_block_boundaries() == 0);
  CHECK(is_zeroth_element(l14_first));
  CHECK_FALSE(is_zeroth_element(ElementId<1>{0, {{{14, 8192}}}}));

  // (c) operator== and hash_value for ids that differ only in the high part,
  // 01|0...0 and 10|0...0; the direction-copy constructor at 10|0...0
  CHECK(l13_first != l14_first);
  CHECK(Hash{}(l13_first) != Hash{}(l14_first));
  // Catches dropping one of the two high bits from operator== or the hash.
  // 01|0...0 vs 11|0...0 differ only in the upper high bit, 10|0...0 vs
  // 11|0...0 only in the lower one
  const ElementId<1> l14_high_first{0, {{{14, 8192}}}};  // 11|0...0
  CHECK(l13_first != l14_high_first);
  CHECK(Hash{}(l13_first) != Hash{}(l14_high_first));
  CHECK(l14_first != l14_high_first);
  CHECK(Hash{}(l14_first) != Hash{}(l14_high_first));
  const DirectionTester<1> l14_first_upper{Direction<1>::upper_xi(), l14_first};
  CHECK(l14_first_upper.direction() == Direction<1>::upper_xi());
  CHECK(static_cast<const ElementId<1>&>(l14_first_upper) == l14_first);
  CHECK(Hash{}(l14_first_upper) == Hash{}(l14_first));

  // (d) id_of_child reading a nonzero high part, 01|1...1 to 11|1...1 and
  // 11|1...0; PUP and string round trips with high part 01
  CHECK(l13_last.id_of_child(0, Side::Upper).segment_id(0) ==
        SegmentId{14, 16383});
  CHECK(l13_last.id_of_child(0, Side::Lower).segment_id(0) ==
        SegmentId{14, 16382});
  // Catches omitting the high-part read in 2D and 3D only. In 1D the checks
  // above start from high part 01; test_child_across_xi_split starts from
  // 00 in its 2D and 3D cases
  CHECK(ElementId<2>{0, {{{13, 8191}, {1, 1}}}}
            .id_of_child(0, Side::Upper)
            .segment_ids() == std::array<SegmentId, 2>{{{14, 16383}, {1, 1}}});
  CHECK(ElementId<3>{0, {{{13, 8191}, {1, 1}, {2, 3}}}}
            .id_of_child(0, Side::Upper)
            .segment_ids() ==
        std::array<SegmentId, 3>{{{14, 16383}, {1, 1}, {2, 3}}});
  CHECK(serialize_and_deserialize(l13_last) == l13_last);
  CHECK(ElementId<1>(get_output(l13_last)) == l13_last);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.ElementId", "[Domain][Unit]") {
  test_element_id();
  test_placement_new_and_hashing();
  test_serialization<1>();
  test_serialization<2>();
  test_serialization<3>();
  test_to_short_id();
  test_extreme_values<1>();
  test_extreme_values<2>();
  test_extreme_values<3>();
  test_child_across_xi_split<1>();
  test_child_across_xi_split<2>();
  test_child_across_xi_split<3>();
  test_xi_high_part_sites();
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ElementId<1>(two_to_the(ElementId<1>::block_id_bits)),
      Catch::Matchers::ContainsSubstring("Block id out of bounds"));
  CHECK_THROWS_WITH(
      ElementId<1>(0, {{{0, 0}}}, two_to_the(ElementId<1>::grid_index_bits)),
      Catch::Matchers::ContainsSubstring("Grid index out of bounds"));
  // The string constructor checks the parsed block id and grid index before
  // storing them; checked after, the truncated field passed the check and
  // [B2048,(L0I0)] became block 0
  static_assert(ElementId<1>::block_id_bits == 11 and
                ElementId<1>::grid_index_bits == 4);
  CHECK_THROWS_WITH(
      ElementId<1>(std::string{"[B2048,(L0I0)]"}),
      Catch::Matchers::ContainsSubstring("Block id out of bounds"));
  CHECK_THROWS_WITH(
      ElementId<1>(std::string{"[B0,(L0I0),G16]"}),
      Catch::Matchers::ContainsSubstring("Grid index out of bounds"));
  // A refinement level beyond the maximum would be silently truncated by the
  // compact segment ID bitfields, so it must be rejected in every dimension.
  const size_t too_fine = ElementId<3>::max_refinement_level + 1;
  CHECK_THROWS_WITH(
      ElementId<1>(0, {{{too_fine, 0}}}, 0),
      Catch::Matchers::ContainsSubstring("Refinement level out of bounds"));
  CHECK_THROWS_WITH(
      ElementId<2>(0, {{{0, 0}, {too_fine, 0}}}, 0),
      Catch::Matchers::ContainsSubstring("Refinement level out of bounds"));
  CHECK_THROWS_WITH(
      ElementId<3>(0, {{{0, 0}, {0, 0}, {too_fine, 0}}}, 0),
      Catch::Matchers::ContainsSubstring("Refinement level out of bounds"));
#endif
}
