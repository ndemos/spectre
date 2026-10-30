// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/rational.hpp>
#include <cstddef>
#include <string>
#include <vector>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t VolumeDim>
void test_initial_element_ids(
    const std::vector<ElementId<VolumeDim>>& element_ids,
    const std::vector<std::array<size_t, VolumeDim>>& initial_refinement_levels,
    size_t grid_index) {
  size_t expected_number_of_elements = 0;
  for (const auto& initial_refinement_levels_of_block :
       initial_refinement_levels) {
    size_t expected_number_of_elements_in_block = 1;
    for (size_t d = 0; d < VolumeDim; ++d) {
      expected_number_of_elements_in_block *=
          two_to_the(gsl::at(initial_refinement_levels_of_block, d));
    }
    expected_number_of_elements += expected_number_of_elements_in_block;
  }
  CHECK(expected_number_of_elements == element_ids.size());
  const boost::rational<size_t> expected_logical_volume_of_blocks(
      initial_refinement_levels.size());
  boost::rational<size_t> logical_volume_of_blocks(0);
  for (const auto& element_id : element_ids) {
    logical_volume_of_blocks += fraction_of_block_volume(element_id);
    CHECK(element_id.grid_index() == grid_index);
  }
  CHECK(expected_logical_volume_of_blocks == logical_volume_of_blocks);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.InitialElementIds", "[Domain][Unit]") {
  const std::vector<std::array<size_t, 1>> initial_refinement_levels_1d{{{2}},
                                                                        {{3}}};
  const auto element_ids_1d = initial_element_ids(initial_refinement_levels_1d);
  test_initial_element_ids(element_ids_1d, initial_refinement_levels_1d, 0);
  test_initial_element_ids(initial_element_ids(initial_refinement_levels_1d, 3),
                           initial_refinement_levels_1d, 3);

  const std::vector<std::array<size_t, 2>> initial_refinement_levels_2d{
      {{2, 0}}, {{3, 1}}};
  const auto element_ids_2d = initial_element_ids(initial_refinement_levels_2d);
  test_initial_element_ids(element_ids_2d, initial_refinement_levels_2d, 0);
  test_initial_element_ids(initial_element_ids(initial_refinement_levels_2d, 3),
                           initial_refinement_levels_2d, 3);

  const std::vector<std::array<size_t, 3>> initial_refinement_levels_3d{
      {{4, 2, 1}}, {{0, 3, 2}}};
  const auto element_ids_3d = initial_element_ids(initial_refinement_levels_3d);
  test_initial_element_ids(element_ids_3d, initial_refinement_levels_3d, 0);
  test_initial_element_ids(initial_element_ids(initial_refinement_levels_3d, 3),
                           initial_refinement_levels_3d, 3);

  {
    INFO("Refinement level limit (checked in all build types)");
    const size_t max_level = ElementId<1>::max_refinement_level;
    // The maximum level is representable without truncation
    const auto finest_ids = initial_element_ids<1>(0, {{max_level}});
    REQUIRE(finest_ids.size() == two_to_the(max_level));
    CHECK(finest_ids.back().segment_id(0) ==
          SegmentId(max_level, two_to_the(max_level) - 1));
    // One level more is rejected, in every dimension and through both
    // overloads
    const std::string expected_message =
        "exceeds the maximum refinement level " + std::to_string(max_level);
    CHECK_THROWS_WITH(initial_element_ids<1>(0, {{max_level + 1}}),
                      Catch::Matchers::ContainsSubstring(expected_message));
    CHECK_THROWS_WITH(initial_element_ids<2>(0, {{0, max_level + 1}}),
                      Catch::Matchers::ContainsSubstring(expected_message));
    CHECK_THROWS_WITH(initial_element_ids<3>(0, {{0, 0, max_level + 1}}),
                      Catch::Matchers::ContainsSubstring(expected_message));
    CHECK_THROWS_WITH(initial_element_ids(std::vector<std::array<size_t, 3>>{
                          {{1, 1, 1}}, {{max_level + 1, 0, 0}}}),
                      Catch::Matchers::ContainsSubstring("of block 1 exceeds"));
  }

  {
    INFO("Block count limit (checked in all build types)");
    const size_t max_blocks = two_to_the(ElementId<3>::block_id_bits);
    // The largest block id is usable: external_boundary_id() also has it, but
    // no element compares equal to that id
    const auto last_block_ids =
        initial_element_ids<3>(max_blocks - 1, {{0, 0, 0}});
    REQUIRE(last_block_ids.size() == 1);
    CHECK(last_block_ids[0].block_id() == max_blocks - 1);
    CHECK(last_block_ids[0] != ElementId<3>::external_boundary_id());
    // The largest allowed number of blocks
    const std::vector<std::array<size_t, 3>> max_domain(max_blocks,
                                                        {{0, 0, 0}});
    const auto max_domain_ids = initial_element_ids(max_domain);
    REQUIRE(max_domain_ids.size() == max_blocks);
    CHECK(max_domain_ids.back().block_id() == max_blocks - 1);
    // One block more is rejected, in every dimension and through both
    // overloads
    const std::string expected_message =
        "Block id " + std::to_string(max_blocks) +
        " exceeds the maximum block id " + std::to_string(max_blocks - 1);
    CHECK_THROWS_WITH(initial_element_ids<1>(max_blocks, {{0}}),
                      Catch::Matchers::ContainsSubstring(expected_message));
    CHECK_THROWS_WITH(initial_element_ids<2>(max_blocks, {{0, 0}}),
                      Catch::Matchers::ContainsSubstring(expected_message));
    CHECK_THROWS_WITH(initial_element_ids<3>(max_blocks, {{0, 0, 0}}),
                      Catch::Matchers::ContainsSubstring(expected_message));
    CHECK_THROWS_WITH(initial_element_ids(std::vector<std::array<size_t, 3>>(
                          max_blocks + 1, {{0, 0, 0}})),
                      Catch::Matchers::ContainsSubstring(expected_message));
    // Catches rejecting only the first id past the limit
    CHECK_THROWS_WITH(initial_element_ids<3>(5000, {{0, 0, 0}}),
                      Catch::Matchers::ContainsSubstring(
                          "Block id 5000 exceeds the maximum block id " +
                          std::to_string(max_blocks - 1)));
  }
}
