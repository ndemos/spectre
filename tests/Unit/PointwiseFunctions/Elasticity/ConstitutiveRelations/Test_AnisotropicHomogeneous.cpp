// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/AnisotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/CubicCrystal.hpp"
#include "Utilities/Gsl.hpp"

namespace Elasticity::ConstitutiveRelations {
namespace {

// Voigt stiffness matrix of a cubic crystal
std::array<std::array<double, 6>, 6> cubic_voigt_matrix(const double c_11,
                                                        const double c_12,
                                                        const double c_44) {
  return {{{c_11, c_12, c_12, 0., 0., 0.},
           {c_12, c_11, c_12, 0., 0., 0.},
           {c_12, c_12, c_11, 0., 0., 0.},
           {0., 0., 0., c_44, 0., 0.},
           {0., 0., 0., 0., c_44, 0.},
           {0., 0., 0., 0., 0., c_44}}};
}

void test_semantics() {
  INFO("Semantics");
  const AnisotropicHomogeneous relation{cubic_voigt_matrix(3., 2., 1.)};
  CHECK(relation == AnisotropicHomogeneous{cubic_voigt_matrix(3., 2., 1.)});
  CHECK(relation != AnisotropicHomogeneous{cubic_voigt_matrix(3., 1., 1.)});
  test_serialization(relation);
  test_copy_semantics(relation);
  const auto created_ptr =
      TestHelpers::test_factory_creation<ConstitutiveRelation<3>,
                                         AnisotropicHomogeneous>(
          "AnisotropicHomogeneous:\n"
          "  StiffnessMatrix:\n"
          "    - [3., 2., 2., 0., 0., 0.]\n"
          "    - [2., 3., 2., 0., 0., 0.]\n"
          "    - [2., 2., 3., 0., 0., 0.]\n"
          "    - [0., 0., 0., 1., 0., 0.]\n"
          "    - [0., 0., 0., 0., 1., 0.]\n"
          "    - [0., 0., 0., 0., 0., 1.]\n")
          ->get_clone();
  REQUIRE(dynamic_cast<const AnisotropicHomogeneous*>(created_ptr.get()) !=
          nullptr);
  CHECK(dynamic_cast<const AnisotropicHomogeneous&>(*created_ptr) == relation);
  AnisotropicHomogeneous moved_relation{cubic_voigt_matrix(3., 2., 1.)};
  test_move_semantics(std::move(moved_relation), relation);
}

void test_consistency_with_cubic_crystal() {
  INFO("Consistency with CubicCrystal");
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const DataVector used_for_size{10};
  const auto strain = make_with_random_values<tnsr::ii<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  // Silicon: c_11=166, c_12=64, c_44=80 (see Test_CubicCrystal.cpp)
  const AnisotropicHomogeneous anisotropic{cubic_voigt_matrix(166., 64., 80.)};
  const CubicCrystal cubic{166., 64., 80.};
  tnsr::II<DataVector, 3> anisotropic_stress{used_for_size.size()};
  tnsr::II<DataVector, 3> cubic_stress{used_for_size.size()};
  anisotropic.stress(make_not_null(&anisotropic_stress), strain, x);
  cubic.stress(make_not_null(&cubic_stress), strain, x);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CHECK_ITERABLE_APPROX(anisotropic_stress.get(i, j),
                            cubic_stress.get(i, j));
    }
  }
}

void test_errors() {
  INFO("Invalid stiffness matrices");
  auto asymmetric = cubic_voigt_matrix(3., 2., 1.);
  asymmetric[0][1] = 2.5;
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{asymmetric},
      Catch::Matchers::ContainsSubstring("must be symmetric in Voigt"));
  // The criterion is that all eigenvalues of the Voigt matrix are positive.
  // For a cubic crystal they are c_11 + 2 c_12 (hydrostatic, once),
  // c_11 - c_12 (twice) and c_44 (three times), and each case below makes
  // exactly one of them non-positive.
  // c_11 - c_12 = -1 < 0
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{cubic_voigt_matrix(2., 3., 1.)},
      Catch::Matchers::ContainsSubstring("must be positive definite"));
  // Hydrostatic instability: c_11 + 2 c_12 = -1 < 0, although
  // c_11 > |c_12|, so that condition alone is not sufficient
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{cubic_voigt_matrix(2., -1.5, 1.)},
      Catch::Matchers::ContainsSubstring("must be positive definite"));
  // c_44 = -1 < 0
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{cubic_voigt_matrix(3., 2., -1.)},
      Catch::Matchers::ContainsSubstring("must be positive definite"));
  // Positive semidefinite: c_44 = 0 gives a zero eigenvalue. The shear rows
  // of the matrix are decoupled and diagonal, so LAPACK deflates them and
  // returns this eigenvalue as exactly 0, which fails the check
  // (min eigenvalue > 0). Only the message is checked, not the value.
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{cubic_voigt_matrix(3., 2., 0.)},
      Catch::Matchers::ContainsSubstring("must be positive definite"));
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Elasticity.ConstitutiveRelations.AnisotropicHomogeneous",
    "[PointwiseFunctions][Unit][Elasticity]") {
  test_semantics();
  test_consistency_with_cubic_crystal();
  test_errors();
}

}  // namespace Elasticity::ConstitutiveRelations
