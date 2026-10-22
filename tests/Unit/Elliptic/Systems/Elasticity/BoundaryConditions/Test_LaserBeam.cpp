// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Elasticity/BoundaryConditions/LaserBeam.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity::BoundaryConditions {

namespace {
std::vector<LaserBeam::ModeInfo> tem00_modes() { return {{0, 0, 1.}}; }

// Must match `HG_MODES` in LaserBeam.py
std::vector<LaserBeam::ModeInfo> hg_modes() {
  return {{0, 2, 1.}, {2, 0, -1.}, {1, 3, 0.5}};
}

// The boundary condition assumes a flat surface, so the (random) unit normal
// is the same at all points. It is parametrized by the angles theta and phi.
template <bool Linearized, bool Tem00>
void apply_boundary_condition(
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
    const tnsr::I<DataVector, 3>& x, const double theta, const double phi,
    const double beam_width) {
  const LaserBeam laser_beam{beam_width, Tem00 ? tem00_modes() : hg_modes()};
  const auto direction = Direction<3>::lower_xi();
  auto face_normal = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
  get<0>(face_normal) = sin(theta) * cos(phi);
  get<1>(face_normal) = sin(theta) * sin(phi);
  get<2>(face_normal) = cos(theta);
  const auto box = db::create<db::AddSimpleTags<
      domain::Tags::Faces<3, domain::Tags::Coordinates<3, Frame::Inertial>>,
      domain::Tags::Faces<3, domain::Tags::FaceNormal<3>>>>(
      DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, x}},
      DirectionMap<3, tnsr::i<DataVector, 3>>{{direction, face_normal}});
  tnsr::I<DataVector, 3> displacement{
      x.begin()->size(), std::numeric_limits<double>::signaling_NaN()};
  tnsr::iJ<DataVector, 3> deriv_displacement{
      x.begin()->size(), std::numeric_limits<double>::signaling_NaN()};
  elliptic::apply_boundary_condition<Linearized, void, tmpl::list<LaserBeam>>(
      laser_beam, box, direction, make_not_null(&displacement),
      n_dot_minus_stress, deriv_displacement);
}

// Pressure on the z = const surface with outward normal (0, 0, -1)
DataVector pressure_on_z_surface(const LaserBeam& laser_beam,
                                 const tnsr::I<DataVector, 3>& x) {
  auto face_normal = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
  get<2>(face_normal) = -1.;
  tnsr::I<DataVector, 3> displacement{
      get<0>(x).size(), std::numeric_limits<double>::signaling_NaN()};
  tnsr::I<DataVector, 3> n_dot_minus_stress{
      get<0>(x).size(), std::numeric_limits<double>::signaling_NaN()};
  const tnsr::iJ<DataVector, 3> deriv_displacement{
      get<0>(x).size(), std::numeric_limits<double>::signaling_NaN()};
  laser_beam.apply(make_not_null(&displacement),
                   make_not_null(&n_dot_minus_stress), deriv_displacement, x,
                   face_normal);
  CHECK(get<0>(n_dot_minus_stress) == DataVector(get<0>(x).size(), 0.));
  CHECK(get<1>(n_dot_minus_stress) == DataVector(get<0>(x).size(), 0.));
  return get<2>(n_dot_minus_stress);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.BoundaryConditions.LaserBeam",
                  "[Unit][Elliptic]") {
  const double beam_width = 2.;
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>, LaserBeam>(
      "LaserBeam:\n"
      "  BeamWidth: 2.\n"
      "  Modes:\n"
      "    - N: 0\n"
      "      M: 0\n"
      "      Weight: 1.");
  REQUIRE(dynamic_cast<const LaserBeam*>(created.get()) != nullptr);
  const auto& laser_beam = dynamic_cast<const LaserBeam&>(*created);
  const auto created_hg = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>, LaserBeam>(
      "LaserBeam:\n"
      "  BeamWidth: 2.\n"
      "  Modes:\n"
      "    - N: 0\n"
      "      M: 2\n"
      "      Weight: 1.\n"
      "    - N: 2\n"
      "      M: 0\n"
      "      Weight: -1.\n"
      "    - N: 1\n"
      "      M: 3\n"
      "      Weight: 0.5");
  REQUIRE(dynamic_cast<const LaserBeam*>(created_hg.get()) != nullptr);
  const auto& laser_beam_hg = dynamic_cast<const LaserBeam&>(*created_hg);

  {
    INFO("Semantics");
    test_serialization(laser_beam);
    test_copy_semantics(laser_beam);
    auto move_laser_beam = laser_beam;
    test_move_semantics(std::move(move_laser_beam), laser_beam);
    test_serialization(laser_beam_hg);
    CHECK(laser_beam != laser_beam_hg);
    CHECK(laser_beam == LaserBeam{beam_width, tem00_modes()});
    CHECK(laser_beam_hg == LaserBeam{beam_width, hg_modes()});
    CHECK(laser_beam != LaserBeam{3., tem00_modes()});
  }
  {
    INFO("Properties");
    CHECK(laser_beam.beam_width() == beam_width);
    CHECK(laser_beam.modes() == tem00_modes());
    CHECK(laser_beam_hg.modes() == hg_modes());
    CHECK(laser_beam.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              3, elliptic::BoundaryConditionType::Neumann});
  }

  // Test applying the boundary conditions. The bounds are for x, the polar
  // and azimuthal angles of the face normal, and the beam width.
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Elliptic/Systems/Elasticity/BoundaryConditions/");
  const std::array<std::pair<double, double>, 4> bounds{
      {{-2., 2.}, {0., M_PI}, {-M_PI, M_PI}, {0.5, 2.}}};
  pypp::check_with_random_values<4>(&apply_boundary_condition<false, true>,
                                    "LaserBeam", {"normal_dot_minus_stress"},
                                    bounds, DataVector{3});
  pypp::check_with_random_values<4>(
      &apply_boundary_condition<false, false>, "LaserBeam",
      {"normal_dot_minus_stress_hermite_gauss"}, bounds, DataVector{3});
  pypp::check_with_random_values<4>(
      &apply_boundary_condition<true, true>, "LaserBeam",
      {"normal_dot_minus_stress_linearized"}, bounds, DataVector{3});
  pypp::check_with_random_values<4>(
      &apply_boundary_condition<true, false>, "LaserBeam",
      {"normal_dot_minus_stress_linearized"}, bounds, DataVector{3});

  {
    INFO("Mode normalization, symmetry and orientation");
    // Uniform grid on the z = 0 plane, wide enough that the profiles vanish
    // at the edges. The trapezoidal rule is then spectrally accurate.
    const double r0 = 1.;
    const size_t num_points_1d = 481;
    const double half_extent = 12. * r0;
    const double spacing =
        2. * half_extent / static_cast<double>(num_points_1d - 1);
    const size_t num_points = square(num_points_1d);
    tnsr::I<DataVector, 3> x{num_points, 0.};
    tnsr::I<DataVector, 3> x_swapped{num_points, 0.};
    for (size_t i = 0; i < num_points_1d; ++i) {
      for (size_t j = 0; j < num_points_1d; ++j) {
        const size_t k = i + num_points_1d * j;
        get<0>(x)[k] = -half_extent + static_cast<double>(i) * spacing;
        get<1>(x)[k] = -half_extent + static_cast<double>(j) * spacing;
        get<0>(x_swapped)[k] = get<1>(x)[k];
        get<1>(x_swapped)[k] = get<0>(x)[k];
      }
    }
    const auto total_force =
        [&x, &r0, &spacing](const std::vector<LaserBeam::ModeInfo>& modes) {
          return sum(pressure_on_z_surface(LaserBeam{r0, modes}, x)) *
                 square(spacing);
        };
    Approx custom_approx = Approx::custom().epsilon(1.e-11).scale(1.);
    // Each mode with unit weight exerts unit total force
    for (const auto& [n, m] : std::vector<std::pair<size_t, size_t>>{
             {0, 0}, {0, 2}, {2, 0}, {1, 3}, {3, 3}, {4, 1}}) {
      CAPTURE(n);
      CAPTURE(m);
      CHECK(total_force({{n, m, 1.}}) == custom_approx(1.));
    }
    CHECK(total_force({{0, 0, 2.5}}) == custom_approx(2.5));
    CHECK(total_force({{0, 2, 1.}, {2, 0, -1.}}) == custom_approx(0.));
    CHECK(total_force(hg_modes()) == custom_approx(0.5));
    // TEM_nm(x, y) = TEM_mn(y, x)
    CHECK_ITERABLE_APPROX(
        pressure_on_z_surface(LaserBeam{r0, {{0, 2, 1.}}}, x),
        pressure_on_z_surface(LaserBeam{r0, {{2, 0, 1.}}}, x_swapped));
    // For a surface normal to z, N is the Hermite index along x. H_2 has its
    // zeros at +-1/sqrt(2), so TEM20 vanishes at x = r_0 / sqrt(2), y = 0
    // but not at x = 0, y = r_0 / sqrt(2).
    tnsr::I<DataVector, 3> x_nodes{size_t{2}, 0.};
    get<0>(x_nodes)[0] = r0 / sqrt(2.);
    get<1>(x_nodes)[1] = r0 / sqrt(2.);
    const auto pressure_tem20 =
        pressure_on_z_surface(LaserBeam{r0, {{2, 0, 1.}}}, x_nodes);
    CHECK(pressure_tem20[0] == approx(0.));
    // H_2(0)^2 e^{-1/2} / (pi r_0^2 2^2 2!) with r_0 = 1
    CHECK(pressure_tem20[1] == approx(exp(-0.5) / (2. * M_PI)));
  }

  {
    INFO("Consistency with half-space mirror solution");
    const DataVector used_for_size{5};
    // Choose an arbitrary set of points on the z=0 surface of the mirror
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> dist(-2. * beam_width, 2. * beam_width);
    auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
        make_not_null(&generator), make_not_null(&dist), used_for_size);
    get<2>(x) = 0.;
    // Choose a constitutive relation with arbitrary parameters
    const ConstitutiveRelations::IsotropicHomogeneous<3> constitutive_relation{
        1., 2.};
    // Get the surface stress from the half-space mirror solution.
    // Disable relative tolerance because we compare with an absolute tolerance
    // in the test below.
    const Solutions::HalfSpaceMirror half_space_mirror{
        beam_width, constitutive_relation, 350, 1e-12, 0.};
    const auto minus_stress_solution = get<Tags::MinusStress<3>>(
        half_space_mirror.variables(x, tmpl::list<Tags::MinusStress<3>>{}));
    // Get the stress normal to the surface. The solution assumes the material
    // extends in positive z-direction, so the normal is (0, 0, -1)
    tnsr::I<DataVector, 3> n_dot_minus_stress_solution{used_for_size.size()};
    get<0>(n_dot_minus_stress_solution) = -get<2, 0>(minus_stress_solution);
    get<1>(n_dot_minus_stress_solution) = -get<2, 1>(minus_stress_solution);
    get<2>(n_dot_minus_stress_solution) = -get<2, 2>(minus_stress_solution);
    auto face_normal = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
    get<2>(face_normal) = -1;
    // Shift the plane where we evaluate the boundary condition along the
    // z-direction, just because it shouldn't affect the result and it might
    // catch issues with computing the coordinate distance
    get<2>(x) += 2.;
    // Compare to the boundary conditions
    tnsr::I<DataVector, 3> displacement{
        used_for_size.size(), std::numeric_limits<double>::signaling_NaN()};
    tnsr::I<DataVector, 3> n_dot_minus_stress{
        used_for_size.size(), std::numeric_limits<double>::signaling_NaN()};
    const tnsr::iJ<DataVector, 3> deriv_displacement{
        used_for_size.size(), std::numeric_limits<double>::signaling_NaN()};
    laser_beam.apply(make_not_null(&displacement),
                     make_not_null(&n_dot_minus_stress), deriv_displacement, x,
                     face_normal);
    for (size_t d = 0; d < 3; ++d) {
      CAPTURE(d);
      CHECK_ITERABLE_APPROX(n_dot_minus_stress.get(d),
                            n_dot_minus_stress_solution.get(d));
    }
  }
}

}  // namespace Elasticity::BoundaryConditions
