// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"
#include "PointwiseFunctions/Elasticity/Stress.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim>
Scalar<DataVector> potential_energy_density(
    // The wrapper constructs a constitutive relation for use in
    // check_with_random_values.
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& strain,
    const tnsr::I<DataVector, Dim>& coordinates, const double bulk_modulus,
    const double shear_modulus) {
  Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>
      constitutive_relation{bulk_modulus, shear_modulus};
  return Elasticity::potential_energy_density<Dim>(
      strain, coordinates, std::move(constitutive_relation));
}

template <size_t Dim>
void test_elastic_potential_energy(const DataVector& used_for_size) {
  pypp::check_with_random_values<4>(
      &potential_energy_density<Dim>, "Elasticity.PotentialEnergy",
      "potential_energy_density", {{{-1., 1.}, {0., 1.}, {0., 1.}, {0., 1.}}},
      used_for_size);
}

template <size_t Dim>
void test_compute_tags(const DataVector& used_for_size) {
  CAPTURE(Dim);
  using strain_tag = Elasticity::Tags::Strain<Dim>;
  using energy_tag = Elasticity::Tags::PotentialEnergyDensity<Dim>;
  using constitutive_relation_tag = Elasticity::Tags::ConstitutiveRelation<Dim>;
  using coordinates_tag = domain::Tags::Coordinates<Dim, Frame::Inertial>;
  TestHelpers::db::test_compute_tag<Elasticity::Tags::StressCompute<Dim>>(
      "Stress");
  TestHelpers::db::test_compute_tag<
      Elasticity::Tags::PotentialEnergyDensityCompute<Dim>>(
      "PotentialEnergyDensity");
  const size_t num_points = used_for_size.size();
  {
    INFO("Energy");
    std::unique_ptr<
        Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>>
        constitutive_relation = std::make_unique<
            Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>(3.,
                                                                          4.);
    const auto box =
        db::create<db::AddSimpleTags<strain_tag, constitutive_relation_tag,
                                     coordinates_tag>,
                   db::AddComputeTags<
                       Elasticity::Tags::StressCompute<Dim>,
                       Elasticity::Tags::PotentialEnergyDensityCompute<Dim>>>(
            tnsr::ii<DataVector, Dim>{num_points, 1.},
            std::move(constitutive_relation),
            tnsr::I<DataVector, Dim>{num_points, 2.});

    const auto expected_energy = Elasticity::potential_energy_density<Dim>(
        get<strain_tag>(box), get<coordinates_tag>(box),
        get<constitutive_relation_tag>(box));
    CHECK(get<energy_tag>(box) == expected_energy);
  }
}

template <size_t Dim>
void test_bulk_shear_split(const DataVector& used_for_size) {
  CAPTURE(Dim);
  TestHelpers::db::test_compute_tag<
      Elasticity::Tags::BulkPotentialEnergyDensityCompute<Dim>>(
      "BulkPotentialEnergyDensity");
  TestHelpers::db::test_compute_tag<
      Elasticity::Tags::ShearPotentialEnergyDensityCompute<Dim>>(
      "ShearPotentialEnergyDensity");
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const auto strain = make_with_random_values<tnsr::ii<DataVector, Dim>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto x = make_with_random_values<tnsr::I<DataVector, Dim>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const double bulk_modulus = 3.;
  const double shear_modulus = 4.;
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>
      constitutive_relation{bulk_modulus, shear_modulus};
  tnsr::II<DataVector, Dim> stress{used_for_size.size()};
  constitutive_relation.stress(make_not_null(&stress), strain, x);
  Scalar<DataVector> bulk_energy{used_for_size.size()};
  Scalar<DataVector> shear_energy{used_for_size.size()};
  Elasticity::bulk_potential_energy_density(make_not_null(&bulk_energy), strain,
                                            stress);
  Elasticity::shear_potential_energy_density(make_not_null(&shear_energy),
                                             strain, stress);
  {
    INFO("Bulk + shear = total");
    const auto total_energy = Elasticity::potential_energy_density<Dim>(
        strain, x, constitutive_relation);
    const DataVector bulk_plus_shear = get(bulk_energy) + get(shear_energy);
    CHECK_ITERABLE_APPROX(bulk_plus_shear, get(total_energy));
  }
  if constexpr (Dim == 3) {
    INFO("Isotropic material: U_bulk = K Theta^2 / 2, U_shear = mu e_ij e_ij");
    DataVector trace_strain = get<0, 0>(strain);
    for (size_t d = 1; d < Dim; ++d) {
      trace_strain += strain.get(d, d);
    }
    DataVector deviatoric_square{used_for_size.size(), 0.};
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        DataVector deviatoric_strain = strain.get(i, j);
        if (i == j) {
          deviatoric_strain -= trace_strain / static_cast<double>(Dim);
        }
        deviatoric_square += square(deviatoric_strain);
      }
    }
    const DataVector expected_bulk = 0.5 * bulk_modulus * square(trace_strain);
    const DataVector expected_shear = shear_modulus * deviatoric_square;
    CHECK_ITERABLE_APPROX(get(bulk_energy), expected_bulk);
    CHECK_ITERABLE_APPROX(get(shear_energy), expected_shear);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Elasticity.PotentialEnergy",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"PointwiseFunctions"};
  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_elastic_potential_energy, (2, 3));
  CHECK_FOR_DATAVECTORS(test_compute_tags, (2, 3));
  CHECK_FOR_DATAVECTORS(test_bulk_shear_split, (2, 3));
}
