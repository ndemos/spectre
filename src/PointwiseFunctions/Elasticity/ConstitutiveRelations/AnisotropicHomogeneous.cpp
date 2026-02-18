// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/AnisotropicHomogeneous.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::ConstitutiveRelations {

namespace {
// Voigt ordering used here:
// 0: xx
// 1: yy
// 2: zz
// 3: yz
// 4: xz
// 5: xy
inline std::array<DataVector, 6> strain_to_voigt(
    const tnsr::ii<DataVector, 3>& strain) {
  return {get<0, 0>(strain),
          get<1, 1>(strain),
          get<2, 2>(strain),
          2.0 * get<1, 2>(strain),
          2.0 * get<0, 2>(strain),
          2.0 * get<0, 1>(strain)};
}

inline void voigt_to_stress(
    const gsl::not_null<tnsr::II<DataVector, 3>*> stress,
    const std::array<DataVector, 6>& s) {
  get<0, 0>(*stress) = s[0];
  get<1, 1>(*stress) = s[1];
  get<2, 2>(*stress) = s[2];

  // tnsr::II stores symmetric components; set only lower-tri
  stress->get(2, 1) = s[3];  // yz
  stress->get(2, 0) = s[4];  // xz
  stress->get(1, 0) = s[5];  // xy
}
}  // namespace

AnisotropicHomogeneous::AnisotropicHomogeneous(
    std::array<std::array<double, 6>, 6> stiffness)
    : stiffness_(std::move(stiffness)) {
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      ASSERT(stiffness_[i][j] == stiffness_[j][i],
             "StiffnessMatrix must be symmetric in Voigt notation, but "
                 << "C[" << i << "][" << j << "]=" << stiffness_[i][j]
                 << " and C[" << j << "][" << i << "]=" << stiffness_[j][i]);
    }
  }
}

std::unique_ptr<ConstitutiveRelation<3>> AnisotropicHomogeneous::get_clone()
    const {
  return std::make_unique<AnisotropicHomogeneous>(*this);
}

void AnisotropicHomogeneous::stress(
    const gsl::not_null<tnsr::II<DataVector, 3>*> stress,
    const tnsr::ii<DataVector, 3>& strain,
    const tnsr::I<DataVector, 3>& /*x*/) const {
  const auto e = strain_to_voigt(strain);

  std::array<DataVector, 6> s{
      make_with_value<DataVector>(strain, 0.0),
      make_with_value<DataVector>(strain, 0.0),
      make_with_value<DataVector>(strain, 0.0),
      make_with_value<DataVector>(strain, 0.0),
      make_with_value<DataVector>(strain, 0.0),
      make_with_value<DataVector>(strain, 0.0)};

  for (size_t a = 0; a < 6; ++a) {
    for (size_t b = 0; b < 6; ++b) {
      s[a] += stiffness_[a][b] * e[b];
    }
    s[a] *= -1.0;  // T = -C e
  }

  voigt_to_stress(stress, s);
}

const std::array<std::array<double, 6>, 6>&
AnisotropicHomogeneous::stiffness_matrix() const {
  return stiffness_;
}

PUP::able::PUP_ID AnisotropicHomogeneous::my_PUP_ID = 0;

void AnisotropicHomogeneous::pup(PUP::er& p) { p | stiffness_; }

bool operator==(const AnisotropicHomogeneous& lhs,
                const AnisotropicHomogeneous& rhs) {
  return lhs.stiffness_matrix() == rhs.stiffness_matrix();
}

bool operator!=(const AnisotropicHomogeneous& lhs,
                const AnisotropicHomogeneous& rhs) {
  return not(lhs == rhs);
}

}  // namespace Elasticity::ConstitutiveRelations
