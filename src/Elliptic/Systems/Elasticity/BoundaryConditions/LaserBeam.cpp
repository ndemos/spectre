// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/BoundaryConditions/LaserBeam.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::BoundaryConditions {

namespace {
// Recursive calculation of Hermite polynomials H_n(x)
DataVector hermite_polynomial(const size_t n, const DataVector& x) {
  if (n == 0) {
    return make_with_value<DataVector>(x, 1.0);
  }
  if (n == 1) {
    return 2.0 * x;
  }
  DataVector h_prev = make_with_value<DataVector>(x, 1.0);  // H_0
  DataVector h_curr = 2.0 * x;                              // H_1
  DataVector h_next = make_with_value<DataVector>(x, 0.0);

  for (size_t i = 1; i < n; ++i) {
    h_next = 2.0 * x * h_curr - 2.0 * static_cast<double>(i) * h_prev;
    h_prev = h_curr;
    h_curr = h_next;
  }
  return h_curr;
}

double factorial(const size_t n) {
  return (n == 1 || n == 0) ? 1.0 : n * factorial(n - 1);
}
}  // namespace

LaserBeam::LaserBeam(double beam_width, const std::vector<ModeInfo>& modes)
    : beam_width_(beam_width), modes_(modes) {}

void LaserBeam::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> /*displacement*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
    const tnsr::iJ<DataVector, 3>& /*deriv_displacement*/,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& face_normal) const {
#ifdef SPECTRE_DEBUG
  ASSERT(get<2>(face_normal)[0] ==
             get<2>(face_normal)[face_normal.begin()->size() - 1],
         "LaserBeam currently assumes a flat surface (constant normal).");
#endif

  // 1. Calculate vector r
  const auto n_dot_x = get<0>(face_normal) * get<0>(x) +
                       get<1>(face_normal) * get<1>(x) +
                       get<2>(face_normal) * get<2>(x);

  const DataVector r_vec_0 = get<0>(x) - n_dot_x * get<0>(face_normal);
  const DataVector r_vec_1 = get<1>(x) - n_dot_x * get<1>(face_normal);
  const DataVector r_vec_2 = get<2>(x) - n_dot_x * get<2>(face_normal);

  // 2. Define a local basis (u, v)
  const double nx_ref = get<0>(face_normal)[0];

  double tx = 1.0, ty = 0.0, tz = 0.0;
  if (std::abs(nx_ref) > 0.9) {
    tx = 0.0;
    ty = 1.0;
  }

  const auto t_dot_n = tx * get<0>(face_normal) + ty * get<1>(face_normal) +
                       tz * get<2>(face_normal);

  DataVector u_vec_0 = tx - t_dot_n * get<0>(face_normal);
  DataVector u_vec_1 = ty - t_dot_n * get<1>(face_normal);
  DataVector u_vec_2 = tz - t_dot_n * get<2>(face_normal);

  // Normalize basis_u
  const DataVector u_mag =
      sqrt(square(u_vec_0) + square(u_vec_1) + square(u_vec_2));

  for (size_t i = 0; i < u_mag.size(); ++i) {
    u_vec_0[i] /= u_mag[i];
    u_vec_1[i] /= u_mag[i];
    u_vec_2[i] /= u_mag[i];
  }

  // basis_v = n x basis_u
  const auto v_vec_0 =
      get<1>(face_normal) * u_vec_2 - get<2>(face_normal) * u_vec_1;
  const auto v_vec_1 =
      get<2>(face_normal) * u_vec_0 - get<0>(face_normal) * u_vec_2;
  const auto v_vec_2 =
      get<0>(face_normal) * u_vec_1 - get<1>(face_normal) * u_vec_0;

  // 3. Project r vector
  const DataVector coord_u =
      r_vec_0 * u_vec_0 + r_vec_1 * u_vec_1 + r_vec_2 * u_vec_2;
  const DataVector coord_v =
      r_vec_0 * v_vec_0 + r_vec_1 * v_vec_1 + r_vec_2 * v_vec_2;

  // 4. Sum Intensities (Incoherent Summation)
  // Scaling factor for Hermite polynomials: sqrt(2)*r/w
  const double sqrt2_over_w = sqrt(2.0) / beam_width_;

  // Accumulator for the final intensity profile (P_total)
  DataVector summed_intensity = make_with_value<DataVector>(coord_u, 0.0);

  for (const auto& mode : modes_) {
    const DataVector H_n = hermite_polynomial(mode.n, coord_u * sqrt2_over_w);
    const DataVector H_m = hermite_polynomial(mode.m, coord_v * sqrt2_over_w);

    // Calculate the "Shape" of the intensity for this mode: (H_n * H_m)^2
    // We square it immediately because we are adding Intensities, not Fields.
    DataVector mode_shape = square(H_n * H_m);

    // Calculate Normalization Integral for this specific mode (n, m)
    // Integral[ (H_n * H_m * exp(-r^2/w^2))^2 dA ] = (pi*w^2/2) * 2^(n+m) * n!
    // * m!
    double normalization_const = (M_PI * square(beam_width_) / 2.0) *
                                 pow(2.0, mode.n + mode.m) * factorial(mode.n) *
                                 factorial(mode.m);

    // Add this mode's contribution to the total pressure
    // P_total += weight * (Shape / Normalization)
    // If weight is negative, this subtracts pressure (virtual work).
    summed_intensity += mode.weight * (mode_shape / normalization_const);
  }

  // 5. Apply Gaussian Envelope
  // The term exp(-2r^2/w^2) is common to all Hermite-Gaussian intensities
  // and was factored out of the integration constant above.
  const DataVector beam_profile =
      summed_intensity *
      exp(-2.0 * (square(coord_u) + square(coord_v)) / square(beam_width_));

  get<0>(*n_dot_minus_stress) = -beam_profile * get<0>(face_normal);
  get<1>(*n_dot_minus_stress) = -beam_profile * get<1>(face_normal);
  get<2>(*n_dot_minus_stress) = -beam_profile * get<2>(face_normal);
}

void LaserBeam::apply_linearized(
    const gsl::not_null<tnsr::I<DataVector, 3>*> /*displacement*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
    const tnsr::iJ<DataVector, 3>& /*deriv_displacement*/) {
  get<0>(*n_dot_minus_stress) = 0.;
  get<1>(*n_dot_minus_stress) = 0.;
  get<2>(*n_dot_minus_stress) = 0.;
}

// Comparison
bool operator==(const LaserBeam& lhs, const LaserBeam& rhs) {
  return lhs.beam_width() == rhs.beam_width() && lhs.modes() == rhs.modes();
}

bool operator!=(const LaserBeam& lhs, const LaserBeam& rhs) {
  return not(lhs == rhs);
}

bool operator==(const LaserBeam::ModeInfo& lhs,
                const LaserBeam::ModeInfo& rhs) {
  return lhs.n == rhs.n && lhs.m == rhs.m && lhs.weight == rhs.weight;
}
bool operator!=(const LaserBeam::ModeInfo& lhs,
                const LaserBeam::ModeInfo& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID LaserBeam::my_PUP_ID = 0;  // NOLINT

}  // namespace Elasticity::BoundaryConditions
