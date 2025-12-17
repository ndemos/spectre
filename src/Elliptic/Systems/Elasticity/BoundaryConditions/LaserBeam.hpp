// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Elasticity::BoundaryConditions {

/*!
 * \brief Applies a pressure profile based on an INCOHERENT sum of
 * Hermite-Gaussian modes.
 *
 * \details
 * This boundary condition calculates the intensity of various Hermite-Gaussian
 * modes individually and sums them up.
 * - Positive weights ADD pressure (pushing into the material).
 * - Negative weights SUBTRACT pressure (pulling out).
 *
 * This is used for differential thermal noise measurements (e.g. TEM02 -
 * TEM20). Note: The beam width 'w' is the 1/e^2 intensity radius.
 */
class LaserBeam : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  struct BeamWidth {
    using type = double;
    static constexpr Options::String help =
        "The width r_0 (1/e^2 intensity) of the beam.";
    // Prevents division by zero in the .cpp file
    static type lower_bound() { return 0.0; }
  };

  struct ModeInfo {
    struct N {
      using type = size_t;
      static constexpr Options::String help = "The Hermite-Gaussian index n";
    };
    struct M {
      using type = size_t;
      static constexpr Options::String help = "The Hermite-Gaussian index m";
    };
    struct Weight {
      using type = double;
      // Updated docs to reflect that this is a multiplier for Intensity, not
      // E-Field.
      static constexpr Options::String help =
          "The intensity multiplier (can be negative)";
      // CRITICAL: No validator here! Negative values allowed for subtraction.
    };

    using options = tmpl::list<N, M, Weight>;
    static constexpr Options::String help =
        "A single mode in the incoherent sum.";

    ModeInfo() = default;
    ModeInfo(size_t in_n, size_t in_m, double in_weight)
        : n(in_n), m(in_m), weight(in_weight) {}

    size_t n{0};
    size_t m{0};
    double weight{0.0};

    // Serialization for parallel distribution
    void pup(PUP::er& p) {
      p | n;
      p | m;
      p | weight;
    }
  };

  struct PolynomialModes {
    using type = std::vector<ModeInfo>;
    static constexpr Options::String help =
        "List of modes to sum. Each entry is [n, m, weight].";
    static std::string name() { return "Modes"; }
  };

  static constexpr Options::String help =
      "A laser beam with arbitrary Hermite-Gaussian incoherent summation.";
  using options = tmpl::list<BeamWidth, PolynomialModes>;

  LaserBeam() = default;
  LaserBeam(const LaserBeam&) = default;
  LaserBeam& operator=(const LaserBeam&) = default;
  LaserBeam(LaserBeam&&) = default;
  LaserBeam& operator=(LaserBeam&&) = default;
  ~LaserBeam() = default;

  /// \cond
  explicit LaserBeam(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(LaserBeam);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<LaserBeam>(*this);
  }

  LaserBeam(double beam_width, const std::vector<ModeInfo>& modes);

  double beam_width() const { return beam_width_; }

  // Accessor required for operator== in the .cpp file
  const std::vector<ModeInfo>& modes() const { return modes_; }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {3, elliptic::BoundaryConditionType::Neumann};
  }

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 ::Tags::Normalized<
                     domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>>;
  using volume_tags = tmpl::list<>;

  // The main physics application function
  void apply(gsl::not_null<tnsr::I<DataVector, 3>*> displacement,
             gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
             const tnsr::iJ<DataVector, 3>& deriv_displacement,
             const tnsr::I<DataVector, 3>& x,
             const tnsr::i<DataVector, 3>& face_normal) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<tnsr::I<DataVector, 3>*> displacement,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
      const tnsr::iJ<DataVector, 3>& deriv_displacement);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    p | beam_width_;
    p | modes_;
  }

 private:
  double beam_width_{std::numeric_limits<double>::signaling_NaN()};
  std::vector<ModeInfo> modes_;
};

// Comparison operators defined in .cpp
bool operator==(const LaserBeam& lhs, const LaserBeam& rhs);
bool operator!=(const LaserBeam& lhs, const LaserBeam& rhs);
bool operator==(const LaserBeam::ModeInfo& lhs, const LaserBeam::ModeInfo& rhs);
bool operator!=(const LaserBeam::ModeInfo& lhs, const LaserBeam::ModeInfo& rhs);

}  // namespace Elasticity::BoundaryConditions
