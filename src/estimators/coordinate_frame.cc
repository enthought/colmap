// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "estimators/coordinate_frame.h"

#include "base/pose.h"
#include "base/undistortion.h"
#include "estimators/utils.h"
#include "optim/ransac.h"
#include "util/logging.h"
#include "util/misc.h"

namespace colmap {
namespace {


Eigen::Vector3d FindBestConsensusAxis(const std::vector<Eigen::Vector3d>& axes,
                                      const double max_distance) {
  if (axes.empty()) {
    return Eigen::Vector3d::Zero();
  }

  std::vector<int> inlier_idxs;
  inlier_idxs.reserve(axes.size());

  std::vector<int> best_inlier_idxs;
  best_inlier_idxs.reserve(axes.size());

  double best_inlier_distance_sum = std::numeric_limits<double>::max();

  for (size_t i = 0; i < axes.size(); ++i) {
    const Eigen::Vector3d ref_axis = axes[i];
    double inlier_distance_sum = 0;
    inlier_idxs.clear();
    for (size_t j = 0; j < axes.size(); ++j) {
      if (i == j) {
        inlier_idxs.push_back(j);
      } else {
        const double distance = 1 - ref_axis.dot(axes[j]);
        if (distance <= max_distance) {
          inlier_distance_sum += distance;
          inlier_idxs.push_back(j);
        }
      }
    }

    if (inlier_idxs.size() > best_inlier_idxs.size() ||
        (inlier_idxs.size() == best_inlier_idxs.size() &&
         inlier_distance_sum < best_inlier_distance_sum)) {
      best_inlier_distance_sum = inlier_distance_sum;
      best_inlier_idxs = inlier_idxs;
    }
  }

  if (best_inlier_idxs.empty()) {
    return Eigen::Vector3d::Zero();
  }

  Eigen::Vector3d best_axis(0, 0, 0);
  for (const auto idx : best_inlier_idxs) {
    best_axis += axes[idx];
  }
  best_axis /= best_inlier_idxs.size();

  return best_axis;
}

}  // namespace

Eigen::Vector3d EstimateGravityVectorFromImageOrientation(
    const Reconstruction& reconstruction,
    const double max_axis_distance) {
  std::vector<Eigen::Vector3d> downward_axes;
  downward_axes.reserve(reconstruction.NumRegImages());
  for (const auto image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    downward_axes.push_back(image.RotationMatrix().row(1));
  }
  return FindBestConsensusAxis(downward_axes, max_axis_distance);
}

}  // namespace colmap
