// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace dogm
{

struct GridCell;
struct ParticlesSoA;

__global__ void moveParticlesKernel(ParticlesSoA particle_array, float x_move, float y_move,
                                    float cos_theta, float sin_theta, int particle_count, int grid_size);

__global__ void moveMapKernel(GridCell* __restrict__ grid_cell_array, const GridCell* __restrict__ old_grid_cell_array,
                              float x_move, float y_move, float cos_theta, float sin_theta, int grid_size);

} /* namespace dogm */
