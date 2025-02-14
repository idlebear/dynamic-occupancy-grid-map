// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace dogm
{

__global__ void setupRandomStatesKernel(curandState* __restrict__ states, int count);

__global__ void initParticlesKernel(ParticlesSoA particle_array, curandState* __restrict__ global_state, float velocity,
                                    int grid_size, int particle_count, float resolution);

__global__ void initBirthParticlesKernel(ParticlesSoA birth_particle_array, curandState* __restrict__ global_state,
                                         float velocity, int grid_size, int particle_count);

__global__ void initGridCellsKernel(GridCellsSoA grid_cell_array,
                                    MeasurementCellsSoA meas_cell_array, int grid_size, int cell_count);

__global__ void reinitGridParticleIndices(GridCellsSoA grid_cell_array, int cell_count);

} /* namespace dogm */
