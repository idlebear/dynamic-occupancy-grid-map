// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <device_launch_parameters.h>

namespace dogm
{
struct MeasurementCell;
}

__global__ void createPolarGridKernel(float2 *polar_grid, const float* __restrict__ measurements,
                                             int width, int height, float resolution);

__global__ void transformPolarGridToCartesian(  dogm::MeasurementCell* __restrict__ meas_grid, int grid_size, float grid_resolution,
                                                const float2* polar_grid, int polar_width,  int polar_height,
                                                float theta_min, float theta_inc, float r_inc, bool use_nearest = true );
