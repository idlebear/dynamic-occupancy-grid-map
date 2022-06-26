// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/mapping/laser_to_meas_grid.h"
#include "dogm/mapping/kernel/measurement_grid.h"

dogm::LaserMeasurementGrid::LaserMeasurementGrid(const Params& laser_params, float grid_length, float grid_resolution)
    : grid_size(static_cast<int>(grid_length / grid_resolution)), grid_resolution( grid_resolution), laser_params(laser_params),
      polar_width(ceil( laser_params.fov / laser_params.angle_increment )), polar_height( ceil( laser_params.max_range / laser_params.resolution ) )
{
    int grid_cell_count = grid_size * grid_size;

    meas_grid.init(grid_cell_count, true);

    theta_min = - (laser_params.fov / 2.0);
    CUDA_CALL(cudaMalloc(&polar_grid, polar_width * polar_height * sizeof(float2)));
}

dogm::LaserMeasurementGrid::~LaserMeasurementGrid()
{
    CUDA_CALL(cudaFree(polar_grid));
    meas_grid.free();
}

dogm::MeasurementCellsSoA dogm::LaserMeasurementGrid::generateGrid(const std::vector<float>& measurements)
{
    const int num_measurements = measurements.size();

    float* d_measurements;
    CUDA_CALL(cudaMalloc(&d_measurements, num_measurements * sizeof(float)));
    CUDA_CALL(
        cudaMemcpy(d_measurements, measurements.data(), num_measurements * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dim_block(32, 32);
    dim3 polar_grid_dim(divUp(polar_width, dim_block.x), divUp(polar_height, dim_block.y));
    dim3 cart_grid_dim(divUp(grid_size, dim_block.x), divUp(grid_size, dim_block.y));

    // convert the measurement information into a polar representation
    createPolarGridKernel<<<polar_grid_dim, dim_block>>>(polar_grid, d_measurements, polar_width, polar_height,
                                                          laser_params.resolution);

    CUDA_CALL(cudaGetLastError());

    // // transform polar representation to a cartesian grid
    transformPolarGridToCartesian<<<cart_grid_dim, dim_block>>>( meas_grid, grid_size, grid_resolution,
        polar_grid, polar_width, polar_height, theta_min, laser_params.angle_increment, laser_params.resolution,
        true );
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaFree(d_measurements));
    CUDA_CALL(cudaDeviceSynchronize());

    return meas_grid;
}
