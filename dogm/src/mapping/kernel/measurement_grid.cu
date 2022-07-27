// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/dogm_types.h"
#include "dogm/mapping/kernel/measurement_grid.h"

#include <cuda_runtime.h>

__device__ float2 combine_masses(float2 prior, float2 meas)
{
    // Masses: mOcc, mFree
    float occ = prior.x;
    float free = prior.y;

    float meas_occ = meas.x;
    float meas_free = meas.y;

    float unknown_pred = 1.0f - occ - free;
    float meas_cell_unknown = 1.0f - meas_occ - meas_free;
    float K = free * meas_occ + occ * meas_free;

    float2 res;
    res.x = (occ * meas_cell_unknown + unknown_pred * meas_occ + occ * meas_occ) / (1.0f - K);
    res.y = (free * meas_cell_unknown + unknown_pred * meas_free + free * meas_free) / (1.0f - K);

    return res;
}

__device__ auto pFree(int i, float p_min, float p_max, int max_range) -> float
{
    return p_min + i * (p_max - p_min) / max_range;
}

__device__ auto pOcc(int r, float zk, int index, float resolution, float stddev_range) -> float
{
    auto occ_max = 0.95f;
    auto diff = float(index - r) * resolution;

    return occ_max * exp(-0.5f * diff * diff / (stddev_range*stddev_range));
}

__device__ auto inverse_sensor_model(int i, float resolution, float zk, float r_max, float stddev_range) -> float2
{
    // Masses: mOcc, mFree

    const float free = pFree(i, 0.15f, 1.0f, r_max);
    float2 res;

    if (isfinite(zk))
    {
        const int r = static_cast<int>(zk / resolution);
        const float occ = pOcc(r, zk, i, resolution, stddev_range);

        if (i <= r) {
             res = occ > free ? make_float2(occ, 0.0f) : make_float2(0.0f, 1.0f - free);
        } else {
            res = occ > 0.5f ? make_float2(occ, 0.0f) : make_float2(0.0f, 0.0f);
        }
    } else {
        res = make_float2(0.0f, 1.0f - free);
    }

    return res;
}

__global__ void createPolarGridKernel(float2* polar_grid, const float* __restrict__ measurements,
    int width, int height, float resolution, float stddev_range)
{
    const unsigned int theta = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int range = blockIdx.y * blockDim.y + threadIdx.y;

    if (theta < width && range < height)
    {
        const float epsilon = 0.00001f;
        const float zk = measurements[theta];

        float2 masses = inverse_sensor_model(range, resolution, zk, height, stddev_range);
        masses.x = max(epsilon, min(1.0f - epsilon, masses.x));
        masses.y = max(epsilon, min(1.0f - epsilon, masses.y));
        polar_grid[range * width + theta] = masses;
    }
}


__global__ void transformPolarGridToCartesian(
    dogm::MeasurementCellsSoA meas_grid, int grid_size, float grid_resolution,
    const float2* polar_grid, int polar_width,  int polar_height,
    float theta_min, float theta_inc, float r_inc, bool use_nearest )
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size)
    {
        // find the true theta/radius that corresponds to the requested (x,y)
        float grid_x = float(x - grid_size / 2.0) * grid_resolution;
        float grid_y = float(y - grid_size / 2.0) * grid_resolution;

        // convert the desired X and Y coordinates in grid form into an angle and
        // radius.  The polar map cell is then found by subtracting the minimum angle and
        // dividing by the increment/resolution to find the actual column.
        float translated_theta = atan2( grid_y, grid_x ) * 180.0 / M_PI  - theta_min;
        float translated_r = sqrt(grid_x * grid_x + grid_y * grid_y);
        int theta1 = int(translated_theta / theta_inc);
        int r1 = int( translated_r / r_inc );

        if( translated_theta >= 0 && theta1 < polar_width && r1 < polar_height ) {
            // use bilinear interpolation to translate the nearest 4 polar objects
            // into the value of the grid coordinate
            int theta2;
            float theta1_prop;
            if( theta1 < polar_width - 1){
                theta2 = theta1 + 1;
                if( use_nearest ) {
                    theta1_prop = (float(theta2) * theta_inc - translated_theta) / theta_inc > 0.5 ? 1 : 0;
                } else {
                    theta1_prop = (float(theta2)*theta_inc - translated_theta)/theta_inc;
                }
            } else {
                theta2 = theta1;
                theta1_prop = 1;
            }
            float theta2_prop = 1 - theta1_prop;

            int r2;
            float r1_prop;
            if( r1 < polar_height - 1 ) {
                r2 = r1 + 1;
                if( use_nearest ) {
                    r1_prop = (float(r2) * r_inc - translated_r)/r_inc > 0.5 ? 1 : 0;
                } else {
                    r1_prop = (float(r2) * r_inc - translated_r)/r_inc;
                }
            } else {
                r2 = r1;
                r1_prop = 1;
            }
            float r2_prop = 1 - r1_prop;

            float2 m11 = polar_grid[ r1 * polar_width + theta1 ];
            float2 m12 = polar_grid[ r2 * polar_width + theta1 ];
            float2 m21 = polar_grid[ r1 * polar_width + theta2 ];
            float2 m22 = polar_grid[ r2 * polar_width + theta2 ];

            auto index = grid_size * y + x;
            meas_grid.occ_mass[index] = m11.x * theta1_prop * r1_prop
                   + m12.x * theta1_prop * r2_prop
                   + m21.x * theta2_prop * r1_prop
                   + m22.x * theta2_prop * r2_prop;
            meas_grid.free_mass[index] = m11.y * theta1_prop * r1_prop
                   + m12.y * theta1_prop * r2_prop
                   + m21.y * theta2_prop * r1_prop
                   + m22.y * theta2_prop * r2_prop;
            meas_grid.likelihood[index] = 1.0f;
            meas_grid.p_A[index] = 1.0f;
        }
    }
}

