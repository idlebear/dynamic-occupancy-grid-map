// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/ego_motion_compensation.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__global__ void moveParticlesKernel(ParticlesSoA particle_array, float x_move, float y_move,
                                    float cos_theta, float sin_theta, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        auto x = particle_array.state[i][0];
        auto y = particle_array.state[i][1];
        auto vx = particle_array.state[i][2];
        auto vy = particle_array.state[i][3];

        particle_array.state[i][0] = cos_theta * x + sin_theta * y - x_move;
        particle_array.state[i][1] = -sin_theta * x + cos_theta * y - y_move;
        particle_array.state[i][2] = cos_theta * vx + sin_theta * vy;
        particle_array.state[i][3] = -sin_theta * vx + cos_theta * vy;
    }
}

__global__ void moveMapKernel(GridCell* __restrict__ grid_cell_array, const GridCell* __restrict__ old_grid_cell_array,
                              float cos_theta, float sin_theta, float x_move, float y_move, int grid_size,
                              float grid_resolution)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size)
    {
        int rel_x = x - grid_size / 2;
        int rel_y = y - grid_size / 2;

        int old_x = int(cos_theta * rel_x + sin_theta * rel_y - x_move) + grid_size / 2;
        int old_y = int(-sin_theta * rel_x + cos_theta * rel_y - y_move) + grid_size / 2;

//        int old_x = int(cos_theta * float(x - grid_size/2) + sin_theta * float(y - grid_size / 2) - x_move);
//        int old_y = int(-sin_theta * float(x) + cos_theta * float(y) - y_move);
//        int new_x = int( float(x) - x_move);
//        int new_y = int( float(y) - y_move);

        if (old_x >= 0 && old_x < grid_size && old_y >= 0 && old_y < grid_size)
        {
            int index = x + grid_size * y;
            int old_index = old_x + grid_size * old_y;
            grid_cell_array[index] = old_grid_cell_array[old_index];

            // rotate the mean cell velocities as well
            // TODO: May also need to 'rotate' the variances as well, but since they are always positive, it
            //       really amounts to a reproportioning.
            auto mean_x_vel = grid_cell_array[index].mean_x_vel;
            auto mean_y_vel = grid_cell_array[index].mean_y_vel;

            grid_cell_array[index].mean_x_vel = cos_theta * mean_x_vel + sin_theta * mean_y_vel;
            grid_cell_array[index].mean_y_vel = -sin_theta * mean_x_vel + cos_theta * mean_y_vel;
        }
    }
}

} /* namespace dogm */
