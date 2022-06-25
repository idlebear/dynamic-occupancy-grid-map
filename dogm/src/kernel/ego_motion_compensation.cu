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
                                    float cos_theta, float sin_theta, int particle_count, int grid_size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        /// calculate rotation relative to the centre
        auto  grid_centre = float(grid_size) / 2;
        auto x = particle_array.state[i][0] - grid_centre - x_move;
        auto y = particle_array.state[i][1] - grid_centre - y_move;
        auto vx = particle_array.state[i][2];
        auto vy = particle_array.state[i][3];

        particle_array.state[i][0] = cos_theta * x + sin_theta * y + grid_centre;
        particle_array.state[i][1] = -sin_theta * x + cos_theta * y + grid_centre;
        particle_array.state[i][2] = cos_theta * vx + sin_theta * vy;
        particle_array.state[i][3] = -sin_theta * vx + cos_theta * vy;
    }
}


__global__ void moveMapKernel(GridCell* __restrict__ grid_cell_array, const GridCell* __restrict__ old_grid_cell_array,
                              float x_move, float y_move, float cos_theta, float sin_theta, int grid_size)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size)
    {
        /// calculate rotation relative to the centre
        auto  grid_centre = float(grid_size / 2.0);
        auto rel_x = float(x) - grid_centre - x_move;
        auto rel_y = float(y) - grid_centre - y_move;

        auto new_x = int( cos_theta * rel_x + sin_theta * rel_y + grid_centre);
        auto new_y = int(-sin_theta * rel_x + cos_theta * rel_y + grid_centre);

        if (new_x >= 0 && new_x < grid_size && new_y >= 0 && new_y < grid_size)
        {
            unsigned int index = x + grid_size * y;
            unsigned int new_index = new_x + grid_size * new_y;
            grid_cell_array[new_index] = old_grid_cell_array[index];

            // rotate the mean cell velocities as well
            // TODO: May also need to 'rotate' the variances as well, but since they are always positive, it
            //       really amounts to a reproportioning.
            auto mean_x_vel = grid_cell_array[new_index].mean_x_vel;
            auto mean_y_vel = grid_cell_array[new_index].mean_y_vel;

            grid_cell_array[index].mean_x_vel = cos_theta * mean_x_vel + sin_theta * mean_y_vel;
            grid_cell_array[index].mean_y_vel = -sin_theta * mean_x_vel + cos_theta * mean_y_vel;

        }
    }
}

} /* namespace dogm */
