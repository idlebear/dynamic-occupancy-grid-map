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

// __global__ void moveParticlesKernel(ParticlesSoA particle_array, float x_move, float y_move,
//                                     float cos_theta, float sin_theta, int particle_count, int grid_size)
// {
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
//     {
//         /// calculate rotation relative to the centre
//         auto  grid_centre = float(grid_size) / 2;
//         auto x = particle_array.state[i][0] - grid_centre - x_move;
//         auto y = particle_array.state[i][1] - grid_centre - y_move;
//         auto vx = particle_array.state[i][2];
//         auto vy = particle_array.state[i][3];

//         particle_array.state[i][0] = cos_theta * x + sin_theta * y + grid_centre;
//         particle_array.state[i][1] = -sin_theta * x + cos_theta * y + grid_centre;
//         particle_array.state[i][2] = cos_theta * vx + sin_theta * vy;
//         particle_array.state[i][3] = -sin_theta * vx + cos_theta * vy;
//     }
// }


__global__ void moveParticlesKernel(ParticlesSoA particle_array, int x_move, int y_move,
                                    float cos_theta, float sin_theta, int particle_count,
                                    float resolution, int grid_size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        particle_array.state[i][0] -= (x_move * resolution);
        particle_array.state[i][1] -= (y_move * resolution);

        particle_array.grid_cell_idx[i] = static_cast<int>(particle_array.state[i][1] / resolution) * grid_size
        + static_cast<int>(particle_array.state[i][0] / resolution);
    }
}




__global__ void moveMapKernel(GridCellsSoA grid_cell_array, GridCellsSoA old_grid_cell_array,
                              MeasurementCellsSoA meas_cell_array, ParticlesSoA particle_array,
                              int x_move, int y_move, float cos_theta, float sin_theta, int grid_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    float eps = 0.001f;

    if (x < grid_size && y < grid_size)
    {

        /// calculate rotation relative to the centre
        int grid_centre = grid_size / 2;
        auto rel_x = x - grid_centre + x_move;
        auto rel_y = y - grid_centre + y_move;
        auto old_x = int( cos_theta * rel_x - sin_theta * rel_y + grid_centre);
        auto old_y = int( sin_theta * rel_x + cos_theta * rel_y + grid_centre);

        int old_index = old_x + grid_size * old_y;
        int new_index = x + grid_size * y;

        // BUGBUG -- keep the revised correction around for comparison
        // int old_y = y + y_move;
        // int old_x = x + x_move;
        // int old_index = old_x + grid_size * old_y;

        if (old_x >= 0 && old_x < grid_size && old_y >= 0 && old_y < grid_size && meas_cell_array.occ_mass[old_index] > eps)
        {
            grid_cell_array.copy(old_grid_cell_array, new_index, old_index);

            // rotate the mean cell velocities as well
            // TODO: May also need to 'rotate' the variances as well, but since they are always positive, it
            //       really amounts to a reproportioning.
            auto mean_x_vel = grid_cell_array.mean_x_vel[new_index];
            auto mean_y_vel = grid_cell_array.mean_y_vel[new_index];

            grid_cell_array.mean_x_vel[new_index] = cos_theta * mean_x_vel + sin_theta * mean_y_vel;
            grid_cell_array.mean_y_vel[new_index] = -sin_theta * mean_x_vel + cos_theta * mean_y_vel;
        }
        else
        {
            // delete particles on old cells? looks like it break something
            // for (int i = old_grid_cell_array.start_idx[old_index]; i < old_grid_cell_array.end_idx[old_index]; ++i)
            //     particle_array.weight[i] = 0;
            grid_cell_array.start_idx[new_index] = -1;
            grid_cell_array.end_idx[new_index] = -1;
            grid_cell_array.new_born_occ_mass[new_index] = 0.0f;
            grid_cell_array.pers_occ_mass[new_index] = 0.0f;
            grid_cell_array.free_mass[new_index] = 0.0f;
            grid_cell_array.occ_mass[new_index] = 0.0f;
            grid_cell_array.pred_occ_mass[new_index] = 0.0f;

            grid_cell_array.mu_A[new_index] = 0.0f;
            grid_cell_array.mu_UA[new_index] = 0.0f;

            grid_cell_array.w_A[new_index] = 0.0f;
            grid_cell_array.w_UA[new_index] = 0.0f;

            grid_cell_array.mean_x_vel[new_index] = 0.0f;
            grid_cell_array.mean_y_vel[new_index] = 0.0f;
            grid_cell_array.var_x_vel[new_index] = 0.0f;
            grid_cell_array.var_y_vel[new_index] = 0.0f;
            grid_cell_array.covar_xy_vel[new_index] = 0.0f;

        }
    }
}

} /* namespace dogm */
