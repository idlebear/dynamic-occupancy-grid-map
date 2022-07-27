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
                                    int particle_count, float resolution, int grid_size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        particle_array.state[i][0] -= x_move;
        particle_array.state[i][1] -= y_move;

        particle_array.grid_cell_idx[i] = std::nearbyint(particle_array.state[i][1] / resolution) * grid_size
            + std::nearbyint(particle_array.state[i][0] / resolution);
    }
}




__global__ void moveMapKernel(GridCellsSoA grid_cell_array, GridCellsSoA old_grid_cell_array,
                              MeasurementCellsSoA meas_cell_array, ParticlesSoA particle_array,
                              int x_grid_move, int y_grid_move, int grid_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    float eps = 0.001f;

    if (x < grid_size && y < grid_size)
    {
        auto old_x = x + x_grid_move;
        auto old_y = y + y_grid_move;
        int old_index = old_x + grid_size * old_y;
        int new_index = x + grid_size * y;

        if (old_x >= 0 && old_x < grid_size && old_y >= 0 && old_y < grid_size )
        {
            grid_cell_array.copy(old_grid_cell_array, new_index, old_index);
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
