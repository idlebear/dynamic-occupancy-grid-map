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

        particle_array.state[i][0] = cos_theta * x + sin_theta * y - float(x_move);
        particle_array.state[i][1] = -sin_theta * x + cos_theta * y - float(y_move);
    }
}

__global__ void moveMapKernel(GridCell* __restrict__ grid_cell_array, const GridCell* __restrict__ old_grid_cell_array,
                              float cos_theta, float sin_theta, float x_move, float y_move, int grid_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size)
    {
        int new_x = int(cos_theta * float(x) + sin_theta * float(y) - x_move);
        int new_y = int(-sin_theta * float(x) + cos_theta * float(y) - y_move);

        if (new_x >= 0 && new_x < grid_size && new_y >= 0 && new_y < grid_size)
        {
            int index = x + grid_size * y;
            int new_index = new_x + grid_size * new_y;
            grid_cell_array[new_index] = old_grid_cell_array[index];
        }
    }
}

} /* namespace dogm */
