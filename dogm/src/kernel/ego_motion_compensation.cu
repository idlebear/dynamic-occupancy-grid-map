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

__global__ void moveParticlesKernel(ParticlesSoA particle_array, float x_move, float y_move, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        particle_array.state[i][0] -= x_move;
        particle_array.state[i][1] -= y_move;
    }
}

__global__ void moveMapKernel(GridCell* __restrict__ grid_cell_array, const GridCell* __restrict__ old_grid_cell_array,
                              float x_move, float y_move, int grid_size, float grid_resolution)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size)
    {
        int old_x = x + x_move;
        int old_y = y - y_move;

        if (old_x >= 0 && old_x < grid_size && old_y >= 0 && old_y < grid_size)
        {
            int index = x + grid_size * y;
            int old_index = old_x + grid_size * old_y;
            grid_cell_array[index] = old_grid_cell_array[old_index];
        }
    }
}

} /* namespace dogm */
