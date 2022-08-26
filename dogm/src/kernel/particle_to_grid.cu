// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include <cassert>

#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/particle_to_grid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "dogm/common.h"

namespace dogm
{

__device__ bool is_first_particle(const ParticlesSoA& particle_array, int i)
{
    return i == 0 || particle_array.grid_cell_idx[i] != particle_array.grid_cell_idx[i - 1];
}

__device__ bool is_last_particle(const ParticlesSoA& particle_array, int particle_count, int i)
{
    return i == particle_count - 1 || particle_array.grid_cell_idx[i] != particle_array.grid_cell_idx[i + 1];
}

__global__ void particleToGridKernel(const ParticlesSoA particle_array, GridCellsSoA grid_cell_array,
                                     float* __restrict__ weight_array, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        int j = particle_array.grid_cell_idx[i];

        if (is_first_particle(particle_array, i))
        {
            grid_cell_array.start_idx[j] = i;
        }
        if (is_last_particle(particle_array, particle_count, i))
        {
            grid_cell_array.end_idx[j] = i;
        }

        weight_array[i] = particle_array.weight[i];
    }
}


__global__ void sum_grid_weights(const GridCellsSoA grid_cell_array, const float* accum, float* sums, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x) {
        int start_idx = grid_cell_array.start_idx[i];
        int end_idx = grid_cell_array.end_idx[i];

        if( start_idx != -1 ) {
            sums[i] = subtract( accum, start_idx, end_idx );
        }
    }
}

void check_weights( const ParticlesSoA particles, int particle_count, const GridCellsSoA grid_cell_array, int cell_count,
                    dim3 grid_map_grid, dim3 block_dim ) {

    // accumulate the current/next weights
    thrust::device_vector<float> weight_accum(particle_count);
    accumulate(particles.weight, weight_accum);
    float weight_max = weight_accum.back();
    printf("Final weight sum: %f\n", weight_max);

    float *accum_ptr = thrust::raw_pointer_cast(weight_accum.data());
    thrust::device_vector<float> sums(cell_count);
    float *sum_ptr = thrust::raw_pointer_cast(sums.data());

    sum_grid_weights<<<grid_map_grid, block_dim>>>(grid_cell_array, accum_ptr, sum_ptr, cell_count);

    thrust::device_vector<float>::iterator it = sums.begin();
    for (int index = 0; it != sums.end(); ++it, ++index) {
        float val = *it;
        if( val < 0.0f ) {
            printf("Weight underflow at %d: %f\n", index, val);
        } else if( val > 1.0f ) {
            printf("Weight overflow at %d: %f\n", index, val);
        }
    }
}

} /* namespace dogm */
