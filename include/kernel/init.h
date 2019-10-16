#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void initParticlesKernel(Particle* particle_array, int width, int height, int particle_count);

__global__ void initGridCellsKernel(GridCell* grid_cell_array, int width, int height, int cell_count);
