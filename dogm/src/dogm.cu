/*
MIT License

Copyright (c) 2019 Michael Kösel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "dogm.h"
#include "dogm_types.h"
#include "cuda_utils.h"
#include "common.h"

#include "kernel/measurement_grid.h"
#include "kernel/init.h"
#include "kernel/predict.h"
#include "kernel/particle_to_grid.h"
#include "kernel/mass_update.h"
#include "kernel/init_new_particles.h"
#include "kernel/update_persistent_particles.h"
#include "kernel/statistical_moments.h"
#include "kernel/resampling.h"

#include "opengl/renderer.h"
#include "opengl/texture.h"
#include "opengl/framebuffer.h"

#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>

#include <cuda_runtime.h>

int DOGM::BLOCK_SIZE = 256;

DOGM::DOGM(const GridParams& params, const LaserSensorParams& laser_params)
	: params(params),
	  laser_params(laser_params),
	  grid_size(static_cast<int>(params.size / params.resolution)),
	  particle_count(params.particle_count),
	  grid_cell_count(grid_size * grid_size),
	  new_born_particle_count(params.new_born_particle_count)
{
	grid_cell_array.resize(grid_cell_count);
	particle_array.resize(particle_count);
	particle_array_next.resize(particle_count);
	birth_particle_array.resize(new_born_particle_count);
	meas_cell_array.resize(grid_cell_count);
	polar_meas_cell_array.resize(100 * grid_size);
	weight_array.resize(particle_count);
	birth_weight_array.resize(new_born_particle_count);
	born_masses_array.resize(grid_cell_count);

	initialize();
}

DOGM::~DOGM()
{
}

void DOGM::initialize()
{
	initParticlesKernel<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, grid_size);

	initBirthParticlesKernel<<<divUp(new_born_particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(birth_particle_array, grid_size);

	initGridCellsKernel<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(grid_cell_array, meas_cell_array, grid_size);

	CHECK_ERROR(cudaGetLastError());
	
	renderer = std::make_unique<Renderer>(grid_size, laser_params.fov, params.size, laser_params.max_range);
}

void DOGM::updateParticleFilter(float dt)
{
	particlePrediction(dt);
	particleAssignment();
	gridCellOccupancyUpdate();
	updatePersistentParticles();
	initializeNewParticles();
	statisticalMoments();
	resampling();

	particle_array = particle_array_next;

	CHECK_ERROR(cudaDeviceSynchronize());
}

void DOGM::updateMeasurementGridFromArray(const std::vector<float2>& measurements)
{
	thrust::device_vector<float2> d_measurements(measurements);
	float2* d_measurements_array = thrust::raw_pointer_cast(d_measurements.data());

	dim3 block_dim(32, 32);
	dim3 cart_grid_dim(divUp(grid_size, block_dim.x), divUp(grid_size, block_dim.y));

	gridArrayToMeasurementGridKernel<<<cart_grid_dim, block_dim>>>(meas_cell_array, d_measurements_array, grid_size);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
}

void DOGM::updateMeasurementGrid(float* measurements, int num_measurements)
{
	//std::cout << "DOGM::updateMeasurementGrid" << std::endl;

	float* d_measurements;
	CHECK_ERROR(cudaMalloc(&d_measurements, num_measurements * sizeof(float)));
	CHECK_ERROR(cudaMemcpy(d_measurements, measurements, num_measurements * sizeof(float), cudaMemcpyHostToDevice));

	const int polar_width = num_measurements;
	const int polar_height = grid_size;

	dim3 block_dim(32, 32);
	dim3 grid_dim(divUp(polar_width, block_dim.x), divUp(polar_height, block_dim.y));
	dim3 cart_grid_dim(divUp(grid_size, block_dim.x), divUp(grid_size, block_dim.y));

	const float anisotropy_level = 16.0f;
	Texture polar_texture(polar_width, polar_height, anisotropy_level);
	cudaSurfaceObject_t polar_surface;
	
	// create polar texture
	polar_texture.beginCudaAccess(&polar_surface);
	createPolarGridTextureKernel2<<<grid_dim, block_dim>>>(polar_surface, polar_meas_cell_array.data().get(), d_measurements, polar_width, polar_height, params.resolution);

	CHECK_ERROR(cudaGetLastError());
	polar_texture.endCudaAccess(polar_surface);
	
	// render cartesian image to texture using polar texture
	renderer->renderToTexture(polar_texture);
	
	Framebuffer* framebuffer = renderer->getFrameBuffer();
	cudaSurfaceObject_t cartesian_surface;

	framebuffer->beginCudaAccess(&cartesian_surface);
	// transform RGBA texture to measurement grid
	cartesianGridToMeasurementGridKernel<<<cart_grid_dim, block_dim>>>(meas_cell_array, cartesian_surface, grid_size);

	CHECK_ERROR(cudaGetLastError());
	framebuffer->endCudaAccess(cartesian_surface);

	CHECK_ERROR(cudaFree(d_measurements));
	CHECK_ERROR(cudaDeviceSynchronize());
}

void DOGM::particlePrediction(float dt)
{
	//std::cout << "DOGM::particlePrediction" << std::endl;

	glm::mat4x4 transition_matrix(1, 0, dt, 0, 
                                  0, 1, 0, dt, 
                                  0, 0, 1, 0, 
                                  0, 0, 0, 1);


	// FIXME: glm uses column major, we need row major
	transition_matrix = glm::transpose(transition_matrix);

	predictKernel<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, grid_size, params.persistence_prob, 
		transition_matrix, params.process_noise_position, params.process_noise_velocity);

	CHECK_ERROR(cudaGetLastError());
}

void DOGM::particleAssignment()
{
	//std::cout << "DOGM::particleAssignment" << std::endl;

	reinitGridParticleIndices<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(grid_cell_array);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

	thrust::sort(particle_array.begin(), particle_array.end(), GPU_LAMBDA(Particle x, Particle y)
	{
		return x.grid_cell_idx < y.grid_cell_idx;
	});

	particleToGridKernel<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, grid_cell_array, weight_array);

	CHECK_ERROR(cudaGetLastError());
}

void DOGM::gridCellOccupancyUpdate()
{
	//std::cout << "DOGM::gridCellOccupancyUpdate" << std::endl;

	CHECK_ERROR(cudaDeviceSynchronize());
	thrust::device_vector<float> weights_accum(particle_count);
	accumulate(weight_array, weights_accum);

	gridCellPredictionUpdateKernel<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(grid_cell_array, particle_array, weight_array,
		weights_accum, meas_cell_array, born_masses_array, params.birth_prob, params.persistence_prob);

	CHECK_ERROR(cudaGetLastError());
}

void DOGM::updatePersistentParticles()
{
	//std::cout << "DOGM::updatePersistentParticles" << std::endl;

	updatePersistentParticlesKernel1<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, meas_cell_array,
		weight_array);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

	thrust::device_vector<float> weights_accum(particle_count);
	accumulate(weight_array, weights_accum);

	updatePersistentParticlesKernel2<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(grid_cell_array,
		weights_accum);

	CHECK_ERROR(cudaGetLastError());

	updatePersistentParticlesKernel3<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, meas_cell_array,
		grid_cell_array, weight_array);

	CHECK_ERROR(cudaGetLastError());
}

void DOGM::initializeNewParticles()
{
	//std::cout << "DOGM::initializeNewParticles" << std::endl;

	initBirthParticlesKernel<<<divUp(new_born_particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(birth_particle_array, grid_size);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

	thrust::device_vector<float> particle_orders_accum(grid_cell_count);
	accumulate(born_masses_array, particle_orders_accum);

	normalize_particle_orders(particle_orders_accum, grid_cell_count, new_born_particle_count);

	initNewParticlesKernel1<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, grid_cell_array,
		meas_cell_array, weight_array, born_masses_array, birth_particle_array, particle_orders_accum);

	CHECK_ERROR(cudaGetLastError());

	initNewParticlesKernel2<<<divUp(new_born_particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(birth_particle_array,
		grid_cell_array, grid_size);

	CHECK_ERROR(cudaGetLastError());

	CHECK_ERROR(cudaDeviceSynchronize());
	thrust::sort(birth_particle_array.begin(), birth_particle_array.end(), GPU_LAMBDA(Particle x, Particle y)
	{
		return x.grid_cell_idx < y.grid_cell_idx;
	});

	copyBirthWeightKernel<<<divUp(new_born_particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(birth_particle_array, birth_weight_array);

	CHECK_ERROR(cudaGetLastError());

	//thrust::device_ptr<float> weight(weight_array);
	//float res_max = *thrust::max_element(weight, weight + particle_count);
	//printf("Persistent max: %f\n", res_max);

	//thrust::device_ptr<float> birth_weight(birth_weight_array);
	//float res2_max = *thrust::max_element(birth_weight, birth_weight + new_born_particle_count);
	//printf("New born max: %f\n", res2_max);
}

void DOGM::statisticalMoments()
{
	//std::cout << "DOGM::statisticalMoments" << std::endl;

	thrust::device_vector<float> vel_x(particle_count);
	thrust::device_vector<float> vel_y(particle_count);
	thrust::device_vector<float> vel_x_squared(particle_count);
	thrust::device_vector<float> vel_y_squared(particle_count);
	thrust::device_vector<float> vel_xy(particle_count);

	statisticalMomentsKernel1<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, weight_array,
		vel_x, vel_y, vel_x_squared, vel_y_squared, vel_xy);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

	thrust::device_vector<float> vel_x_accum(particle_count);
	accumulate(vel_x, vel_x_accum);

	thrust::device_vector<float> vel_y_accum(particle_count);
	accumulate(vel_y, vel_y_accum);

	thrust::device_vector<float> vel_x_squared_accum(particle_count);
	accumulate(vel_x_squared, vel_x_squared_accum);

	thrust::device_vector<float> vel_y_squared_accum(particle_count);
	accumulate(vel_y_squared, vel_y_squared_accum);

	thrust::device_vector<float> vel_xy_accum(particle_count);
	accumulate(vel_xy, vel_xy_accum);

	statisticalMomentsKernel2<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(grid_cell_array, vel_x_accum,
		vel_y_accum, vel_x_squared_accum, vel_y_squared_accum, vel_xy_accum);

	CHECK_ERROR(cudaGetLastError());
}

void DOGM::resampling()
{
	//std::cout << "DOGM::resampling" << std::endl;

	CHECK_ERROR(cudaDeviceSynchronize());

	const int max = particle_count + new_born_particle_count;
	thrust::device_vector<int> rand_array(particle_count);
	thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(particle_count), rand_array.begin(),
		GPU_LAMBDA(int index)
	{
		//int seed = hash(index);
		thrust::default_random_engine rng;// (seed);
		thrust::uniform_int_distribution<int> dist(0, max);
		rng.discard(index);
		return dist(rng);
	});
	thrust::sort(rand_array.begin(), rand_array.end());

	thrust::device_vector<float> joint_weight_array;
	joint_weight_array.insert(joint_weight_array.end(), weight_array.begin(), weight_array.end());
	joint_weight_array.insert(joint_weight_array.end(), birth_weight_array.begin(), birth_weight_array.end());

	thrust::device_vector<float> joint_weight_accum(joint_weight_array.size());
	accumulate(joint_weight_array, joint_weight_accum);

	thrust::device_vector<int> idx_resampled(particle_count);
	calc_resampled_indices(joint_weight_accum, rand_array, idx_resampled);

	float joint_max = joint_weight_accum.back();

	printf("joint_max: %f\n", joint_max);

	resamplingKernel<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, particle_array_next,
		birth_particle_array, idx_resampled, joint_max);

	CHECK_ERROR(cudaGetLastError());
}
