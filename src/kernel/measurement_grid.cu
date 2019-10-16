#include "kernel/measurement_grid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI 3.14159265358979323846f

__device__ float binary_bayes_filter(float likelihood, float prior)
{
	float l = (likelihood / (1.0f - likelihood)) * (prior / (1.0f - prior));

	return l / (1.0 + l);
}

__device__ float pFree(int i, float p_min, float p_max, int max_range)
{
	return p_min + i * (p_max - p_min) / max_range;
}

__device__ float pOcc(int r, float zk, int index)
{
	float alpha = 1.0f;
	float delta = 2.5f;

	//return (alpha / (delta * sqrt(2.0f * PI))) * exp(-0.5f * (index - r) * (index - r) / (delta * delta));
	return 0.8f * exp(-0.5f * (index - r) * (index - r) / (delta * delta));
}

__device__ float inverse_sensor_model(int i, float resolution, float zk, float r_max)
{
	if (isfinite(zk))
	{
		int r = (int)(zk / resolution);

		if (i <= r)
		{
			return max(pFree(i, 0.01, 0.5f, r_max), pOcc(r, zk, i));
		}

		return max(0.5f, pOcc(r, zk, i));
	}
	else
	{
		return pFree(i, 0.01, 0.5f, r_max);
	}
}

__device__ float2 probability_to_masses(float prob)
{
	// Masses: mOcc, mFree
	if (prob == 0.5f)
	{
		return make_float2(0.0f, 0.0f);
	}
	else
	{
		return make_float2(prob, 1.0f - prob);
	}
}

__global__ void createPolarGridTextureKernel(cudaSurfaceObject_t polar, float* measurements, int width, int height, float resolution)
{
	const int theta = blockIdx.x * blockDim.x + threadIdx.x;
	const int range = blockIdx.y * blockDim.y + threadIdx.y;

	if (theta < width && range < height)
	{
		const float epsilon = 0.00001f;
		const float zk = measurements[theta];

		float prob = inverse_sensor_model(range, resolution, zk, height);
		prob = max(epsilon, min(1.0f - epsilon, prob));

		surf2Dwrite(prob, polar, theta * sizeof(float), range);
	}
}

__global__ void fusePolarGridTextureKernel(cudaSurfaceObject_t polar, float* measurements, int width, int height, float resolution)
{
	const int theta = blockIdx.x * blockDim.x + threadIdx.x;
	const int range = blockIdx.y * blockDim.y + threadIdx.y;

	if (theta < width && range < height)
	{
		const float epsilon = 0.00001f;
		const float zk = measurements[theta];

		float prior = surf2Dread<float>(polar, theta * sizeof(float), range);
		float likelihood = inverse_sensor_model(range, resolution, zk, height);
		likelihood = max(epsilon, min(1.0f - epsilon, likelihood));

		float prob = binary_bayes_filter(likelihood, prior);
		//prob = max(epsilon, min(1.0f - epsilon, prob));

		surf2Dwrite(prob, polar, theta * sizeof(float), range);
	}
}

__global__ void cartesianGridToMeasurementGridKernel(MeasurementCell* meas_grid, cudaSurfaceObject_t cart, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = (height - y - 1) * width + x;

	if (x < width && y < height)
	{
		float4 color = surf2Dread<float4>(cart, x * sizeof(float4), y);
		float2 masses = probability_to_masses(color.x);

		meas_grid[index].occ_mass = masses.x;
		meas_grid[index].free_mass = masses.y;

		meas_grid[index].likelihood = 1.0f;
		meas_grid[index].p_A = 1.0f;
	}
}
