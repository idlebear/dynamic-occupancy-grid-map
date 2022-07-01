// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include "cuda_utils.h"
#include <glm/vec4.hpp>

namespace dogm
{

struct GridCell
{
    int start_idx;
    int end_idx;
    float new_born_occ_mass;
    float pers_occ_mass;
    float free_mass;
    float occ_mass;
    float pred_occ_mass;
    float mu_A;
    float mu_UA;

    float w_A;
    float w_UA;

    float mean_x_vel;
    float mean_y_vel;
    float var_x_vel;
    float var_y_vel;
    float covar_xy_vel;
};

struct MeasurementCell
{
    float free_mass;
    float occ_mass;
    float likelihood;
    float p_A;
};

struct Particle
{
    int grid_cell_idx;
    float weight;
    bool associated;
    glm::vec4 state;
};

struct GridCellsSoA
{
    int* start_idx;
    int* end_idx;
    float* new_born_occ_mass;
    float* pers_occ_mass;
    float* free_mass;
    float* occ_mass;
    float* pred_occ_mass;
    float* mu_A;
    float* mu_UA;

    float* w_A;
    float* w_UA;

    float* mean_x_vel;
    float* mean_y_vel;
    float* var_x_vel;
    float* var_y_vel;
    float* covar_xy_vel;

    void *blk_ptr;

    int size;
    bool device;

    GridCellsSoA() : size(0), device(true) {}

    GridCellsSoA(int new_size, bool is_device) { init(new_size, is_device); }

    void init(int new_size, bool is_device)
    {
        size = new_size;
        device = is_device;
        auto bytes = size * sizeof(GridCell);

        if (device) {
            CUDA_CALL(cudaMalloc((void **) &blk_ptr, bytes));
        } else {
            blk_ptr = malloc(bytes);
        }

        __initialize_ptrs();
    }

    void __initialize_ptrs()
    {
        start_idx = reinterpret_cast<int *>(blk_ptr);
        end_idx = reinterpret_cast<int *>(start_idx + size);
        new_born_occ_mass = reinterpret_cast<float *>(end_idx + size);
        pers_occ_mass = reinterpret_cast<float *>(new_born_occ_mass + size);
        free_mass = reinterpret_cast<float *>(pers_occ_mass + size);
        occ_mass = reinterpret_cast<float *>(free_mass + size);
        pred_occ_mass = reinterpret_cast<float *>(occ_mass + size);
        mu_A = reinterpret_cast<float *>(pred_occ_mass + size);
        mu_UA = reinterpret_cast<float *>(mu_A + size);

        w_A = reinterpret_cast<float *>(mu_UA + size);
        w_UA = reinterpret_cast<float *>(w_A + size);

        mean_x_vel = reinterpret_cast<float *>(w_UA + size);
        mean_y_vel = reinterpret_cast<float *>(mean_x_vel + size);
        var_x_vel = reinterpret_cast<float *>(mean_y_vel + size);
        var_y_vel = reinterpret_cast<float *>(var_x_vel + size);
        covar_xy_vel = reinterpret_cast<float *>(var_y_vel + size);
    }

    void free()
    {
        assert(size);
        if (device)
        {
            CUDA_CALL(cudaFree(blk_ptr));
        }
        else
        {
            ::free(blk_ptr);
        }
        blk_ptr = nullptr;
        size = 0;
        __initialize_ptrs();
    }

    void copy(const GridCellsSoA& other, cudaMemcpyKind kind)
    {
        assert(size && size == other.size);
        auto bytes = size * sizeof(GridCell);
        CUDA_CALL(cudaMemcpy(blk_ptr, other.blk_ptr, bytes, kind));
        __initialize_ptrs();
    }

    void move( GridCellsSoA& other )
    {
        if( this == &other )
        {
            return;
        }
        if( device != other.device ) {
            copy( other, device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice );
            other.free();
            return;
        }
        free();

        blk_ptr = other.blk_ptr;
        __initialize_ptrs();

        other.blk_ptr = nullptr;
        other.__initialize_ptrs();
    }


    GridCellsSoA& operator=(const GridCellsSoA& other)
    {
        if (this != &other)
        {
            copy(other, cudaMemcpyDeviceToDevice);
        }

        return *this;
    }

    __device__ void copy(const GridCellsSoA& other, int index, int other_index)
    {
        start_idx[index] = other.start_idx[other_index];
        end_idx[index] = other.end_idx[other_index];
        new_born_occ_mass[index] = other.new_born_occ_mass[other_index];
        pers_occ_mass[index] = other.pers_occ_mass[other_index];
        free_mass[index] = other.free_mass[other_index];
        occ_mass[index] = other.occ_mass[other_index];
        pred_occ_mass[index] = other.pred_occ_mass[other_index];
        mu_A[index] = other.mu_A[other_index];
        mu_UA[index] = other.mu_UA[other_index];

        w_A[index] = other.w_A[other_index];
        w_UA[index] = other.w_UA[other_index];

        mean_x_vel[index] = other.mean_x_vel[other_index];
        mean_y_vel[index] = other.mean_y_vel[other_index];
        var_x_vel[index] = other.var_x_vel[other_index];
        var_y_vel[index] = other.var_y_vel[other_index];
        covar_xy_vel[index] = other.covar_xy_vel[other_index];
    }
};

struct MeasurementCellsSoA
{
    float* free_mass;
    float* occ_mass;
    float* likelihood;
    float* p_A;

    int size;
    bool device;

    MeasurementCellsSoA() : size(0), device(true) {}

    MeasurementCellsSoA(int new_size, bool is_device) { init(new_size, is_device); }

    void init(int new_size, bool is_device)
    {
        size = new_size;
        device = is_device;
        if (device)
        {
            CUDA_CALL(cudaMalloc((void**)&free_mass, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&occ_mass, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&likelihood, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&p_A, size * sizeof(float)));
        }
        else
        {
            free_mass = (float*)malloc(size * sizeof(float));
            occ_mass = (float*)malloc(size * sizeof(float));
            likelihood = (float*)malloc(size * sizeof(float));
            p_A = (float*)malloc(size * sizeof(float));
        }
    }

    void free()
    {
        if (device)
        {
            CUDA_CALL(cudaFree(free_mass));
            CUDA_CALL(cudaFree(occ_mass));
            CUDA_CALL(cudaFree(likelihood));
            CUDA_CALL(cudaFree(p_A));
        }
        else
        {
            ::free(free_mass);
            ::free(occ_mass);
            ::free(likelihood);
            ::free(p_A);
        }
    }

    void copy(const MeasurementCellsSoA& other, cudaMemcpyKind kind)
    {
        CUDA_CALL(cudaMemcpy(free_mass, other.free_mass, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(occ_mass, other.occ_mass, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(likelihood, other.likelihood, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(p_A, other.p_A, size * sizeof(float), kind));
    }

    MeasurementCellsSoA& operator=(const MeasurementCellsSoA& other)
    {
        if (this != &other)
        {
            copy(other, cudaMemcpyDeviceToDevice);
        }

        return *this;
    }

    __device__ void copy(const MeasurementCellsSoA& other, int index, int other_index)
    {
        free_mass[index] = other.free_mass[other_index];
        occ_mass[index] = other.occ_mass[other_index];
        likelihood[index] = other.likelihood[other_index];
        p_A[index] = other.p_A[other_index];
    }
};

struct ParticlesSoA
{
    glm::vec4* state;
    int* grid_cell_idx;
    float* weight;
    bool* associated;

    int size;
    bool device;

    ParticlesSoA() : size(0), device(true) {}

    ParticlesSoA(int new_size, bool is_device) { init(new_size, is_device); }

    void init(int new_size, bool is_device)
    {
        size = new_size;
        device = is_device;
        if (device)
        {
            CUDA_CALL(cudaMalloc((void**)&state, size * sizeof(glm::vec4)));
            CUDA_CALL(cudaMalloc((void**)&grid_cell_idx, size * sizeof(int)));
            CUDA_CALL(cudaMalloc((void**)&weight, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&associated, size * sizeof(bool)));
        }
        else
        {
            state = (glm::vec4*)malloc(size * sizeof(glm::vec4));
            grid_cell_idx = (int*)malloc(size * sizeof(int));
            weight = (float*)malloc(size * sizeof(float));
            associated = (bool*)malloc(size * sizeof(bool));
        }
    }

    void free()
    {
        if (device)
        {
            CUDA_CALL(cudaFree(state));
            CUDA_CALL(cudaFree(grid_cell_idx));
            CUDA_CALL(cudaFree(weight));
            CUDA_CALL(cudaFree(associated));
        }
        else
        {
            ::free(state);
            ::free(grid_cell_idx);
            ::free(weight);
            ::free(associated);
        }
    }

    void copy(const ParticlesSoA& other, cudaMemcpyKind kind)
    {
        CUDA_CALL(cudaMemcpy(grid_cell_idx, other.grid_cell_idx, size * sizeof(int), kind));
        CUDA_CALL(cudaMemcpy(weight, other.weight, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(associated, other.associated, size * sizeof(bool), kind));
        CUDA_CALL(cudaMemcpy(state, other.state, size * sizeof(glm::vec4), kind));
    }

    ParticlesSoA& operator=(const ParticlesSoA& other)
    {
        if (this != &other)
        {
            copy(other, cudaMemcpyDeviceToDevice);
        }

        return *this;
    }

    __device__ void copy(const ParticlesSoA& other, int index, int other_index)
    {
        grid_cell_idx[index] = other.grid_cell_idx[other_index];
        weight[index] = other.weight[other_index];
        associated[index] = other.associated[other_index];
        state[index] = other.state[other_index];
    }
};

} /* namespace dogm */
