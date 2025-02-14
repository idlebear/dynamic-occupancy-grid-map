// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/mass_update.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__device__ float predict_free_mass(float grid_cell_free_mass, float m_occ_pred, float alpha = 0.9)
{
    return min(alpha * grid_cell_free_mass, 1.0f - m_occ_pred);
}

__device__ float2 update_masses(float m_occ_pred, float m_free_pred, const MeasurementCellsSoA meas_cells, int meas_idx)
{
    float unknown_pred = 1.0f - m_occ_pred - m_free_pred;
    float meas_unknown = 1.0f - meas_cells.free_mass[meas_idx] - meas_cells.occ_mass[meas_idx];
    float K = m_free_pred * meas_cells.occ_mass[meas_idx] + m_occ_pred * meas_cells.free_mass[meas_idx];

    float occ_mass =
        (m_occ_pred * meas_unknown + unknown_pred * meas_cells.occ_mass[meas_idx] + m_occ_pred * meas_cells.occ_mass[meas_idx]) / (1.0f - K);
    float free_mass =
        (m_free_pred * meas_unknown + unknown_pred * meas_cells.free_mass[meas_idx] + m_free_pred * meas_cells.free_mass[meas_idx]) / (1.0f - K);

    return make_float2(occ_mass, free_mass);
}

__device__ float separate_newborn_part(float m_occ_pred, float m_occ_up, float p_B)
{
    return (m_occ_up * p_B * (1.0f - m_occ_pred)) / (m_occ_pred + p_B * (1.0f - m_occ_pred));
}

__device__ void store_values(float rho_b, float rho_p, float m_free_up, float m_occ_up, float m_occ_pred,
                             GridCellsSoA grid_cell_array, int i)
{
    grid_cell_array.pers_occ_mass[i] = rho_p;
    grid_cell_array.new_born_occ_mass[i] = rho_b;
    grid_cell_array.free_mass[i] = m_free_up;
    grid_cell_array.occ_mass[i] = m_occ_up;
    grid_cell_array.pred_occ_mass[i] = m_occ_pred;
}

__device__ void normalize_weights(float* __restrict__ weight_array, int start_idx,
                                  int end_idx, float occ_pred, float prev_occ_pred)
{
    for (int i = start_idx; i < end_idx + 1; i++) {
        weight_array[i] = weight_array[i] * occ_pred / prev_occ_pred;
    }
}

__global__ void gridCellPredictionUpdateKernel(GridCellsSoA grid_cell_array, ParticlesSoA particle_array,
                                               float* __restrict__ weight_array,
                                               const float* __restrict__ weight_array_accum,
                                               const MeasurementCellsSoA meas_cell_array,
                                               float* __restrict__ born_masses_array, float p_S, float p_B, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        int start_idx = grid_cell_array.start_idx[i];
        int end_idx = grid_cell_array.end_idx[i];

        float  m_occ_pred = 0.0f;  // No prediction information from internal state
        if (start_idx != -1) {
            m_occ_pred = subtract(weight_array_accum, start_idx, end_idx);

            // truncate to p_S probability (from dogma py)
            if( m_occ_pred > p_S ) {
                normalize_weights( particle_array.weight, start_idx, end_idx, p_S, m_occ_pred );
                m_occ_pred = p_S;
            }
        }

        float m_free_pred = predict_free_mass(grid_cell_array.free_mass[i], m_occ_pred);
        float2 masses_up = update_masses(m_occ_pred, m_free_pred, meas_cell_array, i);
        float rho_b = separate_newborn_part(m_occ_pred, masses_up.x, p_B);
        float rho_p = masses_up.x - rho_b;
        born_masses_array[i] = rho_b;

        store_values(rho_b, rho_p, masses_up.y, masses_up.x, m_occ_pred, grid_cell_array, i);
    }
}

} /* namespace dogm */
