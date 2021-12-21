// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef LASER_TO_MEAS_GRID_H
#define LASER_TO_MEAS_GRID_H

#include "dogm/dogm_types.h"
#include <memory>
#include <vector>

namespace dogm {

    class LaserMeasurementGrid
    {
    public:
        struct Params
        {
            float max_range;
            float resolution;
            float fov;
        };

        LaserMeasurementGrid(const Params& lidar_params, float grid_length, float grid_resolution);
        ~LaserMeasurementGrid();

        dogm::MeasurementCell* generateGrid(const std::vector<float>& measurements);

    private:
        dogm::MeasurementCell* meas_grid;
        int grid_size;
        float grid_resolution;

        Params laser_params;

        int polar_width;
        int polar_height;
        float2* polar_grid;
        float theta_min;
    };

}

#endif  // LASER_TO_MEAS_GRID_H