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
            float angle_increment;
            float stddev_range;
        };

        LaserMeasurementGrid(const Params& lidar_params, float grid_length, float grid_resolution);
        ~LaserMeasurementGrid();

        dogm::MeasurementCellsSoA generateGrid(const std::vector<float>& measurements, float angle_offset);

    private:
        dogm::MeasurementCellsSoA meas_grid;
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