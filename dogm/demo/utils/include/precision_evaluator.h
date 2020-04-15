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

#ifndef PRECISION_EVALUATOR_H
#define PRECISION_EVALUATOR_H

#include "dbscan.h"
#include "dogm/dogm_types.h"
#include "simulator.h"

#include <vector>

struct PointWithVelocity
{
    float x{0.0f};
    float y{0.0f};
    float v_x{0.0f};
    float v_y{0.0f};
};

class PrecisionEvaluator
{
public:
    PrecisionEvaluator(const SimulationData _sim_data, const float _resolution);
    void evaluateAndStoreStep(int simulation_step_index, const std::vector<Point<dogm::GridCell>>& cells_with_velocity,
                              bool print_current_precision = false);
    void printSummary();

private:
    void accumulateErrors(const PointWithVelocity& error);
    PointWithVelocity computeClusterMean(const Cluster<dogm::GridCell>& cluster);

    SimulationData sim_data;
    float resolution;
    PointWithVelocity cumulative_error;
    int number_of_detections;
    int number_of_unassigned_detections;
};

#endif  // PRECISION_EVALUATOR_H
