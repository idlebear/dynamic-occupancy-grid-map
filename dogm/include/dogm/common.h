// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

template <typename T>
inline void accumulate(T* arr, thrust::device_vector<T>& result)
{
    thrust::device_ptr<T> ptr(arr);
    thrust::inclusive_scan(ptr, ptr + result.size(), result.begin());
}

template <typename T>
inline void accumulate(thrust::device_vector<T>& arr, thrust::device_vector<T>& result)
{
    thrust::inclusive_scan(arr.begin(), arr.end(), result.begin());
}

template <typename T>
struct _normalize_fn {
    T normalizer;
    _normalize_fn( T val ) {
        normalizer = val;
    }
    __device__ __host__ T operator()(const T v) const { return v / normalizer; }
};

template <typename T>
void normalize(T* arr, int num_elements)
{
    thrust::device_ptr<T> ptr(arr);
    auto sum_total = thrust::reduce( ptr, ptr+num_elements);
    thrust::transform(ptr, ptr + num_elements, ptr, _normalize_fn<T>(sum_total));
}


template <typename T>
inline __device__ __host__ T subtract(T* accum_array, int start_idx, int end_idx)
{
    if (start_idx == 0)
    {
        return accum_array[end_idx];
    }
    return accum_array[end_idx] - accum_array[start_idx - 1];
}

template <typename T>
inline __device__ __host__ T clamp(T a, T lower, T upper)
{
    return max(min(a, upper), lower);
}

inline __host__ __device__ unsigned int hash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}
