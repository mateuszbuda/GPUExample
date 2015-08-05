//
//  kernel.metal
//  GPUExample
//
//  Created by Mateusz Buda on 12/04/15.
//  Copyright (c) 2015 inFullMobile. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

constant int THREADGROUP_SIZE = 256;

/* map */

kernel void map(const device int *array [[ buffer(0) ]],
                device int *result [[ buffer(1) ]],
                uint id [[ thread_position_in_grid ]],
                uint tid [[ thread_index_in_threadgroup ]],
                uint bid [[ threadgroup_position_in_grid ]],
                uint blockDim [[ threads_per_threadgroup ]]) {
    
    uint i = bid * blockDim + tid;
    
    result[i] = int(cos(float(array[i])));
}

/* naive reduction */

// (kernel | vertex | fragment)
kernel void reduce1(const device int *array [[ buffer(0) ]],
                   volatile device atomic_int *result [[ buffer(1) ]],
                   uint id [[ thread_position_in_grid ]],
                   uint tid [[ thread_index_in_threadgroup ]],
                   uint bid [[ threadgroup_position_in_grid ]],
                   uint blockDim [[ threads_per_threadgroup ]]) {
    
    threadgroup int shared_memory[THREADGROUP_SIZE];
    
    uint i = bid * blockDim + tid;
    
    shared_memory[tid] = int(cos(float(array[i])));
    
    threadgroup_barrier(mem_flags::mem_none);
    
    // reduction in shared memory
    for (uint s = 1; s < blockDim; s *= 2) {
        if (tid % (2 * s) == 0) {
            shared_memory[tid] += shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    
    if (0 == tid) {
        atomic_fetch_add_explicit(result, shared_memory[0], memory_order_relaxed);
    }
}

/* changed thred id performing reduction */

kernel void reduce2(const device int *array [[ buffer(0) ]],
                    volatile device atomic_int *result [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]],
                    uint tid [[ thread_index_in_threadgroup ]],
                    uint bid [[ threadgroup_position_in_grid ]],
                    uint blockDim [[ threads_per_threadgroup ]]) {
    
    threadgroup int shared_memory[THREADGROUP_SIZE];
    
    uint i = bid * blockDim + tid;
    
    shared_memory[tid] = int(cos(float(array[i])));
    
    threadgroup_barrier(mem_flags::mem_none);
    
    // reduction in shared memory
    for (uint s = 1; s < blockDim; s *= 2) {
        uint index = 2 * s * tid;
        
        if (index < blockDim) {
            shared_memory[index] += shared_memory[index + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    
    if (0 == tid) {
        atomic_fetch_add_explicit(result, shared_memory[0], memory_order_relaxed);
    }
}

/* connected memory space */

kernel void reduce3(const device int *array [[ buffer(0) ]],
                    volatile device atomic_int *result [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]],
                    uint tid [[ thread_index_in_threadgroup ]],
                    uint bid [[ threadgroup_position_in_grid ]],
                    uint blockDim [[ threads_per_threadgroup ]]) {
    
    threadgroup int shared_memory[THREADGROUP_SIZE];
    
    uint i = bid * blockDim + tid;
    
    shared_memory[tid] = int(cos(float(array[i])));
    
    threadgroup_barrier(mem_flags::mem_none);
    
    // reduction in shared memory
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] += shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    
    if (0 == tid) {
        atomic_fetch_add_explicit(result, shared_memory[0], memory_order_relaxed);
    }
}

/* halved number of blocks */

kernel void reduce4(const device int *array [[ buffer(0) ]],
                   volatile device atomic_int *result [[ buffer(1) ]],
                   uint id [[ thread_position_in_grid ]],
                   uint tid [[ thread_index_in_threadgroup ]],
                   uint bid [[ threadgroup_position_in_grid ]],
                   uint blockDim [[ threads_per_threadgroup ]]) {
    
    threadgroup int shared_memory[THREADGROUP_SIZE];
    
    uint i = bid * (blockDim * 2) + tid;
    
    shared_memory[tid] = int(cos(float(array[i]))) + int(cos(float(array[i + blockDim])));
    
    threadgroup_barrier(mem_flags::mem_none);
    
    // reduction in shared memory
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] += shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
    
    if (0 == tid) {
        atomic_fetch_add_explicit(result, shared_memory[0], memory_order_relaxed);
    }
}

// source for parallel reduction:
// Optimizing Parallel Reduction in CUDA by Mark Harris
// https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
