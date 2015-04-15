//
//  kernel.metal
//  GPUExample
//
//  Created by Mateusz Buda on 12/04/15.
//  Copyright (c) 2015 inFullMobile. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

constant int THREADGROUP_SIZE = 512;

kernel void reduce1(const device int *array [[ buffer(0) ]],
                   volatile device atomic_int *result [[ buffer(1) ]],
                   uint id [[ thread_position_in_grid ]],
                   uint tid [[ thread_index_in_threadgroup ]],
                   uint bid [[ threadgroup_position_in_grid ]],
                   uint blockDim [[ threads_per_threadgroup ]]) {
    
    threadgroup int shared_memory[THREADGROUP_SIZE];
    
    uint i = bid * blockDim + tid;
    
    shared_memory[tid] = array[i];
    
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

kernel void reduce2(const device int *array [[ buffer(0) ]],
                    volatile device atomic_int *result [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]],
                    uint tid [[ thread_index_in_threadgroup ]],
                    uint bid [[ threadgroup_position_in_grid ]],
                    uint blockDim [[ threads_per_threadgroup ]]) {
    
    threadgroup int shared_memory[THREADGROUP_SIZE];
    
    uint i = bid * blockDim + tid;
    
    shared_memory[tid] = array[i];
    
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

kernel void reduce3(const device int *array [[ buffer(0) ]],
                    volatile device atomic_int *result [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]],
                    uint tid [[ thread_index_in_threadgroup ]],
                    uint bid [[ threadgroup_position_in_grid ]],
                    uint blockDim [[ threads_per_threadgroup ]]) {
    
    threadgroup int shared_memory[THREADGROUP_SIZE];
    
    uint i = bid * blockDim + tid;
    
    shared_memory[tid] = array[i];
    
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

kernel void reduce4(const device int *array [[ buffer(0) ]],
                   volatile device atomic_int *result [[ buffer(1) ]],
                   uint id [[ thread_position_in_grid ]],
                   uint tid [[ thread_index_in_threadgroup ]],
                   uint bid [[ threadgroup_position_in_grid ]],
                   uint blockDim [[ threads_per_threadgroup ]]) {
    
    threadgroup int shared_memory[THREADGROUP_SIZE];
    
    uint i = bid * (blockDim * 2) + tid;
    
    shared_memory[tid] = array[i] + array[i + blockDim];
    
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