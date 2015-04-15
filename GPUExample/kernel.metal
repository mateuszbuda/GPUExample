//
//  kernel.metal
//  GPUExample
//
//  Created by Mateusz Buda on 12/04/15.
//  Copyright (c) 2015 inFullMobile. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

constant int WARP_SIZE = 512;

kernel void reduce(const device int *array [[ buffer(0) ]],
                   volatile device atomic_int *result [[ buffer(1) ]],
                   uint id [[ thread_position_in_grid ]],
                   uint tid [[ thread_index_in_threadgroup ]],
                   uint bid [[ threadgroup_position_in_grid ]],
                   uint blockDim [[ threads_per_threadgroup ]]) {
    threadgroup int shared_memory[WARP_SIZE];
    
    uint i = bid * (blockDim * 2) + tid;
    
    shared_memory[tid] = array[i] + array[i + blockDim];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // reduction in shared memory
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] += shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (0 == tid) {
        atomic_fetch_add_explicit(result, shared_memory[0], memory_order_relaxed);
    }
}