# GPUExample
GPGPU Example with Apple's Metal API

## Project overview
### KernelSelectionController.swift
Table View with available kernels to compare CPU and GPU performance

### ViewController.swift
Performs CPU and GPU computations. Shows execution times.

### kernel.metal
Kernels that are executed on GPU with Metal API.

#### map
Simple map that applies cosine function to each element of input array.

#### reduce1
Naive parallel reduction (computes sum of cosine of each input array element).
![reduce1](https://raw.githubusercontent.com/mateuszbuda/GPUExample/master/reduce1.png)

#### reduce2
Changed threads performing reduction.
![reduce2](https://raw.githubusercontent.com/mateuszbuda/GPUExample/master/reduce2.png)

#### reduce3
Accessing connected memory space.
![reduce3](https://raw.githubusercontent.com/mateuszbuda/GPUExample/master/reduce3.png)

#### reduce4
The same as in reduce3 but first reduction step is performed when copying data to shared memory, so we need half the number of threads that we needed in the previous reduce versions.

## NOTICE
Graphics presenting reduction optimization steps source:
Optimizing Parallel Reduction in CUDA by Mark Harris
https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
