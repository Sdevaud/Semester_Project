#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "cuda_utils.cuh"


__global__ void compute_S(float* Qd, float* Kd, float* Sd, int M, int N) {
    int i = blockIdx.y;
    int j = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float smem[1024];
    float partial_sum = 0.0f;

    for (int k = tid; k < N; k += blockDim.x) {
        float q = Qd[i * N + k];
        float k_val = Kd[j * N + k]; 
        partial_sum += q * k_val;
    }

    smem[tid] = partial_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float scale = rsqrtf((float)N);
        Sd[i * M + j] = smem[0] * scale; 
    }
}

__global__ void compute_O(float* Sd, float* Vd, float* Od, int M, int N, int D) {
    int i = blockIdx.y;
    int j = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float smem[1024];
    float partial_sum = 0.0f;

    for (int k = tid; k < N; k += blockDim.x) {
        float Sx = Sd[i * N + k];    
        float Vx = Vd[k * D + j];    
        partial_sum += Sx * Vx;
    }

    smem[tid] = partial_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        Od[i * D + j] = smem[0];  
    }
}


__global__ void softmax_kernel_3(float* xd, float* resd, int M, int N) {
    // max and norm reduction will happen in shared memory (static)
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    // number of threads in a warp
    unsigned int warp_size = 32;
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    // each thread will have its own local max
    // we store it in shared memory for reduction
    // smem[tid] = local_max;
    // __syncthreads();

    // warp level reduction using XOR shuffle ('exchanges' the values in the threads)
    // note: if there are 256 threads in one block (8 warps of 32 threads each)
    // the following for loop reduces the value in all the 8 warps
    // the 8 warps contain the 8 maximum values of the 32 threads that reside in those warps
    // float val = smem[tid];
    float val = local_max;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    // when blockDim is greater than 32, we need to do a block level reduction
    // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
    // the global max will be stored in the first warp
    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            // which warp are we at?
            // store the value in its first thread index
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        // first warp will do global reduction only
        // this is possible because we stored the values in the shared memory
        // so the threads in the first warp will read from it and then reduce
        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : -INFINITY;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        // this is for when the number of threads in a block are not
        // greater than the warp size, in that case we already reduced
        // so we can store the value
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    // we got the global row max now
    float row_max = smem[0];
    __syncthreads();

    // each thread will have its own local_norm
    // we will store the corrected local_norm in the shared memory
    // smem[tid] = local_norm * expf(local_max - row_max);
    // __syncthreads();

    // same reduction algorithm as above, but instead of max reduction
    // we do a sum reduction i.e. we accumulate the values
    // val = smem[tid];
    val = local_norm * expf(local_max - row_max);
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        // first warp will do global reduction
        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : 0.0f;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    float row_norm = smem[0];
    __syncthreads();

    // finally, compute softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

/*
Runs the online softmax kernel: `id = 3`
*/
float run_kernel_3(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // grid size and block size for this kernel
    // change as necessary
    dim3 block_size(1024);
    dim3 grid_size(M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_3<<<grid_size, block_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time for shfl : %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

