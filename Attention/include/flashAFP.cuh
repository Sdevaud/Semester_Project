#ifndef FLASHAFP_ATTENTION
#define FLASHAFP_ATTENTION



__global__ void softmax_kernel_3(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
float run_kernel_3(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
__global__ void compute_S(float* Qd, float* Kd, float* Sd, int M, int N);
__global__ void compute_O(float* Sd, float* Vd, float* Od, int M, int N, int D);

#endif // end FLASHAFP_ATTENTION