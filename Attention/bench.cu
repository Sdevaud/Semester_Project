#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_utils.cuh"
#include "flashAFP.cuh"

float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

void scale_matrix_by_sqrtN(float* mat, int rows, int cols, int d) {
    float scale = rsqrtf((float)d);

    for (int i = 0; i < rows * cols; ++i) {
        mat[i] *= scale;
    }
}

void matmul_S(const float* A, const float* Bt, float* C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * Bt[j * N + k];  // Bt[j][k] = B[k][j]
            }
            C[i * P + j] = sum;
        }
    }

    scale_matrix_by_sqrtN(C, M, P, N);
}

void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


void softmax_matrix(const float* input, float* output, int M, int N) {
    for (int i = 0; i < M; i++) {
        float xmax = input[i * N];
        for (int j = 1; j < N; j++) {
            float val = input[i * N + j];
            if (val > xmax) xmax = val;
        }

        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float val = expf(input[i * N + j] - xmax);
            output[i * N + j] = val;
            sum += val;
        }

        for (int j = 0; j < N; j++) {
            output[i * N + j] /= sum;
        }
    }
}


void test_value(const float* A, const float* B, int M, int N) {

    bool result = true;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(A[i * N + j]-B[i * N + j]) > 0.0001) 
                result = false;
        }
    }

    printf("the test is: %s\n", result ? "true" : "false");
    return;
}

void print_matrix(const char* name, const float* mat, int rows, int cols, int precision) {
    printf("\n%s:\n", name);
    char format[16];
    snprintf(format, sizeof(format), "%%.%df ", precision);  // e.g. "%.2f "

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf(format, mat[i * cols + j]);
        }
        printf("\n");
    }
}

void test_compute_S() {
    int M = 1024;
    int N = 2048;
    int matsize = M * N;
    int size_QK = M * N * sizeof(float);
    int size_S = M * M * sizeof(float);

    float *Q = (float*)malloc(size_QK);
    float *K = (float*)malloc(size_QK);
    float *O = (float*)malloc(size_S); 

    for (int i = 0; i < matsize; ++i) { 
        Q[i] = random_normal_clamped(-10, 10);
        K[i] = random_normal_clamped(-10, 10);
    }

    //print_matrix("Q", Q, M, N, 1);
    //print_matrix("K", K, M, N, 1);

    float *Qd, *Kd, *Od;
    CUDA_CHECK(cudaMalloc(&Qd, size_QK));
    CUDA_CHECK(cudaMalloc(&Kd, size_QK));
    CUDA_CHECK(cudaMalloc(&Od, size_S));

    CUDA_CHECK(cudaMemcpy(Qd, Q, size_QK, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Kd, K, size_QK, cudaMemcpyHostToDevice));

    dim3 block_size(32);
    dim3 grid_size(M, M);
    compute_S<<<grid_size, block_size>>>(Qd, Kd, Od, M, N);

    CUDA_CHECK(cudaMemcpy(O, Od, size_S, cudaMemcpyDeviceToHost)); 


    //print_matrix("O", O, M, M, 3);

    float *test = (float*)malloc(size_S);
    matmul_S(Q, K, test, M, N, M);
    
    //print_matrix("test", test, M, M, 3);

    test_value(O, test, M, M);
    
    
    free(Q);
    free(K);
    free(O);
    free(test);
    cudaFree(Qd);
    cudaFree(Kd);
    cudaFree(Od);

}

void easy_flash_attention(float* Q, float* K, float* V, float* S, float* P, float* O, int M, int N) {
    matmul_S(Q, K, S, M, N, M);
    softmax_matrix(S, P, M, M);
    matmul(P, V, O, M, M, N);
}

int main() {

    //test_compute_S();

    int M = 1024;
    int N = 2048;
    int matsize = M * N;
    int size_QKV = M * N * sizeof(float);
    int size_S = M * M * sizeof(float);

    float *Q = (float*)malloc(size_QKV);
    float *K = (float*)malloc(size_QKV);
    float *V = (float*)malloc(size_QKV); 
    float *S = (float*)malloc(size_S);
    float *P = (float*)malloc(size_S);
    float *O = (float*)malloc(size_QKV);

    for (int i = 0; i < matsize; ++i) { 
        Q[i] = random_normal_clamped(-10, 10);
        K[i] = random_normal_clamped(-10, 10);
        V[i] = random_normal_clamped(-10, 10);
    }

    
    float *Qd, *Kd, *Vd, *Sd, *Pd, *Od;

    // we compute Q @ K.T = S
    CUDA_CHECK(cudaMalloc(&Qd, size_QKV));
    CUDA_CHECK(cudaMalloc(&Kd, size_QKV));
    CUDA_CHECK(cudaMalloc(&Sd, size_S));

    CUDA_CHECK(cudaMemcpy(Qd, Q, size_QKV, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Kd, K, size_QKV, cudaMemcpyHostToDevice));

    dim3 block_size_S(32);
    dim3 grid_size_S(M, M);
    compute_S<<<grid_size_S, block_size_S>>>(Qd, Kd, Sd, M, N);
    CUDA_CHECK(cudaMemcpy(S, Sd, size_S, cudaMemcpyDeviceToHost)); 


    // we compute softmax(S) = P
    CUDA_CHECK(cudaMalloc(&Sd, size_S));
    CUDA_CHECK(cudaMalloc(&Pd, size_S));

    dim3 block_size_P(1024);
    dim3 grid_size_P(M);
    softmax_kernel_3<<<grid_size_P, block_size_P>>>(Sd, Pd, M, N);
    CUDA_CHECK(cudaMemcpy(P, Pd, size_S, cudaMemcpyDeviceToHost));

    //we compute flash attention P @ V = O
    CUDA_CHECK(cudaMalloc(&Pd, size_S));
    CUDA_CHECK(cudaMalloc(&Vd, size_QKV));

    CUDA_CHECK(cudaMemcpy(Pd, P, size_S, cudaMemcpyHostToDevice));

    dim3 block_size_O(32);
    dim3 grid_size_O(M, M);
    compute_O<<<grid_size_O, block_size_O>>>(Pd, Vd, Od, M, N, M);
    CUDA_CHECK(cudaMemcpy(O, Od, size_QKV, cudaMemcpyDeviceToHost));

    float *test = (float*)malloc(size_QKV);
    easy_flash_attention(Q, K, V, S, P, test, M, N);
    test_value(O, test, M, N);


    
    free(Q);
    free(K);
    free(V);
    free(S);
    free(P);
    free(O);
    free(test);
    cudaFree(Qd);
    cudaFree(Kd);
    cudaFree(Vd);
    cudaFree(Sd);
    cudaFree(Pd);
    cudaFree(Od);

    
}