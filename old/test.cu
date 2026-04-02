#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void multiply(float* a, float* b, float* c, int length) {
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < length)
    {
        c[workIndex] = a[workIndex] * b[workIndex];
    }
}

void initArray(float* a, int length) {
    for (int i = 0; i < length; i++) {
        a[i] = i + 1;
    }
}

void unifiedMemory(int length) {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;

    cudaMallocManaged(&a, length*sizeof(float));
    cudaMallocManaged(&b, length*sizeof(float));
    cudaMallocManaged(&c, length*sizeof(float));

    initArray(a, length);
    initArray(b, length);

    int threads = 128;
    int blocks = cuda::ceil_div(length, threads);

    multiply<<<blocks, threads>>>(a, b, c, length);
    cudaDeviceSynchronize();

    for (int i = 0; i < length; i++){
        printf("%f\n", c[i]);
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

int main() {
    int length = 100;
    unifiedMemory(length);		
    return 0;
}