#include "simulation.h"
#include "config.h"
#include "globals.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <iostream>
#include <cmath>

float* microDensity  = nullptr;
float* microDensity2 = nullptr;
float* macroDensity  = nullptr;
float* xVelocity     = nullptr;
float* yVelocity     = nullptr;
float* smoke         = nullptr;
float* smoke2        = nullptr;
int*   barrier       = nullptr;

bool showSmoke = false;

__constant__ int   d_directionXVector[9];
__constant__ int   d_directionYVector[9];
__constant__ float d_directionWeights[9];
__constant__ int   d_oppositeDirection[9];

__constant__ float d_transformationMatrix[9][9];
__constant__ float d_transformationMatrixInverse[9][9];
__constant__ float d_relaxationRates[9];

__constant__ float d_tau;


__global__ void collisionMRT(float* microDensity, float* macroDensity,
                             float* xVelocity, float* yVelocity,
                             int* barrier, int cellCount)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= cellCount || barrier[n]) return;

    float localDensity[9];
    for (int d = 0; d < 9; d++)
        localDensity[d] = microDensity[d * cellCount + n];

    float totalDensity = 0.0f, vX = 0.0f, vY = 0.0f;
    for (int d = 0; d < 9; d++) {
        totalDensity += localDensity[d];
        vX += d_directionXVector[d] * localDensity[d];
        vY += d_directionYVector[d] * localDensity[d];
    }
    vX /= totalDensity;
    vY /= totalDensity;

    macroDensity[n] = totalDensity;
    xVelocity[n]    = vX;
    yVelocity[n]    = vY;

    // Transform to moment space
    float m[9] = {0};
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            m[i] += d_transformationMatrix[i][j] * localDensity[j];

    // Equilibrium moments
    float uSq = vX * vX + vY * vY;
    float meq[9];
    meq[0] =  totalDensity;
    meq[1] = -2.0f * totalDensity + 3.0f * totalDensity * uSq;
    meq[2] =  totalDensity        - 3.0f * totalDensity * uSq;
    meq[3] =  totalDensity * vX;
    meq[4] = -totalDensity * vX;
    meq[5] =  totalDensity * vY;
    meq[6] = -totalDensity * vY;
    meq[7] =  totalDensity * (vX * vX - vY * vY);
    meq[8] =  totalDensity * vX * vY;

    // Relax
    for (int i = 0; i < 9; i++)
        m[i] -= d_relaxationRates[i] * (m[i] - meq[i]);

    // Back-transform
    float f[9] = {0};
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            f[i] += d_transformationMatrixInverse[i][j] * m[j];

    for (int d = 0; d < 9; d++)
        microDensity[d * cellCount + n] = f[d];
}

__global__ void streaming(float* microDensity, float* microDensity2,
                          int* barrier, int xSize, int ySize, int cellCount)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= cellCount) return;

    int x = n % xSize;
    int y = n / xSize;

    if (barrier[n]) {
        for (int d = 0; d < 9; d++)
            microDensity2[d * cellCount + n] = microDensity[d * cellCount + n];
        return;
    }

    for (int d = 0; d < 9; d++) {
        int prevX = x - d_directionXVector[d];
        int prevY = y - d_directionYVector[d];

        bool valid = (prevX >= 0 && prevX < xSize &&
                      prevY >= 0 && prevY < ySize &&
                      !barrier[prevY * xSize + prevX]);

        microDensity2[d * cellCount + n] = valid
            ? microDensity[d * cellCount + prevY * xSize + prevX]
            : microDensity[d_oppositeDirection[d] * cellCount + n];  // bounce-back
    }
}

__global__ void advectSmoke(float* smoke, float* smoke2,
                            float* xVelocity, float* yVelocity,
                            int* barrier, int xSize, int ySize,
                            int cellCount, float advScale)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= cellCount) return;

    if (barrier[n]) {
        smoke2[n] = smoke[n];
        return;
    }

    int x = n % xSize;
    int y = n / xSize;

    float prevX = fmaxf(0.0f, fminf((float)(xSize - 1), x - xVelocity[n] * advScale));
    float prevY = fmaxf(0.0f, fminf((float)(ySize - 1), y - yVelocity[n] * advScale));

    int x0 = (int)prevX, x1 = min(x0 + 1, xSize - 1);
    int y0 = (int)prevY, y1 = min(y0 + 1, ySize - 1);
    float sx = prevX - x0;
    float sy = prevY - y0;

    float s0 = smoke[y0 * xSize + x0] * (1 - sx) + smoke[y0 * xSize + x1] * sx;
    float s1 = smoke[y1 * xSize + x0] * (1 - sx) + smoke[y1 * xSize + x1] * sx;
    smoke2[n] = s0 * (1 - sy) + s1 * sy;
}


void launchCollisionMRT() {
    collisionMRT<<<blocks, threads>>>(microDensity, macroDensity,
                                      xVelocity, yVelocity, barrier, cellCount);
}

void launchStreaming() {
    streaming<<<blocks, threads>>>(microDensity, microDensity2,
                                   barrier, xSize, ySize, cellCount);
}

void launchAdvectSmoke() {
    advectSmoke<<<blocks, threads>>>(smoke, smoke2, xVelocity, yVelocity,
                                     barrier, xSize, ySize, cellCount,
                                     smokeAdvectionScale);
}


void parseBarrierFile(const std::string& filename, int* barrier) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return;
    }
    int x, y;
    while (infile >> x >> y)
        barrier[y * xSize + x] = 1;
}

void initialize() {
    cudaMallocManaged(&microDensity,  9 * cellCount * sizeof(float));
    cudaMallocManaged(&microDensity2, 9 * cellCount * sizeof(float));
    cudaMallocManaged(&smoke, cellCount * sizeof(float));
    cudaMallocManaged(&smoke2, cellCount * sizeof(float));
    cudaMallocManaged(&macroDensity, cellCount * sizeof(float));
    cudaMallocManaged(&xVelocity, cellCount * sizeof(float));
    cudaMallocManaged(&yVelocity, cellCount * sizeof(float));
    cudaMallocManaged(&barrier, cellCount * sizeof(int));

    cudaMemset(microDensity, 0, 9 * cellCount * sizeof(float));
    cudaMemset(microDensity2, 0, 9 * cellCount * sizeof(float));
    cudaMemset(smoke, 0, cellCount * sizeof(float));
    cudaMemset(smoke2, 0, cellCount * sizeof(float));
    cudaMemset(macroDensity, 0, cellCount * sizeof(float));
    cudaMemset(xVelocity, 0, cellCount * sizeof(float));
    cudaMemset(yVelocity, 0, cellCount * sizeof(float));
    cudaMemset(barrier, 0, cellCount * sizeof(int));

    cudaMemcpyToSymbol(d_directionXVector, directionXVector, sizeof(directionXVector));
    cudaMemcpyToSymbol(d_directionYVector, directionYVector, sizeof(directionYVector));
    cudaMemcpyToSymbol(d_directionWeights, directionWeights, sizeof(directionWeights));
    cudaMemcpyToSymbol(d_oppositeDirection, oppositeDirection, sizeof(oppositeDirection));
    cudaMemcpyToSymbol(d_transformationMatrix, transformationMatrix, sizeof(transformationMatrix));
    cudaMemcpyToSymbol(d_transformationMatrixInverse, transformationMatrixInverse, sizeof(transformationMatrixInverse));
    cudaMemcpyToSymbol(d_relaxationRates, relaxationRates, sizeof(relaxationRates));
    cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));

    parseBarrierFile("../outputs/airfoil_coordinates.txt", barrier);

    for (int x = 0; x < xSize; x++) {
        barrier[x] = 1;
        barrier[(ySize - 1) * xSize + x] = 1;
    }

    /*for (int y = ySize * 2 / 5; y < ySize * 3 / 5; y++){ 
        barrier[y * xSize + xSize / 3] = 1;
    }*/

    int count = 0;
    for (int i = 0; i < cellCount; i++) {
        if (barrier[i]) {
            count++;
        } 
    }
        
    std::cout << "Barrier cell count: " << count << "\n";

    float uSq = inletVelocity * inletVelocity;
    for (int n = 0; n < cellCount; n++) {
        if (barrier[n]) continue;
        xVelocity[n] = inletVelocity;
        yVelocity[n] = 0.0f;
        for (int d = 0; d < 9; d++) {
            float dot = directionXVector[d] * inletVelocity;
            microDensity[d * cellCount + n] =
                directionWeights[d] * (1.0f + 3.0f * dot + 4.5f * dot * dot - 1.5f * uSq);
        }
    }
}
