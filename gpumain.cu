#include <SFML/Graphics.hpp>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>

float* microDensity;
float* microDensity2;
float* macroDensity;
float* xVelocity;
float* yVelocity;
float* smoke;
float* smoke2;
int* barrier;

const int scale = 10;
const int xSize = 90 * scale;
const int ySize = 60 * scale;
const int cellCount = xSize * ySize;
const int renderScale = 2;
const int stepsPerFrame = 5;
const float tau = 0.6;
const float inletVelocity = 0.025f;
const float velocityRange = inletVelocity / 2;
const float smokeAdvectionScale = 20;
bool showSmoke = false;

int directionXVector[] = {0, 0, 0, 1, -1, 1, -1, -1, 1};
int directionYVector[] = {0, 1, -1, 0, 0, 1, -1, 1, -1};
float directionWeights[] = {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36};
int oppositeDirection[] = {0, 2, 1, 4, 3, 7, 5, 6, 8};

__constant__ int d_directionXVector[9];
__constant__ int d_directionYVector[9];
__constant__ float d_directionWeights[9];
__constant__ int d_oppositeDirection[9];

__constant__ float d_tau;

const int threads = 128;
const int blocks = (cellCount + threads - 1) / threads;

void parseBarrierFile(const std::string& filename, int* barrier) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    int x, y;
    while (infile >> x >> y) {
        barrier[y * xSize + x] = 1;  // Set the barrier
    }

    infile.close();
}

void initialize() {
    cudaMallocManaged(&microDensity, 9 * cellCount * sizeof(float));
    cudaMallocManaged(&microDensity2, 9 * cellCount * sizeof(float));
    cudaMallocManaged(&smoke, cellCount * sizeof(float));
    cudaMallocManaged(&smoke2, cellCount * sizeof(float));
    cudaMallocManaged(&macroDensity, cellCount * sizeof(float));
    cudaMallocManaged(&barrier, cellCount * sizeof(int));
    cudaMallocManaged(&xVelocity, cellCount * sizeof(float));
    cudaMallocManaged(&yVelocity, cellCount * sizeof(float));

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

    cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));

    /*for (int n = 0; n < xSize; n++) {
        barrier[n] = 1;
        barrier[(ySize - 1) * xSize + n] = 1;
    }*/

    parseBarrierFile("outputs/airfoil_coordinates.txt", barrier);

    int count = 0;
    for (int i = 0; i < cellCount; i++)
        if (barrier[i]) count++;

    std::cout << "Barrier count = " << count << std::endl;


    /*for (int n = 40; n < 60; n++) {
        barrier[n * xSize + 50] = 1;
    }*/

    float uSquared = inletVelocity * inletVelocity;
    for (int n = 0; n < cellCount; n++) {
        if (barrier[n] == 1) {
            continue;
        }

        xVelocity[n] = inletVelocity;
        yVelocity[n] = 0;
        for (int d = 0; d < 9; d++) {
            float dotProduct = directionXVector[d] * inletVelocity;
            microDensity[d * cellCount + n] = directionWeights[d] * (1 + 3 * dotProduct + 4.5 * dotProduct * dotProduct - 1.5 * uSquared);
        }
    }
}

sf::Color velocityToColor(int x, int y) {
    float vX = xVelocity[y * xSize + x];
    float vY = yVelocity[y * xSize + x];
    float v = std::sqrt(vX * vX + vY * vY);

    float t = (v - inletVelocity) / velocityRange;

    t = std::clamp(t, -1.0f, 1.0f);

    uint8_t r, g, b;

    if (t < 0.0f) {
        float s = t + 1.0f;
        r = static_cast<uint8_t>(s * 255);
        g = static_cast<uint8_t>(s * 255);
        b = 255;
    } else {
        float s = t;
        r = 255;
        g = static_cast<uint8_t>((1.0f - s) * 255);
        b = static_cast<uint8_t>((1.0f - s) * 255);
    }

    return sf::Color(r, g, b);
}

__global__ void collision(float* microDensity, float* macroDensity, float* xVelocity, float* yVelocity, int* barrier, int cellCount) {
    int n = threadIdx.x + blockIdx.x*blockDim.x;
    if (n >= cellCount || barrier[n] == 1) {
        return;
    }
    
    float totalDensity = 0.0f;
    float vX = 0.0f;
    float vY = 0.0f;

    for (int d = 0; d < 9; d++) {
        float f = microDensity[d * cellCount + n];
        totalDensity += f;
        vX += d_directionXVector[d] * f;
        vY += d_directionYVector[d] * f;
    }

    if (totalDensity > 1e-6f) {
        vX /= totalDensity;
        vY /= totalDensity;
    }

    xVelocity[n] = vX;
    yVelocity[n] = vY;
    macroDensity[n] = totalDensity;    

    float uSquared = vX * vX + vY * vY;
    for (int d = 0; d < 9; d++) {
        float dotProduct = d_directionXVector[d] * vX + d_directionYVector[d] * vY;
        float equilibriumDensity = totalDensity * d_directionWeights[d] * (1.0f + 3.0f * dotProduct + 4.5f * dotProduct * dotProduct - 1.5f * uSquared);
        microDensity[d * cellCount + n] = (1.0f - 1.0f / d_tau) * microDensity[d * cellCount + n] + (1.0f / d_tau) * equilibriumDensity;
    }
}

__global__ void streaming(float* microDensity, float* microDensity2, int* barrier, int xSize, int ySize, int cellCount) {
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

        if (prevX >= 0 && prevX < xSize && prevY >= 0 && prevY < ySize && !barrier[prevY * xSize + prevX])
            microDensity2[d * cellCount + n] = microDensity[d * cellCount + prevY * xSize + prevX];
        else
            microDensity2[d * cellCount + n] = microDensity[d_oppositeDirection[d] * cellCount + n];
    }
}

__global__ void advectSmoke(float* smoke, float* smoke2, float* xVelocity, float* yVelocity, int* barrier, int xSize, int ySize, int cellCount, float advScale) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= cellCount) return;

    if (barrier[n]) {
        smoke2[n] = smoke[n];
        return;
    }

    int x = n % xSize;
    int y = n / xSize;

    float vX = xVelocity[n];
    float vY = yVelocity[n];

    float prevX = x - vX * advScale;
    float prevY = y - vY * advScale;

    prevX = fmaxf(0.0f, fminf(prevX, (float)(xSize - 1)));
    prevY = fmaxf(0.0f, fminf(prevY, (float)(ySize - 1)));

    int x0 = (int)prevX;
    int y0 = (int)prevY;
    int x1 = min(x0 + 1, xSize - 1);
    int y1 = min(y0 + 1, ySize - 1);

    float sx = prevX - x0;
    float sy = prevY - y0;

    float s00 = smoke[y0 * xSize + x0];
    float s10 = smoke[y0 * xSize + x1];
    float s01 = smoke[y1 * xSize + x0];
    float s11 = smoke[y1 * xSize + x1];

    float s0 = s00 * (1 - sx) + s10 * sx;
    float s1 = s01 * (1 - sx) + s11 * sx;

    smoke2[n] = s0 * (1 - sy) + s1 * sy;
}

void injectSmoke() {
    /*for (int y = ySize / 3; y < 2 * ySize / 3; y++) {
        int n = y * xSize + 1;
        smoke[n] = 1.0f;
    }*/

    for (int y = 10; y < ySize; y += 25) {
        int n = y * xSize + 1;
        smoke[n] = 1.0f;
        smoke[n + xSize] = 1.0f;
        smoke[n - xSize] = 1.0f;
    }
}

void applyInlet() {
    for (int y = 1; y < ySize - 1; y++) {
        int n = y * xSize + 0;

        float f0 = microDensity[0 * cellCount + n];
        float f1 = microDensity[1 * cellCount + n];
        float f2 = microDensity[2 * cellCount + n];
        float f4 = microDensity[4 * cellCount + n];
        float f6 = microDensity[6 * cellCount + n];
        float f7 = microDensity[7 * cellCount + n];

        float u = inletVelocity;
        float rho = (f0 + f1 + f2 + 2.0f * (f4 + f6 + f7)) / (1.0f - u);

        microDensity[3 * cellCount + n] = f4 + (2.0f / 3.0f) * rho * u;
        microDensity[5 * cellCount + n] = f7 + 0.5f * (f2 - f1) + (1.0f / 6.0f) * rho * u;
        microDensity[8 * cellCount + n] = f6 + 0.5f * (f1 - f2) + (1.0f / 6.0f) * rho * u;
    }
}

void applyOutlet() {
    int xOut = xSize - 1;
    int xSrc = xSize - 2;

    for (int y = 1; y < ySize - 1; y++) {
        int nOut = y * xSize + xOut;
        int nSrc = y * xSize + xSrc;

        for (int d = 0; d < 9; d++) {
            microDensity[d * cellCount + nOut] =
                microDensity[d * cellCount + nSrc];
        }

        xVelocity[nOut] = xVelocity[nSrc];
        yVelocity[nOut] = yVelocity[nSrc];
        macroDensity[nOut] = macroDensity[nSrc];
    }
}

void applyTopBottom() {
    for (int x = 0; x < xSize; x++) {
        int top = x;
        int below = xSize + x;

        int bottom = (ySize - 1) * xSize + x;
        int above = (ySize - 2) * xSize + x;

        for (int d = 0; d < 9; d++) {
            microDensity[d * cellCount + top] = microDensity[d * cellCount + below];
            microDensity[d * cellCount + bottom] = microDensity[d * cellCount + above];
        }
    }
}

void drawStreamlines(sf::RenderWindow& window) {
    sf::VertexArray lines(sf::PrimitiveType::Lines);
    const float scale = 10.0f;

    for (int y = 0; y < ySize; y += 10) {
        for (int x = 0; x < xSize; x += 10) {
            int n = y * xSize + x;
            if (barrier[n]) continue;

            float vX = xVelocity[n];
            float vY = yVelocity[n];

            float length = std::sqrt(vX * vX + vY * vY);

            sf::Vector2f dir(0.f, 0.f);
            if (length != 0.0f) {
                dir.x = vX / length;
                dir.y = vY / length;
            }

            sf::Vertex start;
            start.position = sf::Vector2f(x * renderScale + 0.5f, y * renderScale + 0.5f);
            start.color = sf::Color::Black;

            sf::Vertex end;
            end.position = start.position + dir * scale;
            end.color = sf::Color::Black;

            lines.append(start);
            lines.append(end);
        }
    }

    window.draw(lines);
}

int main() {
    initialize();

    sf::RenderWindow window(sf::VideoMode({(unsigned)(xSize * renderScale), (unsigned)(ySize * renderScale)}), "LBM");

    sf::Image image(sf::Vector2u(xSize, ySize), sf::Color::Black);
    sf::Texture texture(sf::Vector2u(xSize, ySize));
    sf::Sprite  sprite(texture);
    sprite.setScale(sf::Vector2f(renderScale, renderScale));

    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();

            if (event->is<sf::Event::KeyPressed>()) {
                if (event->getIf<sf::Event::KeyPressed>()->code == sf::Keyboard::Key::Space) {
                    showSmoke = !showSmoke;
                }
            }
        }

        for (int step = 0; step < stepsPerFrame; step++) {
            collision<<<blocks, threads>>>(microDensity, macroDensity, xVelocity, yVelocity, barrier, cellCount);
            streaming<<<blocks, threads>>>(microDensity, microDensity2, barrier, xSize, ySize, cellCount);
            advectSmoke<<<blocks, threads>>>(smoke, smoke2, xVelocity, yVelocity, barrier, xSize, ySize, cellCount, smokeAdvectionScale);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
            cudaDeviceSynchronize();
            std::swap(microDensity, microDensity2);
            std::swap(smoke, smoke2);
            applyInlet();
            applyOutlet();
            applyTopBottom();
            injectSmoke();
        }

        for (int y = 0; y < ySize; y++) {
            for (int x = 0; x < xSize; x++) {
                int n = y * xSize + x;

                if (barrier[n]) {
                    image.setPixel({(unsigned)x, (unsigned)y}, sf::Color::Red);
                    continue;
                }

                if (showSmoke) {
                    float s = smoke[n];
                    s = std::clamp(s, 0.0f, 1.0f);
                    uint8_t c = (uint8_t)(s * 255);
                    image.setPixel({(unsigned)x, (unsigned)y}, sf::Color(c, c, c));
                } else {
                    image.setPixel({(unsigned)x, (unsigned)y}, velocityToColor(x, y));
                }
            }
        }

        texture.update(image);
        window.clear();
        window.draw(sprite);
        if (!showSmoke){
            drawStreamlines(window);
        }
        window.display();
    }
}