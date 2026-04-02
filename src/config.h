#pragma once

constexpr int scale = 5;
constexpr int xSize = 90 * scale;
constexpr int ySize = 60 * scale;
constexpr int cellCount = xSize * ySize;
constexpr int renderScale = 2;
constexpr int stepsPerFrame = 5;

constexpr float tau = 0.6f;
constexpr float inletVelocity = 0.1f;
constexpr float velocityRange = inletVelocity / 2.0f;
constexpr float smokeAdvectionScale = 10.0f;

constexpr int threads = 128;
constexpr int blocks  = (cellCount + threads - 1) / threads;

extern int directionXVector[9];
extern int directionYVector[9];
extern float directionWeights[9];
extern int oppositeDirection[9];

extern float transformationMatrix[9][9];
extern float transformationMatrixInverse[9][9];
extern float relaxationRates[9];
