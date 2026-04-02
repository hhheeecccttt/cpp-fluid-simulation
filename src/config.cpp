#include "config.h"

int directionXVector[] = { 0,  0,  0,  1, -1,  1, -1, -1,  1 };
int directionYVector[] = { 0,  1, -1,  0,  0,  1, -1,  1, -1 };
float directionWeights[] = {4.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/9, 1.0f/36, 1.0f/36, 1.0f/36, 1.0f/36};
int oppositeDirection[] = { 0, 2, 1, 4, 3, 7, 5, 6, 8 };

//https://pmc.ncbi.nlm.nih.gov/articles/PMC6266048/
//https://physics.weber.edu/schroeder/javacourse/LatticeBoltzmann.pdf
//https://personal.ems.psu.edu/~fkd/courses/EGEE520/2017Deliverables/LBM_2017.pdf

float transformationMatrix[9][9] = {
    { 1,   1,  1,  1,  1,  1,  1,  1,  1 },
    {-4,  -1, -1, -1, -1,  2,  2,  2,  2 },
    { 4,  -2, -2, -2, -2,  1,  1,  1,  1 },
    { 0,   0,  0,  1, -1,  1, -1, -1,  1 },
    { 0,   0,  0, -2,  2,  1, -1, -1,  1 },
    { 0,   1, -1,  0,  0,  1, -1,  1, -1 },
    { 0,  -2,  2,  0,  0,  1, -1,  1, -1 },
    { 0,  -1, -1,  1,  1,  0,  0,  0,  0 },
    { 0,   0,  0,  0,  0,  1,  1, -1, -1 }
};

float transformationMatrixInverse[9][9] = {
    { 1.0f/9,  -1.0f/9,   1.0f/9,  0,        0,        0,        0,        0,       0      },
    { 1.0f/9,  -1.0f/36, -1.0f/18, 0,        0,        1.0f/6,  -1.0f/6,  -1.0f/4,  0      },
    { 1.0f/9,  -1.0f/36, -1.0f/18, 0,        0,       -1.0f/6,   1.0f/6,  -1.0f/4,  0      },
    { 1.0f/9,  -1.0f/36, -1.0f/18, 1.0f/6,  -1.0f/6,  0,        0,        1.0f/4,   0      },
    { 1.0f/9,  -1.0f/36, -1.0f/18,-1.0f/6,   1.0f/6,  0,        0,        1.0f/4,   0      },
    { 1.0f/9,   1.0f/18,  1.0f/36, 1.0f/6,   1.0f/12, 1.0f/6,   1.0f/12,  0,        1.0f/4 },
    { 1.0f/9,   1.0f/18,  1.0f/36,-1.0f/6,  -1.0f/12,-1.0f/6,  -1.0f/12,  0,        1.0f/4 },
    { 1.0f/9,   1.0f/18,  1.0f/36,-1.0f/6,  -1.0f/12, 1.0f/6,   1.0f/12,  0,       -1.0f/4 },
    { 1.0f/9,   1.0f/18,  1.0f/36, 1.0f/6,   1.0f/12,-1.0f/6,  -1.0f/12,  0,       -1.0f/4 }
};

float relaxationRates[9] = {
    0.0f,          // Density
    1.6f,          // Energy mode
    1.6f,          // Higher-order energy
    0.0f,          // X momentum
    1.2f,          // X energy flux
    0.0f,          // Y momentum 
    1.2f,          // Y energy flux
    1.0f / tau,    // Stress (xx-yy)
    1.0f / tau     // Stress (xy)
};
