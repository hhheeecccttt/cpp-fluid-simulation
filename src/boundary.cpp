#include "boundary.h"
#include "config.h"
#include "globals.h"

void applyInlet() {
    for (int y = 1; y < ySize - 1; y++) {
        int n  = y * xSize;
        float u = inletVelocity;

        float f0 = microDensity[0 * cellCount + n];
        float f1 = microDensity[1 * cellCount + n];
        float f2 = microDensity[2 * cellCount + n];
        float f4 = microDensity[4 * cellCount + n];
        float f6 = microDensity[6 * cellCount + n];
        float f7 = microDensity[7 * cellCount + n];

        float rho = (f0 + f1 + f2 + 2.0f * (f4 + f6 + f7)) / (1.0f - u);

        microDensity[3 * cellCount + n] = f4 + (2.0f / 3.0f) * rho * u;
        microDensity[5 * cellCount + n] = f7 + 0.5f * (f2 - f1) + (1.0f / 6.0f) * rho * u;
        microDensity[8 * cellCount + n] = f6 + 0.5f * (f1 - f2) + (1.0f / 6.0f) * rho * u;
    }
}

void applyOutlet() {
    for (int y = 1; y < ySize - 1; y++) {
        int nOut = y * xSize + (xSize - 1);
        int nSrc = y * xSize + (xSize - 2);

        for (int d = 0; d < 9; d++)
            microDensity[d * cellCount + nOut] = microDensity[d * cellCount + nSrc];

        xVelocity[nOut]  = xVelocity[nSrc];
        yVelocity[nOut]  = yVelocity[nSrc];
        macroDensity[nOut] = macroDensity[nSrc];
    }
}

void applyTopBottom() {
    for (int x = 0; x < xSize; x++) {
        int top    = x;
        int below  = xSize + x;
        int bottom = (ySize - 1) * xSize + x;
        int above  = (ySize - 2) * xSize + x;

        for (int d = 0; d < 9; d++) {
            microDensity[d * cellCount + top]    = microDensity[d * cellCount + below];
            microDensity[d * cellCount + bottom] = microDensity[d * cellCount + above];
        }
    }
}

void injectSmoke() {
    for (int y = 10; y < ySize; y += 25) {
        int n = y * xSize + 1;
        smoke[n]          = 1.0f;
        smoke[n + xSize]  = 1.0f;
        smoke[n - xSize]  = 1.0f;
    }
}
