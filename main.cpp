#include <SFML/Graphics.hpp>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <math.h>

const int scale = 5;
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

std::vector<float> microDensity(9 * cellCount);
std::vector<float> microDensity2(9 * cellCount);

std::vector<float> smoke(cellCount, 0.0f);
std::vector<float> smoke2(cellCount, 0.0f);

std::vector<int> inletMask(cellCount, 0);

std::vector<float> macroDensity(cellCount);
std::vector<int> barrier(cellCount, 0);
std::vector<float> xVelocity(cellCount);
std::vector<float> yVelocity(cellCount);

int directionXVector[] = {0, 0, 0, 1, -1, 1, -1, -1, 1};
int directionYVector[] = {0, 1, -1, 0, 0, 1, -1, 1, -1};
float directionWeights[] = {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36};
int oppositeDirection[] = {0, 2, 1, 4, 3, 7, 5, 6, 8};

void parseBarrierFile(const std::string& filename, std::vector<int>& barrier) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int x, y;
        if (!(iss >> x >> y)) {
            continue;
        }

        barrier[y * xSize + x] = 1;
    }

    infile.close();
}

void initialize() {
    /*for (int n = 0; n < xSize; n++) {
        barrier[n] = 1;
        barrier[(ySize - 1) * xSize + n] = 1;
    }*/

    parseBarrierFile("outputs/airfoil_coordinates.txt", barrier);

    /*for (int n = 40; n < 60; n++) {
        barrier[n * xSize + 50] = 1;
    }*/

    float uSquared = inletVelocity * inletVelocity;
    for (int n = 0; n < cellCount; n++) {
        if (barrier[n] == 1) {
            continue;
        }

        xVelocity[n] = inletVelocity;
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

void collision() {
    //#pragma omp parallel for
    for (int n = 0; n < cellCount; n++) {
        if (barrier[n] == 1) {
            continue;
        }

        float totalDensity = 0;
        float vX = 0;
        float vY = 0;

        for (int d = 0; d < 9; d++) {
            totalDensity += microDensity[d * cellCount + n];
        }

        for (int d = 0; d < 9; d++) {
            vX += directionXVector[d] * microDensity[d * cellCount + n];
            vY += directionYVector[d] * microDensity[d * cellCount + n];
        }

        vX /= totalDensity;
        vY /= totalDensity;

        xVelocity[n] = vX;
        yVelocity[n] = vY;
        macroDensity[n] = totalDensity;

        float uSquared = vX * vX + vY * vY;
        for (int d = 0; d < 9; d++) {
            float dotProduct = directionXVector[d] * vX + directionYVector[d] * vY;
            float equilibriumDensity = totalDensity * directionWeights[d] * (1 + 3 * dotProduct + 4.5 * dotProduct * dotProduct - 1.5 * uSquared);
            microDensity[d * cellCount + n] = (1.0 - 1.0 / tau) * microDensity[d * cellCount + n] + (1.0 / tau) * equilibriumDensity;
        }
    }
}

void streaming() {
    //#pragma omp parallel for collapse(2)
    for (int x = 0; x < xSize; x++) {
        for (int y = 0; y < ySize; y++) {
            int n = y * xSize + x;

            for (int d = 0; d < 9; d++) {
                int prevX = x - directionXVector[d];
                int prevY = y - directionYVector[d];

                if (prevX >= 0 && prevX < xSize && prevY >= 0 && prevY < ySize && !barrier[prevY * xSize + prevX]) {
                    microDensity2[d * cellCount + n] = microDensity[d * cellCount + prevY * xSize + prevX];
                } else {
                    microDensity2[d * cellCount + n] = microDensity[oppositeDirection[d] * cellCount + n];
                }
            }
        }
    }

    std::swap(microDensity, microDensity2);
}

void advectSmoke() {
    for (int x = 0; x < xSize; x++) {
        for (int y = 0; y < ySize; y++) {
            int n = y * xSize + x;

            if (barrier[n]) {
                smoke2[n] = smoke[n];
                continue;
            }

            float vX = xVelocity[y * xSize + x];
            float vY = yVelocity[y * xSize + x];

            float prevX = x - vX * smokeAdvectionScale;
            float prevY = y - vY * smokeAdvectionScale;

            prevX = std::max(0.0f, std::min(prevX, (float)(xSize - 1)));
            prevY = std::max(0.0f, std::min(prevY, (float)(ySize - 1)));

            int x0 = (int)prevX;
            int y0 = (int)prevY;
            int x1 = std::min(x0 + 1, xSize - 1);
            int y1 = std::min(y0 + 1, ySize - 1);

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
    }
    std::swap(smoke, smoke2);
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
            collision();
            streaming();
            applyInlet();
            applyOutlet();
            applyTopBottom();
            injectSmoke();
            advectSmoke();
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