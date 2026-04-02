#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstdint>

#include "config.h"
#include "globals.h"
#include "simulation.h"
#include "boundary.h"
#include "rendering.h"

static void checkCuda() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
}

static void stepSimulation() {
    launchCollisionMRT();
    launchStreaming();
    launchAdvectSmoke();
    checkCuda();
    cudaDeviceSynchronize();

    std::swap(microDensity, microDensity2);
    std::swap(smoke, smoke2);

    applyInlet();
    applyOutlet();
    applyTopBottom();
    injectSmoke();
}

static void updateImage(sf::Image& image) {
    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) {
            int n = y * xSize + x;

            if (barrier[n]) {
                image.setPixel({(unsigned)x, (unsigned)y}, sf::Color::Red);
                continue;
            }

            sf::Color pixel;
            if (showSmoke) {
                float s = std::clamp(smoke[n], 0.0f, 1.0f);
                uint8_t c = static_cast<uint8_t>(s * 255);
                pixel = sf::Color(c, c, c);
            } else {
                pixel = velocityToColor(x, y);
            }
            image.setPixel({(unsigned)x, (unsigned)y}, pixel);
        }
    }
}

int main() {
    initialize();

    sf::RenderWindow window(sf::VideoMode({(unsigned)(xSize * renderScale),(unsigned)(ySize * renderScale)}),"LBM Simulation");

    sf::Image image(sf::Vector2u(xSize, ySize), sf::Color::Black);
    sf::Texture texture(sf::Vector2u(xSize, ySize));
    sf::Sprite sprite(texture);
    sprite.setScale(sf::Vector2f((float)renderScale, (float)renderScale));

    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();

            if (const auto* key = event->getIf<sf::Event::KeyPressed>())
                if (key->code == sf::Keyboard::Key::Space)
                    showSmoke = !showSmoke;
        }

        for (int step = 0; step < stepsPerFrame; step++)
            stepSimulation();

        updateImage(image);
        texture.update(image);

        window.clear();
        window.draw(sprite);
        if (!showSmoke)
            drawStreamlines(window);
        window.display();
    }

    return 0;
}
