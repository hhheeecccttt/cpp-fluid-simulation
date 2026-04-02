#include "rendering.h"
#include "config.h"
#include "globals.h"

#include <cmath>
#include <algorithm>

sf::Color velocityToColor(int x, int y) {
    float vX = xVelocity[y * xSize + x];
    float vY = yVelocity[y * xSize + x];
    float speed = std::sqrt(vX * vX + vY * vY);

    float t = std::clamp((speed - inletVelocity) / velocityRange, -1.0f, 1.0f);

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

void drawStreamlines(sf::RenderWindow& window) {
    sf::VertexArray lines(sf::PrimitiveType::Lines);
    const float arrowLength = 10.0f;

    for (int y = 0; y < ySize; y += 10) {
        for (int x = 0; x < xSize; x += 10) {
            int n = y * xSize + x;
            if (barrier[n]) continue;

            float vX = xVelocity[n];
            float vY = yVelocity[n];
            float len = std::sqrt(vX * vX + vY * vY);

            sf::Vector2f origin((float)(x * renderScale) + 0.5f,
                                (float)(y * renderScale) + 0.5f);

            sf::Vector2f dir(0.f, 0.f);
            if (len > 0.0f) {
                dir.x = (vX / len) * arrowLength;
                dir.y = (vY / len) * arrowLength;
            }

            sf::Vertex start, end;
            start.position = origin;
            start.color    = sf::Color::Black;
            end.position   = origin + dir;
            end.color      = sf::Color::Black;

            lines.append(start);
            lines.append(end);
        }
    }

    window.draw(lines);
}
