#pragma once
#include <string>

void initialize();
void parseBarrierFile(const std::string& filename, int* barrier);
void launchCollisionMRT();
void launchStreaming();
void launchAdvectSmoke();
