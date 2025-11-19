#pragma once

#include <chrono>
#include <cstring>
#include <iostream>

struct Timer {
    std::chrono::steady_clock::time_point start;
    std::string name;

    double lap;

    Timer(std::string name) : name(name) {
        start = std::chrono::steady_clock::now();
    }

    void Lap() {
        auto now = std::chrono::steady_clock::now();
        lap = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count() / (double) 1e9;
        std::cout << "Clock " << name << ": " << std::fixed << lap << " seconds" << std::endl;
    }
};