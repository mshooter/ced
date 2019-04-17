#include "benchmark/benchmark.h"

#include "GenerateRandomPoints.hpp"

static void BM_generateRandomPoints(benchmark::State& state)
{
    using namespace ced::cpu;
    for(auto _ : state)
    {
        std::vector<Point> coordList;
        std::vector<Point> whites;
        unsigned int amountOfPoints = 100;
        unsigned int width = 100; 
        unsigned int height = 100; 
        generateRandomPoints(coordList, whites, amountOfPoints, width, height);
    }
}
BENCHMARK(BM_generateRandomPoints);
