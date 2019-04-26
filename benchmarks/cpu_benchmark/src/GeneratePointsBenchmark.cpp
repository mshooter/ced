#include "benchmark/benchmark.h"

#include "GenerateRandomPoints.hpp"

static void BM_generateRandomPoints(benchmark::State& state)
{
    using namespace ced::cpu;
    for(auto _ : state)
    {
        std::vector<Point> coordList;
        unsigned int amountOfPoints = 100;
        unsigned int width = 100; 
        unsigned int height = 100; 
        generateRandomPoints(coordList, amountOfPoints, width, height);
    }
}
BENCHMARK(BM_generateRandomPoints);
