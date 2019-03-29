#include "benchmark/benchmark.h"

#include "TriOrientation.hpp"
#include "Point.hpp"

static void BM_TriOrientation(benchmark::State& state)
{
    using namespace ced::cpu;
    Point p = {0,0};
    Point p1 = {1,1};
    Point p2 = {2,0};
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(isCCW<Point, float>(p, p1, p2)); 
    }
}
BENCHMARK(BM_TriOrientation);
