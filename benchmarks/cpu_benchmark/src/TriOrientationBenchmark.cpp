#include "benchmark/benchmark.h"
#include "TriOrientation.hpp"

static void BM_TriOrientation(benchmark::State& state)
{
    using namespace ced::cpu;

    for(auto _ : state)
    {
        Point p = {0,0};
        Point p1 = {1,1};
        Point p2 = {2,0};
        benchmark::DoNotOptimize(isCCW<float>(p, p1, p2)); 
    }
}
BENCHMARK(BM_TriOrientation);
