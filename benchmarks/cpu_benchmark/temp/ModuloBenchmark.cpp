#include "benchmark/benchmark.h"
#include <functional>

//  ---------------------------------------------------------
static void BM_manualModulo(benchmark::State& state)
{
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(10%3);
    }
}
BENCHMARK(BM_manualModulo);
//  ---------------------------------------------------------
static void BM_fastModulo(benchmark::State& state)
{
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(std::modulus<int>()(10,3));
    }
}
BENCHMARK(BM_fastModulo);
