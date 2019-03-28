#include "benchmark/benchmark.h"
#include <limits>
#include <cmath>

static void BM_CeilStandard(benchmark::State& state)
{
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(std::ceil(std::sqrt(2)));
    }
}
BENCHMARK(BM_CeilStandard);
//  ----------------------------------------------------------------------------------------------------
static void BM_CeilShift(benchmark::State& state)
{
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(std::numeric_limits<unsigned int>::max() - static_cast<unsigned int>(std::numeric_limits<unsigned int>::max() - std::sqrt(2)));
    }
}
BENCHMARK(BM_CeilShift);
//  ----------------------------------------------------------------------------------------------------
static void BM_CeilCompare(benchmark::State& state)
{
    for(auto _ : state)
    {
        unsigned int hashSize = static_cast<unsigned int>(std::sqrt(2)); 
        if(hashSize < std::sqrt(2)) benchmark::DoNotOptimize(++hashSize);
    }
}
BENCHMARK(BM_CeilCompare);
