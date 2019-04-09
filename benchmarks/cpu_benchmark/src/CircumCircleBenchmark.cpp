#include "benchmark/benchmark.h"

#include "CircumCircle.hpp"
#include "Point.hpp"

static void BM_isPointInCircle(benchmark::State& state)
{
    using namespace ced::cpu;
    Point A = {0,0};
    Point B = {1,1};
    Point C = {3,0};
    Point D = {2,0};
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(isPointInCircle<float>(A, B, C, D));
    }
}
BENCHMARK(BM_isPointInCircle);
//  ------------------------------------------------------------------
static void BM_circumRadius(benchmark::State& state)
{
    using namespace ced::cpu;
    Point A = {0,0};
    Point B = {1,1};
    Point C = {2,0};
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(circumRadius<float>(A, B, C));
    }
}
BENCHMARK(BM_circumRadius);
//  ------------------------------------------------------------------
static void BM_circumCenter(benchmark::State& state)
{
    using namespace ced::cpu;
    Point A = {0,0};
    Point B = {1,1};
    Point C = {2,0};
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(circumCenter(A, B, C));
    }
}
BENCHMARK(BM_circumCenter);

