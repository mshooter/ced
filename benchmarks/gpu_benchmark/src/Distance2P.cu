#include "benchmark/benchmark.h"
#include "Distance2P.cuh"

static void BM_Distance2P(benchmark::State& state)
{
    float x1 = 0.5f
    float y1 = 0.1f
    float x2 = 0.8f
    float y2 = 0.3f
    for(auto _ : state)
    {

        auto start = std::chrono::high_resolution_clock::now();
        benchmark::DoNotOptimize(ced::gpu::distance2P<float>(x1, y1, x2, y2));
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(BM_Distance2P)->Range(a, 1<<17)->UseManualTime();
