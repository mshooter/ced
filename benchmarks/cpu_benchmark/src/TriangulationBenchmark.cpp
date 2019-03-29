#include "benchmark/benchmark.h"

#include "Triangulation.hpp"

static void BM_addTriangle(benchmark::State& state)
{
    using namespace ced::cpu;
    std::vector<int> triangles;
    triangles.reserve(3);
    std::vector<int> halfedges;
    halfedges.reserve(3);
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(addTriangle<int>(1, 2, 0, -1, -1, -1, triangles, halfedges));
    }
}
BENCHMARK(BM_addTriangle);
