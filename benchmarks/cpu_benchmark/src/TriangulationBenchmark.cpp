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
//  ----------------------------------------------------------------------------------------------
static void BM_link(benchmark::State& state)
{
    using namespace ced::cpu;
    std::vector<int> halfedges;
    for(auto _ : state)
    {
        link(0,-1, halfedges);
    }
}
BENCHMARK(BM_link);
//  ----------------------------------------------------------------------------------------------
static void BM_pseudoAngle(benchmark::State& state)
{
    using namespace ced::cpu;
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(pseudo_angle<float, Point>(Point(1,3)));
    }
}
BENCHMARK(BM_pseudoAngle);
//  ----------------------------------------------------------------------------------------------
static void BM_hashkey(benchmark::State& state)
{
    using namespace ced::cpu;
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(hash_key(Point(1,1), Point(1,0), 2));
    }
}
BENCHMARK(BM_hashkey);
