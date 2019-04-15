#include "benchmark/benchmark.h"
#include "AssignTriToPix.hpp"
#include "Triangulation.hpp"

using namespace ced::cpu;
std::vector<Point> white_ptsCoord = {Point(0,0), Point (1,1), Point(2,0), Point(3,1)};

static void BM_assignTriToPix(benchmark::State& state)
{
    std::vector<unsigned int> triangles;
    triangulate(white_ptsCoord, triangles);
    std::vector<Point> ptIdx; 
    std::vector<unsigned int> triangle; 
    unsigned int height = 10;
    unsigned int width = 10;
    for(auto _ : state)
    {
        assignTriToPix(triangle, ptIdx, triangles, white_ptsCoord, height, width);
    }
}
BENCHMARK(BM_assignTriToPix);

static void BM_avgColour(benchmark::State& state)
{
    for(auto _ : state)
    {
    }
}
BENCHMARK(BM_avgColour);

static void BM_assignColToTri(benchmark::State& state)
{
    for(auto _ : state)
    {
    }
}
BENCHMARK(BM_assignColToTri);

static void BM_assignColToPix(benchmark::State& state)
{
    for(auto _ : state)
    {
    }
}
BENCHMARK(BM_assignColToPix);

