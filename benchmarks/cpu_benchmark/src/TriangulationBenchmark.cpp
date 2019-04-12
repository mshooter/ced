#include "benchmark/benchmark.h"

#include "Triangulation.hpp"

static void BM_calculateCentroidCenter(benchmark::State& state)
{
    using namespace ced::cpu;
    Point A = {0,0};
    Point B = {1,1};
    Point C = {3,0};
    Point D = {2,0};
    std::vector<Point> pts = {A, B, C, D};
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(calculateCentroidCenter(pts));
    } 
}
BENCHMARK(BM_calculateCentroidCenter);
//  ---------------------------------------------------------------------
static void BM_createFirstTri(benchmark::State& state)
{
    using namespace ced::cpu;
    Point p1 = {0,0};
    Point p2 = {1,1};
    Point p3 = {2,0};
    
    std::vector<Point> _points = {p1, p2, p3};
    Point sc = calculateCentroidCenter(_points);
    unsigned int i0 = INVALID_IDX; 
    unsigned int i1 = INVALID_IDX; 
    unsigned int i2 = INVALID_IDX; 
    for(auto _ : state)
    {
        createFirstTri(_points, i0, i1, i2, sc);
    } 
}
BENCHMARK(BM_createFirstTri);
//  ---------------------------------------------------------------------
static void BM_hashkey(benchmark::State& state)
{
    using namespace ced::cpu;
    Point p1 = {0,0};
    Point p2 = {1,1};
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(hash_key(p1, p2, 3));
    }
}
BENCHMARK(BM_hashkey);
//  ---------------------------------------------------------------------
static void BM_link(benchmark::State& state)
{
    using namespace ced::cpu;
    std::vector<unsigned int> halfedges;
    halfedges.reserve(3);
    for(auto _ : state)
    {
        link(0, INVALID_IDX, halfedges);
        link(1, INVALID_IDX, halfedges);
        link(2, INVALID_IDX, halfedges);
    }
}
BENCHMARK(BM_link);
//  ---------------------------------------------------------------------
static void BM_addTriangle(benchmark::State& state)
{
    using namespace ced::cpu;
    std::vector<unsigned int> halfedges;
    std::vector<unsigned int> triangles; 
    halfedges.reserve(3);
    triangles.reserve(3);
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(add_triangle(0,1,2,INVALID_IDX, INVALID_IDX, INVALID_IDX,triangles,halfedges));
    }
}
BENCHMARK(BM_addTriangle);
//  ---------------------------------------------------------------------
static void BM_triangulation(benchmark::State& state)
{
    using namespace ced::cpu;

    for(auto _ : state)
    {
        Point p1 = {0,0};
        Point p2 = {1,1};
        Point p3 = {2,0};
        Point p4 = {6,2};
    
        std::vector<Point> verts = {p1, p2, p3, p4};
        std::vector<unsigned int> triangles;
        triangulate(verts, triangles);
    }
}
BENCHMARK(BM_triangulation);
//  ---------------------------------------------------------------------
