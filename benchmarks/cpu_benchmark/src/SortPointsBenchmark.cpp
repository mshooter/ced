#include "benchmark/benchmark.h"

#include "SortPoints.hpp"
#include "Point.hpp"

static void BM_partition(benchmark::State& state)
{
    using namespace ced::cpu; 
    Point A = {0,0};
    Point B = {1,1};
    Point C = {3,0};
    Point D = {2,0};
    std::vector<Point> pts = {A, C, D , B};
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(partition<Point>(pts, 0, pts.size()-1));
    }
}
BENCHMARK(BM_partition);
//  -------------------------------------------------------------------------
static void BM_quickSort(benchmark::State& state)
{
    using namespace ced::cpu; 
    Point A = {0,0};
    Point B = {1,1};
    Point C = {3,0};
    Point D = {2,0};
    std::vector<Point> pts = {A, C, D , B};
    for(auto _ : state)
    {
        quickSort<Point>(pts, 0, pts.size()-1);
    }
}
BENCHMARK(BM_quickSort);
//  -------------------------------------------------------------------------
static void BM_partitionDist(benchmark::State& state)
{
    using namespace ced::cpu; 
    Point A = {0,0};
    Point B = {1,1};
    Point C = {2,0};
    Point D = {3,0};
    Point cc = {1,0};
    std::vector<Point> pts = {A, C, D , B};
    std::vector<unsigned int> ids = {0, 1, 2, 3};
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(partitionDist<float>(ids, pts, cc, 0, pts.size()-1));
    }    
}
BENCHMARK(BM_partitionDist);
//  -------------------------------------------------------------------------
static void BM_quickSortDist(benchmark::State& state)
{
    using namespace ced::cpu; 
    Point A = {0,0};
    Point B = {1,1};
    Point C = {2,0};
    Point D = {3,0};
    Point cc = {1,0};
    std::vector<Point> pts = {A, C, D , B};
    std::vector<unsigned int> ids = {0, 1, 2, 3};
    for(auto _ : state)
    {
        quickSortDist<float>(ids, pts, cc, 0, pts.size()-1);
    }    
}
BENCHMARK(BM_quickSortDist);

