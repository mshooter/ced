#include "benchmark/benchmark.h"

#include "SortPoints.hpp"
#include "Point.hpp"

static void CPU_partition(benchmark::State& state)
{
    using namespace ced::cpu; 
    Point A = {0,0};
    Point B = {1,1};
    Point C = {3,0};
    Point D = {2,0};

    for(auto _ : state)
    {
        std::vector<Point> pts = {A, C, D , B};
        benchmark::DoNotOptimize(partition<Point>(pts, 0, pts.size()-1));
    }
}
BENCHMARK(CPU_partition);
//  -------------------------------------------------------------------------
static void CPU_quickSort(benchmark::State& state)
{
    using namespace ced::cpu; 
    Point A = {0,0};
    Point B = {1,1};
    Point C = {3,0};
    Point D = {2,0};

    for(auto _ : state)
    {
        std::vector<Point> pts = {A, C, D , B};
        quickSort<Point>(pts, 0, pts.size()-1);
    }
}
BENCHMARK(CPU_quickSort);
//  -------------------------------------------------------------------------
static void CPU_partitionDist(benchmark::State& state)
{
    using namespace ced::cpu; 
    Point A = {0,0};
    Point B = {1,1};
    Point C = {2,0};
    Point D = {3,0};
    Point cc = {1,0};

    for(auto _ : state)
    {
        std::vector<Point> pts = {A, C, D , B};
        std::vector<unsigned int> ids = {0, 1, 2, 3};
        benchmark::DoNotOptimize(partitionDist<float>(ids, pts, cc, 0, pts.size()-1));
    }    
}
BENCHMARK(CPU_partitionDist);
//  -------------------------------------------------------------------------
static void CPU_quickSortDist(benchmark::State& state)
{
    using namespace ced::cpu; 
    Point A = {0,0};
    Point B = {1,1};
    Point C = {2,0};
    Point D = {3,0};
    Point cc = {1,0};

    for(auto _ : state)
    {
        std::vector<Point> pts = {A, C, D , B};
        std::vector<unsigned int> ids = {0, 1, 2, 3};
        quickSortDist<float>(ids, pts, cc, 0, pts.size()-1);
    }    
}
BENCHMARK(CPU_quickSortDist);

