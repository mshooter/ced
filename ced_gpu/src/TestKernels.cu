#include "TestKernels.cuh"
#include <cstdio>

namespace ced
{
    namespace gpu
    {
        __global__ void helloWorld()
        {
            printf("Hello World from Gpu");
        }
    }
}
