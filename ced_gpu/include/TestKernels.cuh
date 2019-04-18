#ifndef TESTKERNELS_H_INCLUDED 
#define TESTKERNELS_H_INCLUDED

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace ced
{
    namespace gpu
    {
        __global__ void helloWorld();
    }
}

#endif //TESTKERNELS_H_INCLUDED
  
