#ifndef GAUSSIANFILTER_CUH_INCLUDED
#define GAUSSIANFILTER_CUH_INCLUDED

#include <vector>

namespace ced
{
    namespace gpu
    {
        //  ------------------------------------------------------------------------------
        //  @build: gaussian function
        //  ------------------------------------------------------------------------------
        struct g
        {
            const int d;
            const float s; 
            g(int _d, float _s) : d(_d), s(_s) {}  
            __host__ __device__ 
            float operator()(const int& id)
            {
                int middle = d / 2;
                int y = (id / d) - middle; 
                int x = (id % d) - middle; 
                return std::exp(-((x*x + y*y)/(2.0f*s))) / (2.0f * std::acos(-1.0f) *s);
            }
        };
        //  ------------------------------------------------------------------------------
        //  @build: create gaussian filter
        //  ------------------------------------------------------------------------------
        std::vector<float> gaussianFilter(int _dimension = 5, float _sigma = 1.4f); 
    }
}

#endif // GAUSSIANFILTER_CUH_INCLUDED
