#ifndef GAUSSIANFILTER_CUH_INCLUDED
#define GAUSSIANFILTER_CUH_INCLUDED

#include <vector>

namespace ced
{
    namespace gpu
    {
        //------------------------------------------------------------
        /// @build gaussianFilter 
        /// @param[_in] _dimension : dimension of the filter 
        /// @param[_in] _sigma : the amount of blur 
        /// @return : [vector] values of a gaussianblur filter
        //------------------------------------------------------------
        __host__ std::vector<float> gaussianFilter(int _dimension = 5, float _sigma = 1.4f);
        
    }
}

#endif // GAUSSIANFILTER_CUH_INCLUDED
