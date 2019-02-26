// GaussianFilter.hpp 
#ifndef GAUSSIANFILTER_H_INCLUDED
#define GAUSSIANFILTER_H_INCLUDED

#include <vector>

namespace ced 
{
    namespace cpu
    {
        //------------------------------------------------------------
        /// @build gaussianFilter 
        /// @param[_in] _dimension : dimension of the filter 
        /// @param[_in] _sigma : the amount of blur 
        /// @return : values of a gaussianblur filter
        //------------------------------------------------------------
        std::vector<float> gaussianFilter(int _dimension = 5, float _sigma = 1.4f);
    }
}
#endif //GAUSSIANFILTER_H_INCLUDED


