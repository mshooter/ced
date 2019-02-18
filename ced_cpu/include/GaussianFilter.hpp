// GaussianFilter.hpp 
#ifndef __GAUSSIANFILTER_H_INCLUDED__
#define __GAUSSIANFILTER_H_INCLUDED__

#include <vector>

namespace ced 
{
    //------------------------------------------------------------
    /// @build gaussianFilter 
    /// @param[_in] _dimension : dimension of the filter 
    /// @param[_in] _sigma : the amount of blur 
    /// @return : values of a gaussianblur filter
    //------------------------------------------------------------
    std::vector<float> gaussianFilter(int _dimension = 5, float _sigma = 1.4f);
}
#endif //__GAUSSIANFILTER_H_INCLUDED__


