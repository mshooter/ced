#include "GaussianFilter.hpp"
#include <cmath>

namespace ced
{
    namespace cpu
    {
        std::vector<float> gaussianFilter(int _dimension, float _sigma)
        {
            std::vector<float> filter(_dimension * _dimension);
            const int middle = _dimension / 2; 
            float sum = 0.0f; 
            const float sigma2 = _sigma * _sigma; 
            auto g = [&](auto x, auto y)
            {
                return std::exp(-((x*x + y*y)/(2.0f*sigma2))) / (2.0f * std::acos(-1) *sigma2);
            };
            // create filter 
            for(int y=0; y < _dimension; ++y)
            {
                for(int x=0; x < _dimension; ++x)
                {
                    filter[x+y*_dimension] = g(x-middle, y-middle);
                    sum += filter[x+y*5];
                }
            }
        
            // normalize 
            for(auto& value : filter)
            {
                value /= sum;
            }
            
            return filter;
        }
    }
}
