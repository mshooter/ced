#include "GaussianFilter.hpp"
#include <cmath>
namespace ced
{
    namespace cpu
    {
        std::vector<float> gaussianFilter(int _dimension, float _sigma)
        {
            // declare variables
            std::vector<float> filter(_dimension * _dimension);
            const int middle = _dimension / 2; 
            const float sigma2 = _sigma * _sigma; 
            float sum = 0.0f; 
            // lamda function 
            auto g = [&](auto x, auto y)
            {
                return std::exp(-((x*x + y*y)/(2.0f*sigma2))) / (2.0f * std::acos(-1.0f) *sigma2);
            };
            // create filter 
            for(int id = 0; id < _dimension * _dimension; ++id)
            {
                filter[id] = g((id%_dimension)-middle, (id/_dimension)-middle);
                sum += filter[id];
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
