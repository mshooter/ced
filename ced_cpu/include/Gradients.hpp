// Gradients.hpp 
#ifndef __GADIENTS_H_INCLUDED__
#define __GADIENTS_H_INCLUDED__

#include <vector>

namespace ced 
{
    void calculateGradients(int& _height, int& _width, std::vector<float>& _pixelDatam, std::vector<float>& _orientations,
                            float _minValue = 0.0f, float _maxValue = 1.0f);
}
#endif //__GAUSSIANFILTER_H_INCLUDED__

