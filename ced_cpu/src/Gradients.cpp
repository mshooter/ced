#include "Gradients.hpp"
#include <cmath>
#include <iostream>
namespace ced
{
    void calculateGradients(int& _height, int& _width, std::vector<float>& _pixelData, std::vector<float>& _orientations, float _minValue, float _maxValue)
    {
        // sobel edge detector filters 
        std::vector<float> kernelX = {-1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1};
        std::vector<float> kernelY = {-1, -2, -1,
                                     0,  0,  0,
                                    1, 2, 1};
        _height -= 2; 
        int nwidth = _width - 2;
        _orientations.resize(_height * nwidth);
        for(int i=0; i < _height; ++i)
        {
            for(int j=0; j < nwidth; ++j)
              {
                float Gx = 0.0f;
                float Gy = 0.0f;
                 for(int h=i; h < i + 3; ++h)
                 {
                     for(int w=j; w < j + 3; ++w)
                     {

                          int ibase = (w+h*_width) * 3;
                          int fbase = ((h-i) + (w-j) * 3);
                          Gx +=  (_pixelData[ibase+0] + _pixelData[ibase+1] + _pixelData[ibase+2]) * kernelX[fbase];
                          Gy +=  (_pixelData[ibase+0] + _pixelData[ibase+1] + _pixelData[ibase+2]) * kernelY[fbase];
                     }
                 }
                int base = (j+i*nwidth)* 3;
                float magnitude = std::abs(Gx) + std::abs(Gy); 
                float theta = std::atan(Gy/Gx) * 180/3.14f;
                //std::cout<<theta<<std::endl;
                _pixelData[base+0] = magnitude > _maxValue ? 1.0f : magnitude;
                _pixelData[base+1] = magnitude > _maxValue ? 1.0f : magnitude;
                _pixelData[base+2] = magnitude > _maxValue ? 1.0f : magnitude;

                _pixelData[base+0] = magnitude < _minValue ? 0.0f : magnitude;
                _pixelData[base+1] = magnitude < _minValue ? 0.0f : magnitude;
                _pixelData[base+2] = magnitude < _minValue ? 0.0f : magnitude;
                _orientations.push_back(theta);
              }
          }
        _width = std::move(nwidth);
    }
}
