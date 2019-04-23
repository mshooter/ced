#include "Gradients.hpp"
#include <cmath>
#include <math.h>
#include <iostream>
namespace ced
{
    namespace cpu
    {
        void calculateGradients(int& _height, int& _width, std::vector<float>& _pixelData, std::vector<float>& _orientations)
        {
            // sobel edge detector filters 
            const std::vector<float> kernelX = {-1, 0, 1,
                                                -2, 0, 2,
                                                -1, 0, 1};
     
            const std::vector<float> kernelY = {-1, -2, -1,
                                                 0,  0,  0,
                                                 1,  2,  1};
            
            _height -= 2; 
            const int nwidth = _width - 2;
            _orientations.resize(_height * nwidth);
            for(int x = 0; x < _height* nwidth; ++x)
            {
                int i = x / nwidth;
                int j = x % nwidth;
                float Gx = 0.0f;
                float Gy = 0.0f;
                for(int y = 0; y < 9; ++y)
                {
                    int h = y / 3;
                    int w = y % 3;
                    int ibase = ((w+j)+(h+i)*_width) * 3;
                    int fbase = y;
                    Gx +=  (_pixelData[ibase+0] + _pixelData[ibase+1] + _pixelData[ibase+2]) * kernelX[fbase];
                    Gy +=  (_pixelData[ibase+0] + _pixelData[ibase+1] + _pixelData[ibase+2]) * kernelY[fbase];
                }
                // 3 is the channels
                int base = x * 3;
                float magnitude = std::abs(Gx) + std::abs(Gy); 
                //should be turned off
                _pixelData[base+0] = magnitude;
                _pixelData[base+1] = magnitude;
                _pixelData[base+2] = magnitude;
     
                float pi = 3.14f;
                float pi8 = pi/8.0f;
     
                // round theta
                float theta = std::atan(Gy/Gx);
                if(theta < 0)   theta = fmod((theta + 2 * pi), (2 * pi));
                if(theta <= pi8)    theta = 0.0f;
                else if(theta <= 3 * pi8)   theta = 45.0f;
                else if(theta <= 5 * pi8)   theta = 90.0f;
                else if(theta <= 7 * pi8)   theta = 135.0f;
                else if(theta <= 9 * pi8)   theta = 0.0f; 
                else if(theta <= 11 * pi8)  theta = 45.0f;
                else if(theta <= 13 * pi8)  theta = 90.0f;
                else if(theta <= 15 * pi8)  theta = 135.0f;
                else    theta = 0.0f;
                _orientations.push_back(theta);
     
            } 
            _pixelData.resize(nwidth * _height *3);
            _width = std::move(nwidth);
        }
    }
}
