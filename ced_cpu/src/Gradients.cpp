#include "Gradients.hpp"
#include <cmath>
#include <math.h>
namespace ced
{
    namespace cpu
    {
        void calculateGradients(int& _height,
                                int& _width,
                                std::vector<float>& _red, 
                                std::vector<float>& _green, 
                                std::vector<float>& _blue, 
                                std::vector<float>& _orientations)
        {
            // initialise
            _height -= 2; 
            const int nwidth = _width - 2;
            _orientations.resize(_height * nwidth);
            
            // sobel edge detector filters 
            const std::vector<float> kernelX = {-1, 0, 1,
                                                -2, 0, 2,
                                                -1, 0, 1};
     
            const std::vector<float> kernelY = {-1, -2, -1,
                                                 0,  0,  0,
                                                 1,  2,  1};
            
            // start manipulating the data
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
                    int ibase = ((w+j)+(h+i)*_width);
                    int fbase = y;
                    Gx +=  ((_red[ibase] + _green[ibase] + _blue[ibase])/3.0f) * kernelX[fbase];
                    Gy +=  ((_red[ibase] + _green[ibase] + _blue[ibase])/3.0f) * kernelY[fbase];
                }
                // 3 is the channels
                int base = x;
                float magnitude = std::abs(Gx) + std::abs(Gy); 
                //should be turned off because you not showing the gradients
                _red[base] = magnitude;
                _green[base] = magnitude;
                _blue[base] = magnitude;
     
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
            _red.resize(nwidth * _height);
            _green.resize(nwidth * _height);
            _blue.resize(nwidth * _height);
            _width = std::move(nwidth);
        }
    }
}
