#include "Gradients.hpp"

namespace ced
{
    void calculateGradients(int& _height, int& _width, std::vector<float> _pixelData, std::vector<float>& _mag, std::vector<float>& _orientation)
    {
        // sobel edge detector filters 
        std::vector<float> kernelX = {-1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1};
        std::vector<float> kernelY = {-1, -2, -1,
                                     0,  0,  0,
                                    -1, -2, -1};
        _height -= 2; 
        _width -= 2;
        // iterate through image 
        // iterate through kernels
        // calculate magnitude 
        // calculate orientation
    }
}
