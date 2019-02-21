// Hysterysis.hpp 
#ifndef __HYSTERYSIS_H_INCLUDED__
#define __HYSTERYSIS_H_INCLUDED__


#include <vector>

namespace ced 
{
    void hysterysis(std::vector<float>& _nonMax, int _height, int _width,  float _minValue = 0.0f, float _maxValue = 1.0f);
}
#endif //__NONMAXIMUMSUPRESSION_H_INCLUDED__


