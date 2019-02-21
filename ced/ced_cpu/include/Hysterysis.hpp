// Hysterysis.hpp 
#ifndef HYSTERYSIS_H_INCLUDED
#define HYSTERYSIS_H_INCLUDED


#include <vector>

namespace ced 
{
    void hysterysis(std::vector<float>& _nonMax, int _height, int _width,  float _minValue = 0.0f, float _maxValue = 1.0f);
}
#endif //NONMAXIMUMSUPRESSION_H_INCLUDED


