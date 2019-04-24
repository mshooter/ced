#ifndef GRADIENTS_CUH_INCLUDED
#define GRADIENTS_CUH_INCLUDED

#include <vector>

namespace ced
{
    namespace gpu
    {
        void calculateGradients(int& _height, 
                                int& _width,
                                std::vector<float>& _pixelData,
                                std::vector<float>& _orientations);
        inline int iDivUp(const unsigned int &a, const unsigned int &b){return (a%b != 0 ? (a/b+1) : (a/b));}
    }
}


#endif // GRADIENTS_CUH_INCLUDED

