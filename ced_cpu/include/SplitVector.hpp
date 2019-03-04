// @file: SplitVector.hpp
#ifndef SPLITVECTOR_H_INCLUDED
#define SPLITVECTOR_H_INCLUDED

#include <vector>
namespace ced
{
    namespace cpu
    {
        template <typename T>
        void splitVector(std::vector<std::vector<T>>& fVec, std::vector<T> _pts)
        {
            int sizeVec = _pts.size();
            int midIndex = sizeVec / 2;
            if(sizeVec == 2 || sizeVec == 3)
            {
                fVec.push_back(_pts);
            }
            else
            {
                typename std::vector<T>::const_iterator begin = _pts.begin();
                typename std::vector<T>::const_iterator middle = _pts.begin() + midIndex;
                typename std::vector<T>::const_iterator end = _pts.begin() + sizeVec;
                std::vector<T> nVecL(begin, middle);
                std::vector<T> nVecR(middle, end);
                splitVector(fVec, nVecL);
                splitVector(fVec, nVecR);
            }
        }
    }
}

#endif // SPLITVECTOR_H_INCLUDED
