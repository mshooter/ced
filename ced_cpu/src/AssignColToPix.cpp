#include "AssignColToPix.hpp"
#include "AvgColour.hpp"
#include "AssignColToTri.hpp"

namespace ced
{
    namespace cpu
    {
        void assignColToPix(
                std::vector<float>& imgData,
                const unsigned int& amountOfTriangles, 
                const std::vector<unsigned int> triangleIDs, 
                const std::vector<Point> triPixelIdx,
                const unsigned int& height,
                const unsigned int& width)
        {
            int currtt = 0;
            int oldtt = 0;
            for(unsigned tt = 0 ; tt < amountOfTriangles; ++tt)
            {
                for(auto t : triangleIDs)
                {
                    if(t == tt)
                    {
                        currtt++;
                    }
                }
                std::vector<Point> subPixidx(triPixelIdx.begin() + oldtt, triPixelIdx.begin() + currtt);
                float r = 0.0f;
                float g = 0.0f;
                float b = 0.0f;
                avgColour(imgData, subPixidx, r, g, b, height, width);
                assignColourToTri(imgData, subPixidx, r, g, b, height, width);
                oldtt = currtt;
            }
        }
    }
}
