#include "AssignTriToPix.hpp"
#include "Distance2P.hpp"
namespace ced
{
    namespace cpu 
    {
        void assignTriToPix(    const int& height, 
                                const int& width, 
                                const std::vector<Point>& _mpts, 
                                std::vector<unsigned int>& _pixTriIdx)
        {
            int sizeImage = height * width;
            for(int i = 0; i < sizeImage; ++i)
            {
                int w = i / 3;
                int h = i % 3;
                std::vector<float> distances;
                for(unsigned int t = 0 ; t< _mpts.size(); ++t)
                {
                    // find smallest distance
                    float d = distance2P<float>(Point(w,h), _mpts[t]); 
                    distances.push_back(d);
                    
                }
                auto min = std::min_element(distances.begin(), distances.end());
                unsigned int idx = std::distance(distances.begin(), min);
                _pixTriIdx.push_back(idx);

            }
        }
    }
}
