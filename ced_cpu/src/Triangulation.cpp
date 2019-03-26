#include "Triangulation.hpp"
#include "Compare.hpp"

#include <algorithm>
#include <limits>
#include <cmath>

namespace ced
{
    namespace cpu
    {
        void triangulation::triangulate(std::vector<Point>& _points)
        {
            // assign ids to the points
            std::vector<unsigned int> ids;
            unsigned int sizeOfPoints = _points.size();
            ids.reserve(sizeOfPoints); 
            for(unsigned int i=0; i < sizeOfPoints; ++i)
            {
                ids.push_back(i);
            }
            //for(auto x : ids)
            //{
            //    std::cout<<x<<std::endl;
            //}

            // look for max/min x and max/min y
            float minx = (std::min_element(_points.begin(), _points.end(), compareX))->x;  
            float maxx = (std::max_element(_points.begin(), _points.end(), compareX))->x;  
            float miny = (std::min_element(_points.begin(), _points.end(), compareY))->y;  
            float maxy = (std::max_element(_points.begin(), _points.end(), compareY))->y;  

            // seed center calculation
            float cx = (minx + maxx) / 2.0f;
            float cy = (miny + maxy) / 2.0f;
            
            // assign indexes 
            unsigned int i0;
            unsigned int i1;
            unsigned int i2;

            // iterate over the points, find the closest point to the centroid
            float minDistance = std::numeric_limits<float>::max();
            unsigned int i = 0;
            for(auto point : _points)
            {
                float deltaX = cx - point.x;
                float deltaY = cy - point.y;
                float distance = std::sqrt((deltaX * deltaX) + (deltaY*deltaY));
                if(distance < minDistance)
                {
                   i0 = i;  
                   minDistance = distance;
                }
                ++i;
            } 
        }
    }
}
