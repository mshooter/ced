#include "Triangulation.hpp"
#include "Compare.hpp"
#include "Distance2P.hpp"
#include "TriOrientation.hpp"
#include "CircumCircle.hpp"
#include <algorithm>
#include <limits>

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
            Point c = {cx, cy};
            
            // assign indexes 
            unsigned int i0;
            unsigned int i1;
            unsigned int i2;

            // find seed
            float minDistance = std::numeric_limits<float>::max();
            unsigned int i = 0;
            for(auto point : _points)
            {
                float distance = distance2P<float, Point>(c, point);
                if(distance < minDistance)
                {
                   i0 = i;  
                   minDistance = distance;
                }
                ++i;
            } 
            
            // find point close to seed 
            minDistance = std::numeric_limits<float>::max(); 
            i = 0;
            for(auto point : _points)
            {
                if(i != i0)
                {
                    float distance = distance2P<float, Point>(_points[i0], point);
                    if(distance < minDistance && distance > 0.0f)
                    {
                        i1 = i;
                        minDistance = distance;
                    }
                }
                ++i;
            }

            // find point to create smallest CC with first two points 
            float minRadius = std::numeric_limits<float>::max();
            i = 0; 
            for(auto point : _points)
            {
                if(i != i0 && i != i1)
                {
                    float r = circumRadius(_points[i0], _points[i1], point);
                    if(r < minRadius)
                    {
                        i2 = i;
                        minRadius = r;
                    }
                }
                ++i;
            }
            // throw an error if the radius is equal to numeric_limit
            // check the orientation of the triangle
            if(isCCW<Point, float>(_points[i0], _points[i1], _points[i2]))
            {
                // swap index and points
                // std::swap();
                // std::swap();
            }
        }
    }
}
