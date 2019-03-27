#include <algorithm>
#include <limits>

#include "Triangulation.hpp"
#include "Compare.hpp"
#include "Distance2P.hpp"
#include "TriOrientation.hpp"
#include "CircumCircle.hpp"
#include "SortPoints.hpp"

namespace ced
{
    namespace cpu
    {
        void triangulate(std::vector<Point>& _points)
        {
            triangulation tri; 
            // assign ids to the points
            std::vector<unsigned int> ids;
            unsigned int sizeOfPoints = _points.size();
            ids.reserve(sizeOfPoints); 
            for(unsigned int i=0; i < sizeOfPoints; ++i)
            {
                ids.push_back(i);
            }

            // look for max/min x and max/min y
            float minx = (std::min_element(_points.begin(), _points.end(), compareX))->x;  
            float maxx = (std::max_element(_points.begin(), _points.end(), compareX))->x;  
            float miny = (std::min_element(_points.begin(), _points.end(), compareY))->y;  
            float maxy = (std::max_element(_points.begin(), _points.end(), compareY))->y;  

            // seed center calculation
            float cx = (minx + maxx) / 2.0f;
            float cy = (miny + maxy) / 2.0f;
            Point sc = {cx, cy};
            
            // assign indexes 
            unsigned int i0 = std::numeric_limits<unsigned int>::max();
            unsigned int i1 = std::numeric_limits<unsigned int>::max();
            unsigned int i2 = std::numeric_limits<unsigned int>::max();

            // find seed
            float minDistance = std::numeric_limits<float>::max();
            unsigned int i = 0;
            for(auto point : _points)
            {
                float distance = distance2P<float, Point>(sc, point);
                if(distance < minDistance)
                {
                   i0 = i;  
                   minDistance = distance;
                }
                ++i;
            } 

            Point pi0 = _points[i0];
            // find point close to seed 
            minDistance = std::numeric_limits<float>::max(); 
            i = 0;
            for(auto point : _points)
            {
                if(i != i0)
                {
                    float distance = distance2P<float, Point>(pi0, point);
                    if(distance < minDistance && distance > 0.0f)
                    {
                        i1 = i;
                        minDistance = distance;
                    }
                }
                ++i;
            }
            
            Point pi1 = _points[i1];
            // find point to create smallest CC with first two points 
            float minRadius = std::numeric_limits<float>::max();
            i = 0; 
            for(auto point : _points)
            {
                if(i != i0 && i != i1)
                {
                    float r = circumRadius<Point, float>(pi0, pi1, point);
                    if(r < minRadius)
                    {
                        i2 = i;
                        minRadius = r;
                    }
                }
                ++i;
            }

            Point pi2 = _points[i2];
            // throw an error if the radius is equal to numeric_limit
            // if the orientation is CCW swap set the point values to ccw
            if(isCCW<Point, float>(pi0, pi1, pi2))
            {
                // swap index and points
                std::swap(i1, i2);
                std::swap(pi1, pi2);
            }
            
            // init the circum center
            Point cc = circumCenter(_points[i0], _points[i1], _points[i2]); 

            // sort idpts by distance from the seed triangle circumcenter
            quickSortDist<unsigned int, Point>(ids, _points, cc, 0, sizeOfPoints-1);

            // starting hull is the seed 
            tri.hull_start = i0;
            unsigned int hull_size = 3;
            
            
        }
    }
}
