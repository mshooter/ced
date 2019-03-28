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
            std::vector<int> triangles; // where edge starts
            std::vector<int> halfedges; // reverse of edge 
            std::vector<int> hull_e_prev; // edge to the previous edge 
            std::vector<int> hull_e_next; // edge to the next edge 
            std::vector<int> hull_e_tri; // edge to adjacent triangle   
            std::vector<int> hash; // angular edge hash 
            int hull_start;
            int hash_size; 
            // assign ids to the points
            std::vector<int> ids;
            int sizeOfPoints = _points.size();
            ids.reserve(sizeOfPoints); 
            for(int i=0; i < sizeOfPoints; ++i)
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
            int i0 = std::numeric_limits<int>::max();
            int i1 = std::numeric_limits<int>::max();
            int i2 = std::numeric_limits<int>::max();

            // find seed
            float minDistance = std::numeric_limits<float>::max();
            int i = 0;
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
            quickSortDist<int, Point>(ids, _points, cc, 0, sizeOfPoints-1);

            // starting hull is the seed 
            hull_start = i0;
            int hull_size = 3;
            // setting up the hulls 
            hull_e_next[i0] = hull_e_prev[i2] = i1; 
            hull_e_next[i1] = hull_e_prev[i0] = i2;
            hull_e_next[i2] = hull_e_prev[i1] = i0;
            hull_e_tri[i0]= 0;
            hull_e_tri[i1]= 1;
            hull_e_tri[i2]= 2;
            //initialize hash_size
            hash_size = static_cast<int>(std::sqrt(sizeOfPoints));
            if(hash_size < std::sqrt(sizeOfPoints)) ++hash_size;
            hash.resize(hash_size);
            // hashkeys
            hash[hash_key(pi0, cc, hash_size)] = i0;
            hash[hash_key(pi1, cc, hash_size)] = i1;
            hash[hash_key(pi2, cc, hash_size)] = i2;     
            // reserve space otherwise it doesn't work  
            int max_tri = sizeOfPoints < 3 ? 1 : (2 * sizeOfPoints - 5);
            triangles.reserve(max_tri * 3);
            halfedges.reserve(max_tri * 3);
            // add a triangle
            addTriangle<int>(i0, i1, i2, -1, -1 ,-1, triangles, halfedges);
             
            
        }
        //  --------------------------------------------------------------------------------------
        void link(int _triangleID, int _halfedgeID, std::vector<int>& halfedges)
        {
            // there was a segmentatoin error its because of this the vector is not build yet
            halfedges[_triangleID] = _halfedgeID; 
            if(_halfedgeID != -1)   halfedges[_halfedgeID] = _triangleID; 
        }
    }
}
