#include <algorithm>
#include <limits> 
#include <stdexcept> 

#include "Triangulation.hpp"
#include "params.hpp"
#include "Compare.hpp"
#include "Distance2P.hpp"
#include "CircumCircle.hpp"
#include "TriOrientation.hpp"
#include "SortPoints.hpp"
#include "Pseudoangle.hpp"

namespace ced
{
    namespace cpu
    {
        void triangulate(std::vector<Point> const& _points)
        {
            unsigned int ptsListSize = _points.size();
            // id list
            std::vector<unsigned int> ids;
            ids.reserve(ptsListSize);
            // max and min
            std::vector<unsigned int> triangles; // where edge starts
            std::vector<unsigned int> halfedges = {INVALID_IDX, INVALID_IDX, INVALID_IDX}; // reverse of edge 
            std::vector<unsigned int> hull_prev; // edge to the previous edge 
            std::vector<unsigned int> hull_next; // edge to the next edge 
            std::vector<unsigned int> hull_tri; // edge to adjacent triangle   
            std::vector<unsigned int> hash; // angular edge hash 
            // init a hash table for storing edges of the advancing convex hull
            int hash_size = static_cast<unsigned int>(std::llround(std::ceil(std::sqrt(ptsListSize))));            
            hash.resize(hash_size);
            std::fill(hash.begin(), hash.end(), INVALID_IDX);
            // init arrays for tracking edges of advancing convex hull
            hull_prev.resize(ptsListSize);
            hull_next.resize(ptsListSize);
            hull_tri.resize(ptsListSize);
            // fill ids
            for(unsigned int i=0; i < ptsListSize; ++i)
            {
                ids.push_back(i);   
            }
            // need to calculate centroid 
            Point sc = calculateCentroidCenter(_points);
            unsigned int i0 = INVALID_IDX; 
            unsigned int i1 = INVALID_IDX; 
            unsigned int i2 = INVALID_IDX; 
            createFirstTri(_points, i0, i1, i2, sc);            
            Point pi0 = _points[i0];
            Point pi1 = _points[i1];
            Point pi2 = _points[i2];
            // orientation
            if(isCCW<float>(pi0, pi1, pi2))
            {
                std::swap(i1, i2);
                std::swap(pi1, pi2);
            }
            // circumcenter
            Point cc = circumCenter(pi0, pi1, pi2);
            // sort the points by distance from the seed triangle circumcenter
            quickSortDist<float>(ids, _points, cc, 0, ptsListSize - 1); 
            unsigned int hull_start = i0;
            unsigned int hullSize = 3;
            hull_next[i0] = hull_prev[i2] = i1;
            hull_next[i1] = hull_prev[i0] = i2;
            hull_next[i2] = hull_prev[i1] = i0;
            hull_tri[i0] = 0;
            hull_tri[i1] = 1;
            hull_tri[i2] = 2;
            hash[hash_key(pi0, cc, hash_size)] = i0;
            hash[hash_key(pi1, cc, hash_size)] = i1;
            hash[hash_key(pi2, cc, hash_size)] = i2;
            uint max_triangles = ptsListSize < 3 ? 1 : 2 * ptsListSize - 5; 
            triangles.reserve(max_triangles * 3); 
            halfedges.reserve(max_triangles * 3); 
            add_triangle(i0, i1, i2, INVALID_IDX, INVALID_IDX, INVALID_IDX, triangles, halfedges);
            // iterate over indices
            Point pp = std::numeric_limits<Point>::quiet_NaN();
            for(auto& id : ids)
            {
                uint k = &id - &ids[0]; 
                const uint idx = id; 
                const Point p = _points[idx]; 
                // skip near duplicates
                if(idx != 0 && equalPts(p, pp)) continue;
            }
        }
        //  --------------------------------------------------------------------------------------
        Point calculateCentroidCenter(std::vector<Point> _pts)
        {
            float minx = (std::min_element(_pts.begin(), _pts.end(), compareX))->x; 
            float miny = (std::min_element(_pts.begin(), _pts.end(), compareY))->y; 
            float maxx = (std::max_element(_pts.begin(), _pts.end(), compareX))->x; 
            float maxy = (std::max_element(_pts.begin(), _pts.end(), compareY))->y; 

            float cx = (minx + maxx) / 2.0f;
            float cy = (miny + maxy) / 2.0f;
            
            return Point(cx, cy);
        }
        //  --------------------------------------------------------------------------------------
        void createFirstTri(std::vector<Point> pts, unsigned int& i0, unsigned int& i1, unsigned int& i2, Point centroid)
        {
            // seed point close to centroid
            float min_dist = std::numeric_limits<float>::max();
            unsigned int i = 0;
            for(auto& pt : pts)
            {
                const float d = distance2P<float>(centroid, pt); 
                if(d < min_dist)
                {
                    i0 = i; 
                    min_dist = d;
                }
                ++i;
            }
            // find point close to seed
            min_dist = std::numeric_limits<float>::max();
            i = 0;
            for(auto& pt : pts)
            {
                if(i != i0)
                {
                    const float d = distance2P<float>(pts[i0], pt);
                    if(d < min_dist && d > 0.0f)
                    {
                        i1 = i;
                        min_dist = d;
                    }
                }
                ++i;
            } 
            // third point in circumcircle 
            float min_radius = std::numeric_limits<float>::max();
            i = 0;
            // find point close to seed
            for(auto& pt : pts)
            {
                if(i != i0 && i != i1)
                {
                    const float r  = circumRadius<float>(pts[i0], pts[i1], pt);
                    if(r < min_radius)
                    {
                        i2 = i;
                        min_radius = r;
                    }
                }
                ++i;
            } 
            if(!(min_radius < std::numeric_limits<float>::max()))
            {
                throw std::runtime_error("not triangulation possible");
            }
        }
        //  --------------------------------------------------------------------------------------
        unsigned int hash_key(Point p, Point cc, unsigned int hashSize)
        {
            Point pcc = p - cc; 
            float fangle = pseudo_angle(pcc) * static_cast<float>(hashSize); 
            int iangle = static_cast<int>(fangle);
            return (iangle > fangle ? --iangle : iangle);
        }
        //  --------------------------------------------------------------------------------------
        void link(const unsigned int idx1, const unsigned int idx2, std::vector<uint>& halfedges)
        {
            unsigned int s0 = halfedges.size();
            if(idx1 == s0)
            {
                halfedges.push_back(idx2);
            }
            else if(idx1 < s0)
            {
                halfedges[idx1] = idx2;
            }
            else
            {
                throw std::runtime_error("Cannot link edge");
            }

            if(idx2 != INVALID_IDX)
            {
                unsigned int s1 = halfedges.size();
                if(idx2 == s1)
                {
                    halfedges.push_back(idx1);
                }
                else if(idx2 < s1)
                {
                    halfedges[idx2] = idx1;
                }
                else
                {
                    throw std::runtime_error("Cannot link edge");
                }
            } 
        }
        //  --------------------------------------------------------------------------------------
        unsigned int add_triangle(uint i0, uint i1, uint i2, uint h_i0, uint h_i1, uint h_i2, std::vector<uint>& triangles, std::vector<uint>& halfedges)
        {
            uint triListSize = triangles.size();
            triangles.push_back(i0);
            triangles.push_back(i1);
            triangles.push_back(i2);
            link(triListSize + 0, h_i0, halfedges);
            link(triListSize + 1, h_i1, halfedges);
            link(triListSize + 2, h_i2, halfedges);
            return triListSize;
        } 
    }
}
