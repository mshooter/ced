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
            std::vector<unsigned int> halfedges; // reverse of edge 
            std::vector<unsigned int> hull_prev; // edge to the previous edge 
            std::vector<unsigned int> hull_next; // edge to the next edge 
            std::vector<unsigned int> hull_tri; // edge to adjacent triangle   
            std::vector<unsigned int> hash; // angular edge hash 
            std::vector<unsigned int> edge_stack; 
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
            unsigned int max_triangles = ptsListSize < 3 ? 1 : 2 * ptsListSize - 5; 
            triangles.reserve(max_triangles * 3); 
            halfedges.reserve(max_triangles * 3); 
            add_triangle(i0, i1, i2, INVALID_IDX, INVALID_IDX, INVALID_IDX, triangles, halfedges);
            // iterate over indices
            Point pp = std::numeric_limits<Point>::quiet_NaN();
            for(auto& id : ids)
            {
                unsigned int k = &id - &ids[0]; 
                const unsigned int i = id; 
                const Point p = _points[i]; 
                // skip near duplicates
                if(i > 0 && equalPts(p, pp)) continue;
                pp = p;
                // check java version seems simpler
                // skip seed triangle points
                if( equalPts(p, pi0) || equalPts(p, pi1) || equalPts(p, pi2))   continue;
                // find a visible edge on the convex hull using edge hash
                unsigned int start = 0;
                unsigned int key = hash_key(p, cc, hash_size);
                for(unsigned int j = 0; j < hash_size; ++j)
                {
                    start = hash[(key+j)%hash_size];
                    if(start != INVALID_IDX && start != hull_next[start]) break;
                } 
                start = hull_prev[start];
                unsigned int e = start;
                unsigned int q; 
                while(q = hull_next[e], !isCCW<float>(pp, _points[e], _points[q]))
                {
                    e = q;
                    if(e == start)
                    {
                        e = INVALID_IDX;
                        break;
                    }
                }
                if(e == INVALID_IDX) continue; // likeliy a near duplicate; skip it;
                // add first triangle from the point
                unsigned int t = add_triangle(t, i, hull_next[e], INVALID_IDX,  INVALID_IDX, hull_tri[e], triangles, halfedges);
                //hull_tri[i] = legalize(t+2);
                hull_tri[e] = t;
                ++hullSize;
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
            return (iangle > fangle ? --iangle : iangle) % hashSize;
        }
        //  --------------------------------------------------------------------------------------
        void link(const unsigned int idx1, const unsigned int idx2, std::vector<unsigned int>& halfedges)
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
        unsigned int add_triangle(unsigned int i0, unsigned int i1, unsigned int i2, unsigned int h_i0, unsigned int h_i1, unsigned int h_i2, std::vector<unsigned int>& triangles, std::vector<unsigned int>& halfedges)
        {
            unsigned int triListSize = triangles.size();
            triangles.push_back(i0);
            triangles.push_back(i1);
            triangles.push_back(i2);
            link(triListSize + 0, h_i0, halfedges);
            link(triListSize + 1, h_i1, halfedges);
            link(triListSize + 2, h_i2, halfedges);
            return triListSize;
        } 
        //  --------------------------------------------------------------------------------------
        unsigned int legalise(unsigned int a, std::vector<unsigned int>& edge_stack, std::vector<unsigned int> triangles, std::vector<unsigned int> halfedges, std::vector<Point> pts, unsigned int& hull_start, std::vector<unsigned int>& hull_next, std::vector<unsigned int>& hull_tri)
        {
            // drawing on the webstie
            unsigned int i = 0;
            unsigned int ar = 0;
            edge_stack.clear(); 
            while(true)
            {
                const unsigned int b = halfedges[a];
                const unsigned int a0 = 3 * (a / 3);
                ar = a0 + (a + 2) % 3;
                if(b == INVALID_IDX)
                {
                    if(i > 0)
                    {
                        --i;
                        a = edge_stack[i];
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
                const unsigned int b0 = 3 * (b/3);
                const unsigned int al = a0 + (a + 1) % 3;
                const unsigned int bl = b0 + (b + 2) % 3;
                const unsigned int p0 = triangles[ar];
                const unsigned int pr = triangles[a];
                const unsigned int pl = triangles[al];
                const unsigned int p1 = triangles[bl];
                const bool illegal = isPointInCircle<float>(pts[p0], pts[pr], pts[pl], pts[p1]);
                if(illegal)
                {
                    triangles[a] = p1;
                    triangles[b] = p0;
                    unsigned int hbl = halfedges[bl];
                    // edge swapped on the other side of the hull (rare); fix the halfedge reference
                    if(hbl == INVALID_IDX)
                    {
                        unsigned int e = hull_start; 
                        do
                        {
                            if(hull_tri[e] == bl)
                            {
                                hull_tri[e] = a; 
                                break;
                            }
                            e = hull_next[e];
                        }
                        while(e != hull_start);
                    }
                    link(a, hbl, halfedges);
                    link(b, halfedges[ar], halfedges);
                    link(ar, bl, halfedges);
                    unsigned int br = b0 + (b + 1) % 3;
                    if(i < edge_stack.size())
                        edge_stack[i] = br;
                    else
                        edge_stack.push_back(br);
                    ++i;
                }
                else
                {
                    if(i > 0)
                    {
                        --i;
                        a = edge_stack[i];
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            return ar;
        }
    }
}
