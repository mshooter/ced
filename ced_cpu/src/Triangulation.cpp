#include <algorithm>
#include <limits> 
#include <stdexcept> 

#include "Triangulation.hpp"
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
        void triangulate(std::vector<Point> const& _points, std::vector<int>& triangles)
        {
            int ptsListSize = _points.size();
            int max_triangles = ptsListSize < 3 ? 1 : 2 * ptsListSize - 5; 
            triangles.reserve(max_triangles * 3); 

            // id list
            std::vector<int> ids(ptsListSize);
            // max and min
            std::vector<int> halfedges(max_triangles*3); // reverse of edge 
            std::vector<int> hull_prev(ptsListSize); // edge to the previous edge 
            std::vector<int> hull_next(ptsListSize); // edge to the next edge 
            std::vector<int> hull_tri(ptsListSize); // edge to adjacent triangle   
            int hash_size = static_cast<int>(std::llround(std::ceil(std::sqrt(ptsListSize))));            
            std::vector<int> hash(hash_size); // angular edge hash 
            std::vector<int> edge_stack; 
            std::fill(hash.begin(), hash.end(), -1);
            // fill ids
            for(int i=0; i < ptsListSize; ++i)
            {
                ids.push_back(i);   
            }
            // need to calculate centroid 
            // need to give it myself
            Point sc = calculateCentroidCenter(_points);
            int i0 = -1; 
            int i1 = -1; 
            int i2 = -1; 
            createFirstTri(_points, i0, i1, i2, sc);            
            Point pi0 = _points[i0];
            Point pi1 = _points[i1];
            Point pi2 = _points[i2];
            // orientation
            if((isCCW<float>(pi0, pi1, pi2)) < 0)
            {
                std::swap(i1, i2);
                std::swap(pi1, pi2);
            }
            // circumcenter
            Point cc = circumCenter(pi0, pi1, pi2);
            // sort the points by distance from the seed triangle circumcenter
            quickSortDist<float>(ids, _points, cc, 0, ptsListSize - 1); 
            int hull_start = i0;
            int hullSize = 3;
            hull_next[i0] = hull_prev[i2] = i1;
            hull_next[i1] = hull_prev[i0] = i2;
            hull_next[i2] = hull_prev[i1] = i0;
            hull_tri[i0] = 0;
            hull_tri[i1] = 1;
            hull_tri[i2] = 2;
            hash[hash_key(pi0, cc, hash_size)] = i0;
            hash[hash_key(pi1, cc, hash_size)] = i1;
            hash[hash_key(pi2, cc, hash_size)] = i2;

            add_triangle(i0, i1, i2, -1, -1, -1, triangles, halfedges);
            // iterate over indices
            Point pp = std::numeric_limits<Point>::quiet_NaN();
            for(auto& id : ids)
            {
                int k = &id - &ids[0]; 
                const int i = id; 
                const Point p = _points[i]; 
                // skip near duplicates
                if(k > 0 && equalPts(p, pp)) continue;
                pp = p;
                // check java version seems simpler
                // skip seed triangle points
                if( equalPts(p, pi0) || equalPts(p, pi1) || equalPts(p, pi2))   continue;
                // find a visible edge on the convex hull using edge hash
                int start = 0;
                int key = hash_key(p, cc, hash_size);
                for(int j = 0; j < hash_size; ++j)
                {
                    start = hash[(key+j)%hash_size];
                    if(start != -1 && start != hull_next[start]) break;
                } 
                start = hull_prev[start];
                int e = start;
                int q; 
                while(q = hull_next[e], (!(isCCW<float>(pp, _points[e], _points[q])<0)))
                {
                    e = q;
                    if(e == start)
                    {
                        e = -1;
                        break;
                    }
                }
                if(e == -1) continue; // likeliy a near duplicate; skip it;
                // add first triangle from the point
                int t = add_triangle(e, i, hull_next[e], -1,  -1, hull_tri[e], triangles, halfedges);
                hull_tri[i] = legalise(t+2, edge_stack, triangles, halfedges, hull_next, hull_tri, _points, hull_start);
                hull_tri[e] = t;
                ++hullSize;
                // walk forward through the hull, adding more triangles and flipping recursively
                int next = hull_next[e];
                while(q = hull_next[next], (isCCW<float>(pp, _points[next], _points[q])<0))
                {
                    t = add_triangle(next, i, q, hull_tri[i], -1, hull_tri[next], triangles, halfedges);
                    hull_tri[i] = legalise(t+2, edge_stack, triangles, halfedges, hull_next, hull_tri, _points, hull_start);
                    hull_next[next] = next; // mark as removed 
                    --hullSize;
                    next = q;
                }

                // walk backward through the hull, adding more triangles and flipping recursively
                if(e == start)
                {
                    while(q = hull_prev[e], (isCCW<float>(pp, _points[q], _points[e])<0))
                    {
                       t = add_triangle(q, i, e, -1, hull_tri[e], hull_tri[q], triangles, halfedges);
                       legalise(t+2, edge_stack, triangles, halfedges, hull_next, hull_tri, _points, hull_start);
                       hull_tri[q] = t; 
                       hull_next[e] = e; // mark as removed
                       --hullSize;
                       e = q;
                    }
                }
                // update the hull indices
                hull_prev[i] = e; 
                hull_start = e;  
                hull_prev[next] = i; 
                hull_next[e] = i;
                hull_next[i] = next;
                hash[hash_key(pp, cc, hash_size)] = i;
                hash[hash_key(_points[e], cc, hash_size)] = e;

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
        void createFirstTri(std::vector<Point> pts, int& i0, int& i1, int& i2, Point centroid)
        {
            // seed point close to centroid
            float min_dist = std::numeric_limits<float>::max();
            int i = 0;
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
        int hash_key(Point p, Point cc, int hashSize)
        {
            Point pcc = p - cc; 
            float fangle = pseudo_angle(pcc) * static_cast<float>(hashSize); 
            int iangle = static_cast<int>(fangle);
            return (iangle > fangle ? --iangle : iangle) % hashSize;
        }
        //  --------------------------------------------------------------------------------------
        void link(const int idx1, const int idx2, std::vector<int>& halfedges)
        {
            int s0 = halfedges.size();
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

            if(idx2 != -1)
            {
                int s1 = halfedges.size();
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
        int add_triangle(  int i0, 
                                    int i1, 
                                    int i2, 
                                    int h_i0, 
                                    int h_i1, 
                                    int h_i2, 
                                    std::vector<int>& triangles, 
                                    std::vector<int>& halfedges)
        {
            int triListSize = triangles.size();
            triangles.push_back(i0);
            triangles.push_back(i1);
            triangles.push_back(i2);
            link(triListSize + 0, h_i0, halfedges);
            link(triListSize + 1, h_i1, halfedges);
            link(triListSize + 2, h_i2, halfedges);
            return triListSize;
        } 
        //  --------------------------------------------------------------------------------------
        int legalise(  int a, 
                                std::vector<int>& edge_stack, 
                                std::vector<int> triangles, 
                                std::vector<int> halfedges, 
                                std::vector<int>& hull_next, 
                                std::vector<int>& hull_tri,
                                std::vector<Point> pts, 
                                int& hull_start )
        {
            int i = 0;
            int ar = 0;
            edge_stack.clear(); 
            while(true)
            {
                const int b = halfedges[a];
                const int a0 = 3 * (a / 3);
                ar = a0 + (a + 2) % 3;
                if(b == -1)
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
                const int b0 = 3 * (b/3);
                const int al = a0 + (a + 1) % 3;
                const int bl = b0 + (b + 2) % 3;
                const int p0 = triangles[ar];
                const int pr = triangles[a];
                const int pl = triangles[al];
                const int p1 = triangles[bl];
                const bool illegal = isPointInCircle<float>(pts[p0], pts[pr], pts[pl], pts[p1]);
                if(illegal)
                {
                    triangles[a] = p1;
                    triangles[b] = p0;
                    int hbl = halfedges[bl];
                    // edge swapped on the other side of the hull (rare); fix the halfedge reference
                    if(hbl == -1)
                    {
                        int e = hull_start; 
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
                    int br = b0 + (b + 1) % 3;
                    int edge_stack_size = edge_stack.size();
                    if(i < edge_stack_size)
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
