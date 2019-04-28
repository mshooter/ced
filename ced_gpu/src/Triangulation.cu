#include "Triangulation.cuh"

#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>

#include "Distance2P.cuh"
#include "CCW.cuh"
#include "CircumCircle.cuh"
#include "ThrustFunctors.cuh"
#include "CalculateCentroid.cuh"

namespace ced
{
    namespace gpu
    {
        __host__ void createFirstTri(
            const thrust::device_vector<float>& d_x,
            const thrust::device_vector<float>& d_y,
            int& i0, 
            int& i1, 
            int& i2, 
            const float& cx, 
            const float& cy)
        {
            // first point
            thrust::device_vector<float> d_dist(d_x.size());
            thrust::transform(  thrust::device, 
                                thrust::make_zip_iterator(
                                                            thrust::make_tuple(d_x.begin(), d_y.begin())),
                                thrust::make_zip_iterator(
                                                            thrust::make_tuple(d_x.end(), d_y.end())),
                                d_dist.begin(), 
                                distance2P<float>(cx, cy)); 
            i0 = thrust::min_element(thrust::device, d_dist.begin(), d_dist.end()) - d_dist.begin();

            // second point
            thrust::device_vector<int> d_itr(d_x.size());
            thrust::sequence(d_itr.begin(), d_itr.end());
            auto start  = thrust::make_zip_iterator(thrust::make_tuple(d_dist.begin(), d_itr.begin()));
            auto finish = thrust::make_zip_iterator(thrust::make_tuple(d_dist.end(), d_itr.end()));
            i1 = thrust::min_element(start, finish, min_if(i0, -1)) - start;

            // third point             
            thrust::device_vector<float> d_rad(d_x.size());
            d_itr.resize(d_x.size()); 
            thrust::sequence(d_itr.begin(), d_itr.end());
            float v0x = d_x[i0];
            float v0y = d_y[i0];
            float v1x = d_x[i1];
            float v1y = d_y[i1];
            thrust::transform(  thrust::device, 
                                thrust::make_zip_iterator(
                                                            thrust::make_tuple(d_x.begin(), d_y.begin())),
                                thrust::make_zip_iterator(
                                                            thrust::make_tuple(d_x.end(), d_y.end())),
                                d_rad.begin(), 
                                circumRadius(v0x, v0y, v1x, v1y));
            start  = thrust::make_zip_iterator(thrust::make_tuple(d_rad.begin(), d_itr.begin()));
            finish = thrust::make_zip_iterator(thrust::make_tuple(d_rad.end(), d_itr.end()));
            i2 = thrust::min_element(start, finish, min_if(i0, i1)) - start;
        }   
        //  ----------------------------------------------------------------------------------------------
        __host__ void triangulate(thrust::device_vector<float>& y, thrust::device_vector<float>& x, thrust::device_vector<int>& triangles)
        {
            // init all the variables
            // the amount of points
            int sizeOfPoints = x.size();
            int maxTris = sizeOfPoints < 3 ? 1 : 2 * sizeOfPoints - 5;
            int hashSize = static_cast<int>(std::llround(std::ceil(std::sqrt(sizeOfPoints))));
            thrust::device_vector<int> d_ids(sizeOfPoints);
            thrust::sequence(d_ids.begin(), d_ids.end());
            thrust::device_vector<int> d_hull_prev(sizeOfPoints); 
            thrust::device_vector<int> d_hull_next(sizeOfPoints); 
            thrust::device_vector<int> d_hull_tri(sizeOfPoints); 
            thrust::device_vector<int> d_halfedges(maxTris * 3); 
            triangles.reserve(maxTris * 3);
            thrust::device_vector<int> d_hash(hashSize, -1); 
            // need to have a size for the edge stack!
            // therefor right now we have a host vector to dynamically set size
            std::vector<int> h_edge_stack;
            thrust::device_vector<int> d_edge_stack; 
            float scx = calculateCentroidx(x); 
            float scy = calculateCentroidy(y); 
            // initiate invalid index
            int i0 = -1; 
            int i1 = -1; 
            int i2 = -1; 
            // create first triangle
            createFirstTri(x, y, i0, i1, i2, scx, scy);
            float pi0x = x[i0];
            float pi1x = x[i1];
            float pi2x = x[i2];

            float pi0y = y[i0];
            float pi1y = y[i1];
            float pi2y = y[i2];
            // check the orientation of the triangle
            if(CCW( pi0x,
                    pi0y,
                    pi1x,
                    pi1y,
                    pi2x,
                    pi2y) < 0)
            {
                thrust::swap(i1, i2);
                thrust::swap(pi1x, pi2x);
                thrust::swap(pi1y, pi2y);
            }
            // find the center of the circumcircle of the three points
            float cx = 0.0f;
            float cy = 0.0f;
            circumCenter(   pi0x,
                            pi0y,
                            pi1x,
                            pi1y,
                            pi2x,
                            pi2y,
                            cx, 
                            cy); 
            // create a vector with distances 
            // then based on the distances, sort by key the ids
            thrust::device_vector<float> d_distances(sizeOfPoints);
            thrust::transform(
                                thrust::make_zip_iterator(
                                                            thrust::make_tuple(x.begin(), y.begin())),
                                thrust::make_zip_iterator( 
                                                            thrust::make_tuple(x.end(), y.end())),
                                d_distances.begin(),
                                distance2P<float>(cx, cy));
            thrust::sort_by_key(d_distances.begin(), d_distances.end(), d_ids.begin(), thrust::less<float>());
            // init the hulls and hullsize
            int hull_start = i0;
            int hullSize = 3;
            
            
            
        }
    }
}
