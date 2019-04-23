#include "Triangulation.cuh"

#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>

#include "Distance2P.cuh"
#include "CircumCircle.cuh"
#include "ThrustFunctors.cuh"

namespace ced
{
    namespace gpu
    {
        __host__ void createFirstTri(
            const thrust::device_vector<Point>& d_pts,
            int& i0, 
            int& i1, 
            int& i2, 
            const Point& centroid )
        {
            // first point
            thrust::device_vector<float> d_dist(d_pts.size());
            thrust::transform(thrust::device, d_pts.begin(), d_pts.end(), d_dist.begin(), distance2P<float>(centroid)); 
            i0 = thrust::min_element(thrust::device, d_dist.begin(), d_dist.end()) - d_dist.begin();

            // second point
            thrust::device_vector<int> d_itr(d_pts.size());
            thrust::sequence(d_itr.begin(), d_itr.end());
            auto start  = thrust::make_zip_iterator(thrust::make_tuple(d_dist.begin(), d_itr.begin()));
            auto finish = thrust::make_zip_iterator(thrust::make_tuple(d_dist.end(), d_itr.end()));
            i1 = thrust::min_element(start, finish, min_if(i0, -1)) - start;

            // third point             
            thrust::device_vector<float> d_rad(d_pts.size());
            d_itr.resize(d_pts.size()); 
            thrust::sequence(d_itr.begin(), d_itr.end());
            Point v0 = d_pts[i0];
            Point v1 = d_pts[i1];
            thrust::transform(thrust::device, d_pts.begin(), d_pts.end(), d_rad.begin(), circumRadius(v0, v1));
            start  = thrust::make_zip_iterator(thrust::make_tuple(d_rad.begin(), d_itr.begin()));
            finish = thrust::make_zip_iterator(thrust::make_tuple(d_rad.end(), d_itr.end()));
            i2 = thrust::min_element(start, finish, min_if(i0, i1)) - start;
        }
    }
}
