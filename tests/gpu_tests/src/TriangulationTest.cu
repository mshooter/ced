#include "gtest/gtest.h"
#include "Distance2P.cuh"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <vector>

using namespace ced::gpu;
// create first triangle
TEST(Triangulation, findPointClosestToSeed)
{
    float cx  = 1.0f;
    float cy  = 0.5f;
    float2 cc = make_float2(cx, cy);
    float2 p0 = make_float2(0.0f, 0.0f);
    float2 p1 = make_float2(1.0f, 1.0f);
    float2 p2 = make_float2(2.0f, 0.0f);
    std::vector<float2> h_pts = {p0, p1, p2};
    thrust::device_vector<float2> d_pts = h_pts;
    thrust::device_vector<float> d_dist(h_pts.size());

    thrust::transform(thrust::device, d_pts.begin(), d_pts.end(), d_dist.begin(), distance2P<float>(cc));
    std::vector<float> h_dist(d_dist.begin(), d_dist.end());
    ASSERT_FLOAT_EQ(h_dist[0], 1.25f);
    ASSERT_FLOAT_EQ(h_dist[1], 0.25f);
    ASSERT_FLOAT_EQ(h_dist[2], 1.25f);
    int i0 = thrust::min_element(thrust::device, d_dist.begin(), d_dist.end()) - d_dist.begin();
    EXPECT_EQ(i0, 1);
}
struct min_if
{
    const int i0; 
    const int i1; 
    min_if(int _i0, int _i1) : i0(_i0), i1(_i1){}
    __host__ __device__
    bool operator()(const thrust::tuple<float, int>& curr, const thrust::tuple<float, int>& rhs)
    {
        if((curr.get<1>() != i0) && (curr.get<1>() != i1))
        {   
            return curr.get<0>() < rhs.get<0>();
        }
        else
        {
            return false;
        }
    }
};
// wondering if i cant take the two points closest to the seed ?? 
TEST(Triangulation, findSecondPointClosestToSeed)
{
    // init host
    float cx  = 1.0f;
    float cy  = 0.5f;
    int i0 = 1;
    float2 cc = make_float2(cx, cy);
    float2 p0 = make_float2(0.0f, 0.0f);
    float2 p1 = make_float2(1.0f, 1.0f);
    float2 p2 = make_float2(2.0f, 0.0f);
    std::vector<float2> h_pts = {p0, p1, p2};
    // init device
    thrust::device_vector<float2> d_pts = h_pts;
    thrust::device_vector<float> d_dist(h_pts.size());
    thrust::device_vector<int> d_itr(h_pts.size());
    thrust::sequence(d_itr.begin(), d_itr.end());
    std::vector<int> h_itr(d_itr.begin(), d_itr.end());
    ASSERT_EQ(h_itr[0],0);
    ASSERT_EQ(h_itr[1],1);
    ASSERT_EQ(h_itr[2],2);
    // manip
    thrust::transform(thrust::device, d_pts.begin(), d_pts.end(), d_dist.begin(), distance2P<float>(cc));
    // convert to host again
    std::vector<float> h_dist(d_dist.begin(), d_dist.end());
    // test
    ASSERT_FLOAT_EQ(h_dist[0], 1.25f);
    ASSERT_FLOAT_EQ(h_dist[1], 0.25f);
    ASSERT_FLOAT_EQ(h_dist[2], 1.25f);
    // check if second one is closer
    auto start  = thrust::make_zip_iterator(thrust::make_tuple(d_dist.begin(), d_itr.begin()));
    auto finish = thrust::make_zip_iterator(thrust::make_tuple(d_dist.end(), d_itr.end()));
    int i1 = thrust::min_element(start, finish, min_if(i0, std::numeric_limits<float>::max())) - start;
    EXPECT_EQ(i1, 0);
}
struct circumRadius
{
    const float2 A;
    const float2 B;
    circumRadius(float2 _p0, float2 _p1) : A(_p0), B(_p1) {}
    __host__ __device__
    float operator()(const float2& C)
    {
        float delta_abx = B.x-A.x;
        float delta_aby = B.y-A.y;
        float delta_acx = C.x-A.x;
        float delta_acy = C.y-A.y;
        
        const float dist_ab = delta_abx * delta_abx + delta_aby * delta_aby;
        const float dist_ac = delta_acx * delta_acx + delta_acy * delta_acy;
        const float N = delta_abx * delta_acy - delta_aby * delta_acx;

        const float x = (delta_acy * dist_ab - delta_aby * dist_ac) * 0.5f / N; 
        const float y = (delta_acx * dist_ac - delta_acx * dist_ab) * 0.5f / N; 

        //this is weird ? must check
        if(dist_ab != 0 && dist_ac != 0 && N != 0)
        {
            return x * x + y * y;
        }
        else
        {
            return std::numeric_limits<float>::max();
        } 
    }
};

TEST(Triangulation, findThirdPointCircum)
{
    float cx  = 1.0f;
    float cy  = 0.5f;
    int i0 = 1;
    int i1 = 0;
    float2 cc = make_float2(cx, cy);
    float2 p0 = make_float2(0.0f, 0.0f);
    float2 p1 = make_float2(1.0f, 1.0f);
    float2 p2 = make_float2(2.0f, 0.0f);
    std::vector<float2> h_pts = {p0, p1, p2};
    // init device
    thrust::device_vector<float2> d_pts = h_pts;
    thrust::device_vector<float> d_rad(h_pts.size());
    thrust::device_vector<int> d_itr(h_pts.size());
    thrust::sequence(d_itr.begin(), d_itr.end());
    std::vector<int> h_itr(d_itr.begin(), d_itr.end());
    ASSERT_EQ(h_itr[0],0);
    ASSERT_EQ(h_itr[1],1);
    ASSERT_EQ(h_itr[2],2);
    float2 v0 = d_pts[i0];
    float2 v1 = d_pts[i1];
    // calculate all circumradiuses of three points, of the two first points you find and then the current point
    thrust::transform(thrust::device, d_pts.begin(), d_pts.end(), d_rad.begin(), circumRadius(v0, v1));
    std::vector<float> h_rad(d_rad.begin(), d_rad.end());
    float err = std::numeric_limits<float>::max();
    ASSERT_FLOAT_EQ(h_rad[0], err);
    ASSERT_FLOAT_EQ(h_rad[1], err);
    ASSERT_FLOAT_EQ(h_rad[2], 0.0f);
    auto start  = thrust::make_zip_iterator(thrust::make_tuple(d_rad.begin(), d_itr.begin()));
    auto finish = thrust::make_zip_iterator(thrust::make_tuple(d_rad.end(), d_itr.end()));
    int i2 = thrust::min_element(start, finish, min_if(i0, i1)) - start;
    EXPECT_EQ(i2, 2);
}

#include "Triangulation.cuh"
TEST(Triangulation, functiongpu)
{
    float cx  = 1.0f;
    float cy  = 0.5f;
    int i0 = -1;
    int i1 = -1;
    int i2 = -1;
    float2 cc = make_float2(cx, cy);
    float2 p0 = make_float2(0.0f, 0.0f);
    float2 p1 = make_float2(1.0f, 1.0f);
    float2 p2 = make_float2(2.0f, 0.0f);
    std::vector<float2> h_pts = {p0, p1, p2};
    thrust::device_vector<float2> d_pts = h_pts;
    createFirstTri(d_pts, i0, i1, i2, cc); 
    EXPECT_EQ(i0,1);
    EXPECT_EQ(i1,0);
    EXPECT_EQ(i2,2);
}
