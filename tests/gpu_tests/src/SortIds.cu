#include "gtest/gtest.h"

#include "Distance2P.cuh"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

TEST(SortIDs, createDistances)
{
    std::vector<float> h_x = {0.3f, 0.2f, 0.1f, 0.4f};
    std::vector<float> h_y = {0.3f, 0.2f, 0.1f, 0.4f};
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;
    thrust::device_vector<int> d_indices(4);
    thrust::sequence(d_indices.begin(), d_indices.end());
    thrust::device_vector<float> d_distances(4);
    float cx = 0.0f;
    float cy = 0.0f;
    thrust::transform(
                        thrust::make_zip_iterator(
                                                    thrust::make_tuple(d_x.begin(), d_y.begin())),
                        thrust::make_zip_iterator( 
                                                    thrust::make_tuple(d_x.end(), d_y.end())),
                        d_distances.begin(),
                        ced::gpu::distance2P<float>(cx, cy));
    EXPECT_FLOAT_EQ(d_distances[0], 0.18f);
    EXPECT_FLOAT_EQ(d_distances[1], 0.08f);
    EXPECT_FLOAT_EQ(d_distances[2], 0.02f);
    EXPECT_FLOAT_EQ(d_distances[3], 0.32f);
    thrust::sort_by_key(d_distances.begin(), d_distances.end(), d_indices.begin(), thrust::less<float>()); 
    EXPECT_EQ(d_indices[0], 2); 
    EXPECT_EQ(d_indices[1], 1); 
    EXPECT_EQ(d_indices[2], 0); 
    EXPECT_EQ(d_indices[3], 3); 
}
