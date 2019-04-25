#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <vector>

TEST(AssignPixToTri, assignPixToTriThrust)
{
    // -------------host---------------------
    std::vector<float> h_values = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};    
    // min and max coordinates
    // this will give us a total amount of 8 pixels
    int minx = 1;
    int maxx = 4;
    int miny = 1;
    int maxy = 5;
    ASSERT_EQ((maxx)*(maxy), 12); 
    // the width of my image is 7
    // is formula x + y * width
    int start_itr = minx + miny * 7;
    ASSERT_EQ(start_itr,8);
    int end_itr = maxx + maxy * 7; 
    ASSERT_EQ(end_itr, 8);
    // -------------device-------------------
    thrust::device_vector<float> d_values = h_values;
    thrust::device_vector<int> d_indices(h_values.size());
    
    thrust::counting_iterator<int> itr
}
