#include "gtest/gtest.h"
#include "Distance2P.hpp"
#include "Point.hpp"

TEST(Distance2P, distanceFunction)
{
    using namespace ced::cpu;
    Point c = {2,1};
    Point p = {4,3};

    float d = distance2P<float, Point>(c, p);
    EXPECT_FLOAT_EQ(d, std::sqrt(8));
}
