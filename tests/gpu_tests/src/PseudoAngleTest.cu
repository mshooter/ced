#include "gtest/gtest.h"
#include "Point.hpp"
#include "PseudoAngle.cuh"

TEST(PseudoAngle, pseudoangle)
{
    ced::cpu::Point p = {0.0f, 1.0f};
    float theta = pseudo_angle(p);
    EXPECT_EQ(theta, 1.0f/4.0f);
}
