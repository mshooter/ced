#include "gtest/gtest.h"
#include "Edge.hpp"

TEST(Edge, edgeConstructor)
{
    ced::cpu::Point* p1 = new ced::cpu::Point(0,0);
    ced::cpu::Point* p2 = new ced::cpu::Point(1,0);
    ced::cpu::Edge edge(p1, p2);
    
    ASSERT_EQ(edge.startPoint->x, 0);
    ASSERT_EQ(edge.startPoint->y, 0);

    ASSERT_EQ(edge.endPoint->x, 1);
    ASSERT_EQ(edge.endPoint->y, 0);
}
