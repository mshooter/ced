#include "gtest/gtest.h"
#include "GenerateRandomPoints.hpp"

TEST(GenerateRandomPoints, generateRandomPoints)
{
    std::vector<ced::cpu::Point> p = ced::cpu::generateRandomPoints(5, 10, 10);
    int sizeCoords = p.size();
    EXPECT_EQ(5, sizeCoords);
}
//--------------------------------------------------------------------------------
TEST(GenerateRandomPoints, pointVectorSize)
{
    const int numberOfPoints = 5; 
    int height = 10;
    int width = 10; 
    std::vector<ced::cpu::Point> m_coordinates(numberOfPoints); 
    for(int i=0; i < 5; ++i)
    {
        m_coordinates[i] = (ced::cpu::Point(rand()%height, rand()%width)); 
    }
    int sizeCoords = m_coordinates.size();
    EXPECT_EQ(numberOfPoints, sizeCoords);
}
