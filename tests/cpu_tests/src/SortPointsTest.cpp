#include "gtest/gtest.h"
#include "SortPoints.hpp"

TEST(sortPoints, partition)
{
    ced::cpu::Point p1(0,0);
    ced::cpu::Point p2(1,0);
    ced::cpu::Point p3(2,0);
    ced::cpu::Point p4(3,0);
    std::vector<ced::cpu::Point> unsortedList = {p4, p3, p2, p1};
    // choose pivot element to be placed on the right position 
    ced::cpu::Point pivotElement = unsortedList[0];
    int i = 0;
    for(int j=0+1; j <= 3; ++j)
    {
        if(unsortedList[j].x < pivotElement.x)
        {
            i++;
            std::swap(unsortedList[i], unsortedList[j]);
        }
        if(unsortedList[j].x == pivotElement.x)
        {
            if(unsortedList[j].y < pivotElement.y)
            {
                i++;
                std::swap(unsortedList[i], unsortedList[j]);
            }
        }
    }
    std::swap(unsortedList[i], unsortedList[0]);
    EXPECT_EQ(i, 3);
    EXPECT_EQ(unsortedList[0].x, 0);
    EXPECT_EQ(unsortedList[1].x, 2);
    EXPECT_EQ(unsortedList[2].x, 1);
    EXPECT_EQ(unsortedList[3].x, 3);
}

TEST(sortPoints, partitionFunction)
{
    ced::cpu::Point p1(1,0);
    ced::cpu::Point p2(1,2);
    ced::cpu::Point p3(2,0);
    ced::cpu::Point p4(3,0);
    std::vector<ced::cpu::Point> unsortedList = {p4, p3, p2, p1};
    int i = ced::cpu::partition(unsortedList, 0, 3);
    EXPECT_EQ(i, 3);
}


TEST(sortPoints, quickSortFunction)
{
    ced::cpu::Point p1(3,0);
    ced::cpu::Point p2(3,1);
    ced::cpu::Point p3(20,0);
    ced::cpu::Point p4(21,0);
    std::vector<ced::cpu::Point> unsortedList = {p4, p3, p2, p1};
    ced::cpu::quickSort(unsortedList, 0, 3);
    EXPECT_EQ(unsortedList[0].x, 3);
    EXPECT_EQ(unsortedList[0].y, 0);

    EXPECT_EQ(unsortedList[1].x, 3);
    EXPECT_EQ(unsortedList[1].y, 1);

    EXPECT_EQ(unsortedList[2].x, 20);
    EXPECT_EQ(unsortedList[3].x, 21);
}
