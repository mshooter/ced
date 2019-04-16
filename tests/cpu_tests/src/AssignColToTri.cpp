#include "gtest/gtest.h"
#include <iostream>

TEST(AssignColToTri, getPixDepKey)
{
    std::multimap<unsigned int, unsigned int> pixTri ={
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 4},
        {1, 5},
        {1, 6},
        {1, 7},
    };

    typedef std::multimap<unsigned int, unsigned int>::iterator pixTriItr;
    std::pair<pixTriItr, pixTriItr> result = pixTri.equal_range(1);
    // get the values
    //for(pixTriItr it = result.first; it != result.second; ++it)
    //{
    //    std::cout<<it->second<<std::endl;
    //}
    int count = std::distance(result.first, result.second);
    EXPECT_EQ(count, 4);
 
}
#include "AvgColour.hpp"
#include "AssignColToTri.hpp"
#include "AssignColToPix.hpp"
TEST(AssignColToTri, assignColourToTri)
{
    using namespace ced::cpu;
    std::multimap<unsigned int, unsigned int> pixTri ={
        {0, 0},
        {0, 1},
        {0, 2},
        {1, 3},
        {1, 4},
        {1, 5},
    };
    std::vector<float> image = {0.3f, 0.3f, 0.3f,   0.0f, 0.0f, 0.0f,   0.5f, 0.1f, 0.2f,
                                0.3f, 0.3f, 0.3f,   0.0f, 0.0f, 0.0f,   0.1f, 0.1f, 0.8f};

    assignColToPix(image, pixTri, (unsigned int)2);
//   typedef std::multimap<unsigned int, unsigned int>::iterator pixTriItr;
//   unsigned int amountOfTri = 2;
//   std::vector<unsigned int> triPix;
//   for(unsigned int t = 0 ; t < amountOfTri; ++t)
//   {
//       triPix.clear();
//       std::pair<pixTriItr, pixTriItr> result = pixTri.equal_range(t);
//       for(pixTriItr it = result.first; it != result.second; ++it)
//       {
//           triPix.push_back(it->second);
//       }
//       float r = 0;
//       float g = 0;
//       float b = 0;
//       avgColour(image, triPix, r, g, b);
//       assignColourToTri(image, triPix, r, g, b);
//   }
//   ASSERT_EQ(triPix[0], (unsigned int)3);
//   ASSERT_EQ(triPix[1], (unsigned int)4);
//   ASSERT_EQ(triPix[2], (unsigned int)5);

//  float r = 0;
//  float g = 0;
//  float b = 0;
//  avgColour(image, triPix, r, g, b);
//   float amountOfPixelsInTri = triPix.size();
//   ASSERT_FLOAT_EQ(amountOfPixelsInTri, 3.0f);
//   for(auto id : triPix)
//   {
//       r += image[id*3 + 0];
//       g += image[id*3 + 1];
//       b += image[id*3 + 2];
//   }
//   ASSERT_FLOAT_EQ(r, 0.4f);
//   ASSERT_FLOAT_EQ(g, 0.4f);
//   ASSERT_FLOAT_EQ(b, 1.1f);
//
//   r /= amountOfPixelsInTri;
//   g /= amountOfPixelsInTri;
//   b /= amountOfPixelsInTri;
//   ASSERT_FLOAT_EQ(r, 0.4f/3.0f);
//   ASSERT_FLOAT_EQ(g, 0.4f/3.0f);
//   ASSERT_FLOAT_EQ(b, 1.1f/3.0f);
//   assignColourToTri(image, triPix, r, g, b);
    
//  for(auto id : triPix)
//  {
//      image[id*3+ 0] = r;
//      image[id*3+ 1] = g;
//      image[id*3+ 2] = b;
//  }
    EXPECT_FLOAT_EQ(image[0], 0.8f/3.0f);
    EXPECT_FLOAT_EQ(image[1], 0.4f/3.0f);
    EXPECT_FLOAT_EQ(image[2], 0.5f/3.0f);

    EXPECT_FLOAT_EQ(image[3], 0.8f/3.0f);
    EXPECT_FLOAT_EQ(image[4], 0.4f/3.0f);
    EXPECT_FLOAT_EQ(image[5], 0.5f/3.0f);

    EXPECT_FLOAT_EQ(image[6], 0.8f/3.0f);
    EXPECT_FLOAT_EQ(image[7], 0.4f/3.0f);
    EXPECT_FLOAT_EQ(image[8], 0.5f/3.0f);

    EXPECT_FLOAT_EQ(image[9] , 0.4f/3.0f);
    EXPECT_FLOAT_EQ(image[10], 0.4f/3.0f);
    EXPECT_FLOAT_EQ(image[11], 1.1f/3.0f);

    EXPECT_FLOAT_EQ(image[12], 0.4f/3.0f);
    EXPECT_FLOAT_EQ(image[13], 0.4f/3.0f);
    EXPECT_FLOAT_EQ(image[14], 1.1f/3.0f);

    EXPECT_FLOAT_EQ(image[15], 0.4f/3.0f );
    EXPECT_FLOAT_EQ(image[16], 0.4f/3.0f );
    EXPECT_FLOAT_EQ(image[17], 1.1f/3.0f );

} 

