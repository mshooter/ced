#include "gtest/gtest.h"
#include "Triangulation.cuh"

TEST(Triangulation, hashkey)
{

    ced::cpu::Point p = {0.0f, 2.0f};
    ced::cpu::Point cc = {3.0f, 1.0f};

    unsigned int size = 3;
    unsigned int key = ced::gpu::hashKey(p, cc, size);
}
