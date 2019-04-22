#include "Point.cuh"

namespace ced
{
    namespace gpu
    {
        __host__ __device__ bool Point::operator==(const Point& rhs) const
        {
            return ((x == rhs.x) && (y == rhs.y));
        }
        //  ----------------------------------------------------------
        __host__ __device__ Point Point::operator+(const Point& rhs) const
        {
            Point p;
            p.x = x + rhs.x;
            p.y = y + rhs.y;
            return p;
        }
        //  ----------------------------------------------------------
        __host__ __device__ Point Point::operator-(const Point& rhs) const
        {
            Point p;
            p.x = x - rhs.x;
            p.y = y - rhs.y;
            return p;
        }
        //  ----------------------------------------------------------
        __host__ __device__ Point Point::operator*(const Point& rhs) const
        {
            Point p;
            p.x = x * rhs.x;
            p.y = y * rhs.y;
            return p;
        }
        //  ----------------------------------------------------------
        __host__ __device__ Point Point::operator/(const float scalar) const
        {
            Point p;
            p.x = x / scalar;
            p.y = y / scalar;
            return p;
        }
        //  ----------------------------------------------------------
        __host__ __device__ bool equalPts(Point p1, Point p2)
        {
            return (std::fabs(p1.x - p2.x) <= EPSILON) && (std::fabs(p1.y - p2.y) <= EPSILON);
        } 
    }
}

