#ifndef POINT_CUH_INCLUDED
#define POINT_CUH_INCLUDED

#include "params.hpp"

namespace ced
{
    namespace gpu
    {
        struct Point
        {
            //  ----------------------------------------------------------
            /// @build: Point constructor
            /// @param[_in] : coordinate x 
            /// @param[_in] : coordinate y 
            //  ----------------------------------------------------------
            template <typename T>
            __host__ __device__ Point(const T _x, const T _y);
            //  ----------------------------------------------------------
            /// @build: default constructor
            //  ----------------------------------------------------------
            __host__ __device__ Point() {;}
            //  ----------------------------------------------------------
            /// @build: assign operator 
            //  ----------------------------------------------------------
            Point& operator=(const Point&) = default;
            //  ----------------------------------------------------------
            /// @build: compare operator 
            //  ----------------------------------------------------------
            __host__ __device__ bool operator==(const Point& rhs) const;
            //  ----------------------------------------------------------
            /// @build: add operator 
            //  ----------------------------------------------------------
            __host__ __device__ Point operator+(const Point& rhs) const;
            //  ----------------------------------------------------------
            /// @build: substract operator 
            //  ----------------------------------------------------------
            __host__ __device__ Point operator-(const Point& rhs) const;
            //  ----------------------------------------------------------
            /// @build: multiply operator 
            //  ----------------------------------------------------------
            __host__ __device__ Point operator*(const Point& rhs) const;
            //  ----------------------------------------------------------
            /// @build: divide operator 
            //  ----------------------------------------------------------
            __host__ __device__ Point operator/(const float scalar) const;
            //  ----------------------------------------------------------
            /// @build: x coordinate 
            //  ----------------------------------------------------------
            float x;
            //  ----------------------------------------------------------
            /// @build: y coordinate 
            //  ----------------------------------------------------------
            float y;
        };
        //  ----------------------------------------------------------
        /// @build: checks if the points are equal
        //  ----------------------------------------------------------
        __host__ __device__ bool equalPts(Point p1, Point p2);
        //  ----------------------------------------------------------
        /// @build: dotProduct of two points
        //  ----------------------------------------------------------
        template <typename T>
        __host__ __device__ T dot(Point p1, Point p2);
        //  ----------------------------------------------------------
        //  ----------------------------------------------------------
        /// @build: template implementation
        //  ----------------------------------------------------------
        #include "Point.inl"
    }
}

#endif // POINT_CUH_INCLUDED
