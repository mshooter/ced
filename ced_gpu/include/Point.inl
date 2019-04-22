template <typename T>
__host__ __device__ Point::Point(const T _x, const T _y)
{
    x = static_cast<float>(std::move(_x));
    y = static_cast<float>(std::move(_y));
}

template <typename T>
__host__ __device__ T dot(Point p1, Point p2)
{
    return static_cast<T>(p1.x * p2.x + p1.y * p2.y);
}

