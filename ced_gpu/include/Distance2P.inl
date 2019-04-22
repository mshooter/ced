template <typename T>
__device__ T distance2P(T x1, T y1, T x2, T y2)
{
    T deltaX = x1 - x2; 
    T deltaY = y1 - y2;

    T distance = (deltaX * deltaX) + (deltaY * deltaY);
    return distance;
}
