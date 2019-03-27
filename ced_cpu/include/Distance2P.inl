template <typename T, typename U>
T distance2P(U _p1, U _p2)
{
    T deltaX = _p1.x - _p2.x;
    T deltaY = _p1.y - _p2.y;

    T distance = std::sqrt((deltaX * deltaX) + (deltaY * deltaY));

    return distance;
}
