template <typename T>
T distance2P(Point _p1, Point _p2)
{
    T deltaX = _p1.x - _p2.x;
    T deltaY = _p1.y - _p2.y;

    T distance = (deltaX * deltaX) + (deltaY * deltaY);

    return distance;
}
