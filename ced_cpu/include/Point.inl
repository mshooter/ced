template <typename T>
Point::Point(const T _x, const T _y)
{
    x = (int)std::move(_x);
    y = (int)std::move(_y);
}
