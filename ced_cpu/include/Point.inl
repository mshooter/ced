template <typename T>
Point::Point(const T _x, const T _y)
{
    x = static_cast<float>(std::move(_x));
    y = static_cast<float>(std::move(_y));
}
