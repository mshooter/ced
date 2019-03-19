template <typename T>
bool isReverseEdgeInHull(T _edge, std::vector<T> _hull)
{
    // swap the elements in the edge
    std::swap(_edge.startPoint, _edge.endPoint);  
    for(auto edge : _hull)
    {
        if(_edge == edge)
        {
            return true;
        }
    }
    return false;
}
