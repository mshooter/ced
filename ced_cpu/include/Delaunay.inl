template <typename T>
bool isReverseEdgeInHull(T _edge, std::vector<T> _hull)
{
    // swap the elements in the edge
    std::swap(_edge.startPoint, _edge.endPoint);  
    // i dont think you need to iterate over because you dont need to search, but this is an option
    for(auto edge : _hull)
    {
        if(_edge == edge)
        {
            return true;
        }
    }
    return false;
}
//--------------------------------------------------------------------------
template <typename T>
void insertBeforeElement(T _element, typename std::vector<T>::iterator _bei, std::vector<T>& _hull)
{
    // bei is base index
    if(_bei != _hull.begin())
    {
        _hull.insert(_bei - 1, _element);
    }
    else
    {
        _hull.insert(_bei, _element);
    }
}
//--------------------------------------------------------------------------
template <typename T>
void insertAfterElement(T _element, typename std::vector<T>::iterator _bei, std::vector<T>& _hull)
{
    if(_bei != _hull.end())
    {
        _hull.insert(_bei + 1, _element);
    }
    else
    {
        _hull.insert(_bei, _element);
    }
}

