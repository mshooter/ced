template <typename T>
bool pointIsLeft(T x_0, T x_1, T x_2, T y_0, T y_1, T y_2)
{
    T v = ((x_2 - x_0) * (y_1 - y_0)) - ((x_1 - x_0) * (y_2 - y_0)); 
    if(v <= 0)  {return true;}
    else        {return false;}
}
//--------------------------------------------------------------------------
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
typename std::vector<T>::iterator insertAfterElement(T _element, typename std::vector<T>::iterator _bei, std::vector<T>& _hull)
{
    if(_bei != _hull.end())
    {
        typename std::vector<T>::iterator it = _hull.insert(_bei + 1, _element);
        return it;
    }
    else
    {
        typename std::vector<T>::iterator it = _hull.insert(_bei, _element);
        return it;
    }
}
//  --------------------------------------------------------------------------


