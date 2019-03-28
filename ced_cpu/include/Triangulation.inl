template <typename T, typename U>
T pseudo_angle(const U _point)
{
    T N = ((std::abs(_point.x) + std::abs(_point.y)) == 0) ? 0.1 : (std::abs(_point.x) + std::abs(_point.y)); 
    const T p = _point.x / N;  
    T result = 0;
    if(_point.y > 0)
        result = static_cast<T>(3.0) - p;  
    else
    {
        result = static_cast<T>(1.0) + p;
    }
    return (result/static_cast<T>(4.0)); 
}
//  --------------------------------------------------------------------------------------------
template <typename T, typename U>
T hash_key(const U _point, const U _cc, const T _hashSize) 
{
    T dx = static_cast<T>(_point.x - _cc.x);
    T dy = static_cast<T>(_point.y - _cc.y);
    // floor
    float anglef = pseudo_angle<float, U>(U(dx, dy)) * static_cast<float>(_hashSize);
    int anglei = static_cast<int>(anglef); 
    if(anglei > anglef) --anglei;  
    return anglei;
}
//  --------------------------------------------------------------------------------------------
template <typename T>
T addTriangle(T _id0, T _id1, T _id2, T _ahe0, T _ahe1, T _ahe2, std::vector<T>& _triangles, std::vector<T>& halfedges)
{
    // do i need this ? 
    T t = _triangles.size();
    _triangles.insert(_triangles.end(), {_id0, _id1, _id2});
    
    link(t + 0, _ahe0, halfedges);
    link(t + 1, _ahe1, halfedges);
    link(t + 2, _ahe2, halfedges);
    
    t+=3;

    return t;
}
