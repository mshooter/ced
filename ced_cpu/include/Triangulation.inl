template <typename T>
void addTriangle(T _id0, T _id1, T _id2, T _ahe0, T _ahe1, T _ahe2, std::vector<T>& _triangles)
{
    // do i need this ? 
    // T t = _triangles.size();
    _triangles.insert(_triangles.end(), {_id0, _id1, _id2});
    // link?
    // return t;
}
