template <typename T>
int partition(std::vector<T>& pt, int lo, int hi)
{
    // element to move pivot point
    T piv = pt[lo];
    int i = lo; 
    for(int j=lo+1; j <= hi; j++)
    {
        if(pt[j].x < piv.x)
        {
            i=i+1; 
            std::swap(pt[i], pt[j]);
        }

        if(pt[j].x == piv.x)
        {
            if(pt[j].y <= piv.y)
            {
                i++;
                std::swap(pt[i], pt[j]);
            }
        }
    }
    std::swap(pt[i], pt[lo]);
    return i;
}
//  ----------------------------------------------------------
template <typename T>
void quickSort(std::vector<T>& _pts, int lo, int hi)
{
    int r;
    if(lo < hi)
    {
        r = partition(_pts, lo, hi);
        quickSort(_pts, lo, r-1);
        quickSort(_pts, r+1, hi);
    }
}
//  ----------------------------------------------------------
template <typename T>
int partitionDist(std::vector<unsigned int>& _ids, std::vector<Point> _points, Point cc, int lo, int hi)
{
    // element to pivot 
    T dist1 = distance2P<T>(_points[lo], cc);
    // index
    int i=lo;
    for(int j=lo+1; j <= hi; ++j)
    {
        T dist2 = distance2P<T>(_points[j], cc); 
        if(dist2 < dist1)
        {
            i=i+1;
            std::swap(_ids[i], _ids[j]);
        }
    }
    std::swap(_ids[i], _ids[lo]);
    return i;
}
//  ----------------------------------------------------------
template <typename T>
void quickSortDist(std::vector<unsigned int>& _ids, std::vector<Point> _points, Point cc, int lo, int hi)
{
    int r; 
    if(lo < hi)
    {
        r = partitionDist<T>(_ids, _points, cc, lo, hi);
        quickSortDist<T>(_ids, _points, cc, lo, r-1);
        quickSortDist<T>(_ids, _points, cc, r+1, hi);
    }
}


