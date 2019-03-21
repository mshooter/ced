template <typename T>
int partition(std::vector<T>& pt, int p, int q)
{
    // element to move pivot point
    T piv = pt[p];
    int i = p; 
    for(int j=p+1; j <= q; j++)
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
    std::swap(pt[i], pt[p]);
    return i;
}

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
