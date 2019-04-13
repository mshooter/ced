template <typename T>
T isCCW(Point _p1, Point _p2, Point _p3)
{
    T y_21 = (_p2.y - _p1.y); 
    T x_21 = (_p2.x - _p1.x); 

    T y_32 = (_p3.y - _p2.y); 
    T x_32 = (_p3.x - _p2.x); 

    T ccw = (y_21 * x_32 - x_21 * y_32);
    // ccw < 0 
    return ccw ;    
}

