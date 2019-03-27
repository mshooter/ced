template <typename T, typename U>
bool isCCW(T _p1, T _p2, T _p3)
{
    U y_21 = (_p2.y - _p1.y); 
    U x_21 = (_p2.x - _p1.x); 

    U y_32 = (_p3.y - _p2.y); 
    U x_32 = (_p3.x - _p2.x); 

    U ccw = (y_21 * x_32 - x_21 * y_32);
    return ccw < 0;    
}
