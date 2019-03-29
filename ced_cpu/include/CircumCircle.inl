template <typename T>
bool isPointInCircle(T A, T B, T C, T D)
{
    int m_00 = A.x - D.x;
    int m_01 = B.x - D.x;
    int m_02 = C.x - D.x;
    
    int m_10 = A.y - D.y;
    int m_11 = B.y - D.y;
    int m_12 = C.y - D.y;
    
    int m_20 = m_00 * m_00 + m_10 * m_10;
    int m_21 = m_01 * m_01 + m_11 * m_11;
    int m_22 = m_02 * m_02 + m_12 * m_12;
    
    int determinant = (m_00 * m_11 * m_22) + (m_10 * m_21 * m_02) + (m_20 * m_01 * m_12) -
                        (m_20 * m_11 * m_02) - (m_10 * m_01 * m_22) - (m_00 * m_21 * m_12); 
    
    // negative is inside the circle 
    return determinant < 0;
}

template <typename T, typename U>
U circumRadius(T A, T B, T C)
{
    // edges need to do the minus operator
    U ax = (B.x - A.x);
    U ay = (B.y - A.y);
    U bx = (C.x - B.x);
    U by = (C.y - B.y);
    U cx = (A.x - C.x);
    U cy = (A.y - C.y);
    U axx = ax * ax;
    U ayy = ay * ay;
    U bxx = bx * bx;
    U byy = by * by;
    U cxx = cx * cx;
    U cyy = cy * cy;
    U a = std::sqrt(axx + ayy);
    U b = std::sqrt(bxx + byy);
    U c = std::sqrt(cxx + cyy);
    return (a * b * c) / std::sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c));
}

template <typename T>
T circumCenter(T P, T Q, T R)
{
    float a, b, c, e, f, g; 
    a = Q.y - P.y; 
    b = P.x - Q.x; 
    c = a*(P.x) + b*(P.y); 
    e = R.y - Q.y; 
    f = Q.x - R.x; 
    g = e*(Q.x) + f*(Q.y); 

    T mp = ((P+Q)/2.0f);
    c = -b * (mp.x) + a * (mp.y);
    b = -b;
    std::swap(a,b);

    T mpp = ((Q+R) / 2.0f);
    g = -f * (mpp.x) + e * (mpp.y);
    f = -f;
    std::swap(e,f);

    float D = a * f - e * b;

    if(D!=0)
    {
        float x = (f*c - b*g) / D;
        float y = (a*g - e*c) / D;
        return T{x, y};
    }
    else
    {
        float x = (f*c - b*g) / (1);
        float y = (a*g - e*c) / (1);
        return T{x, y};
    }

}
 
