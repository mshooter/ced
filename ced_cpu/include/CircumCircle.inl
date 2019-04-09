template <typename T>
bool isPointInCircle(Point A, Point B, Point C, Point D)
{
    T m_00 = A.x - D.x;
    T m_01 = B.x - D.x;
    T m_02 = C.x - D.x;
    
    T m_10 = A.y - D.y;
    T m_11 = B.y - D.y;
    T m_12 = C.y - D.y;
    
    T m_20 = m_00 * m_00 + m_10 * m_10;
    T m_21 = m_01 * m_01 + m_11 * m_11;
    T m_22 = m_02 * m_02 + m_12 * m_12;
    
    T determinant = (m_00 * m_11 * m_22) + (m_10 * m_21 * m_02) + (m_20 * m_01 * m_12) -
                        (m_20 * m_11 * m_02) - (m_10 * m_01 * m_22) - (m_00 * m_21 * m_12); 
    
    // negative is inside the circle 
    return determinant < 0;
}

template <typename T>
T circumRadius(Point A, Point B, Point C)
{
    Point delta_ab = B-A;
    Point delta_ac = C-A;
    
    const T dist_ab = delta_ab.x * delta_ab.x + delta_ab.y * delta_ab.y;
    const T dist_ac = delta_ac.x * delta_ac.x + delta_ac.y * delta_ac.y;
    const T N = delta_ab.x * delta_ac.y - delta_ab.y * delta_ac.x;

    const T x = (delta_ac.y * dist_ab - delta_ab.y * dist_ac) * 0.5 / N; 
    const T y = (delta_ac.x * dist_ac - delta_ac.x * dist_ab) * 0.5 / N; 

    // this is weird ? must check
    if(dist_ab != 0 && dist_ac != 0 && N != 0)
    {
        return x * x + y * y;
    }
    else
    {
        return std::numeric_limits<float>::max();
    }
}

 
