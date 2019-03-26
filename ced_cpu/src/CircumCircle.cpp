#include "CircumCircle.hpp"
namespace ced
{
    namespace cpu
    {
        bool isPointInCircle(Point A, Point B, Point C, Point D)
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
        //  ----------------------------------------------------------------------------------------
        float circumRadius(Point A, Point B, Point C)
        {
            // edges need to do the minus operator
            float ax = static_cast<float>(B.x - A.x);
            float ay = static_cast<float>(B.y - A.y);
            float bx = static_cast<float>(C.x - B.x);
            float by = static_cast<float>(C.y - B.y);
            float cx = static_cast<float>(A.x - C.x);
            float cy = static_cast<float>(A.y - C.y);
            float axx = ax * ax;
            float ayy = ay * ay;
            float bxx = bx * bx;
            float byy = by * by;
            float cxx = cx * cx;
            float cyy = cy * cy;
            float a = std::sqrt(axx + ayy);
            float b = std::sqrt(bxx + byy);
            float c = std::sqrt(cxx + cyy);
            return (a * b * c) / std::sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c));
        }
        //  ----------------------------------------------------------------------------------------
        Point circumCenter(Point A, Point B, Point C)
        {
            // calculate midpoints
            Point midAB = {(A+B)/2}; 
            Point midAC = {(A+C)/2}; 
            float slopeAB = (B.y - A.y) / (B.x - A.x); 
            slopeAB = -(1/slopeAB);
            float slopeAC = (C.y - A.y) / (C.x - A.x); 
            slopeAC = -(1/slopeAC);
            // solving mx + b = y 
            // solve for b 
            float bAB = midAB.y - slopeAB * midAB.x;
            float bAC = midAC.y - slopeAC * midAC.x;
            float x = (bAB - bAC) / (slopeAC - slopeAB);
            float y = (slopeAB * x) + bAB; 
            return Point{x, y};
        }
    }
}


