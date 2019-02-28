#include "Point.hpp"
#include <utility>

namespace ced
{
    namespace cpu
    {
        Point::Point(const int x, const int y)
        {
            m_x = std::move(x);
            m_y = std::move(y);
        }
        
        int Point::getX()
        {
            return m_x;
        }

        int Point::getY()
        {
            return m_y;
        }

        void Point::setX(int x)
        {
            m_x = std::move(x);
        }

        void Point::setY(int y)
        {
            m_y = std::move(y);
        }
    }
}
