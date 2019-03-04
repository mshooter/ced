#include "DrawLine.hpp"

namespace ced
{
    namespace cpu
    {
        void drawLine(Point _p1, Point _p2, std::vector<float>& _image, int _width)
        {
            int x_0 = _p1.getX();   
            int y_0 = _p1.getY();   

            int x_1 = _p2.getX();   
            int y_1 = _p2.getY();   
            
            int x = _p1.getX();
            int y = _p1.getY();

            int deltaX = x_1 - x_0;
            int deltaY = y_1 - y_0;
            
            int D = 2 * deltaY - deltaX; 

            while(x<x_1)
            {
                if(D>=0)
                {
                    _image[(x + y * _width) * 3 + 0] = 1;
                    _image[(x + y * _width) * 3 + 1] = 0;
                    _image[(x + y * _width) * 3 + 2] = 0;
                    y++;
                    D += (2*deltaY - 2*deltaX);
                }
                else
                {
                    _image[(x + y * _width) * 3 + 0] = 1;
                    _image[(x + y * _width) * 3 + 1] = 0;
                    _image[(x + y * _width) * 3 + 2] = 0;
                    D += (2*deltaY);
                }
                x++;
            }
        }   
    }
}
