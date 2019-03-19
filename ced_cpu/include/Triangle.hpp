#ifndef TRIANGLE_H_INCLUDED
#define TRIANGLE_H_INCLUDED

#include <vector>
#include "Point.hpp"
// TODO: smartpointers
namespace ced
{
    namespace cpu
    {
        class Triangle
        {
            public:
                //-------------------------------------
                /// @ build: constructor
                //-------------------------------------
                Triangle(); 
                //-------------------------------------
                /// @ build: constructor
                /// @ param[_in]: vertex1
                /// @ param[_in]: vertex2
                /// @ param[_in]: vertex3
                //-------------------------------------
                Triangle(Point p1, Point p2, Point p3);
                //-------------------------------------
                /// @ build: constructor
                /// @ param[_in]: list of points
                //-------------------------------------
                Triangle(std::vector<Point> _points);
                //-------------------------------------
                /// @ build: constructor
                /// @ param[_in]: list of triangles
                //-------------------------------------
                Triangle(std::vector<Triangle> _triangles);
                //-------------------------------------
                /// @ build: constructor
                /// @ param[_in]: list of points
                /// @ param[_in]: list of triangles
                //-------------------------------------
                Triangle(std::vector<Point> _points, std::vector<Triangle> _triangles);
                //-------------------------------------
                /// @ build: destructor
                //-------------------------------------
                ~Triangle() = default; 
                //-------------------------------------
                /// @ build: add vertex to vertex list
                /// @ param[_in]: point
                //-------------------------------------
                void addVertex(Point _vertex);
                //-------------------------------------
                /// @ build: add triangle to triangle list
                /// @ param[_in]: triangle
                //-------------------------------------
                void addTriangle(Triangle _triangle);
                //-------------------------------------
                /// @ build: get vertex list
                /// @ return: vector of points
                //-------------------------------------
                std::vector<Point> getVertices();
                //-------------------------------------
                /// @ build: get triangle list
                /// @ return: vector of triangles
                //-------------------------------------
                std::vector<Triangle> getNeighbourTriangles();
            private: 
                //-------------------------------------
                /// @build vertices 
                //-------------------------------------
                std::vector<Point> m_vertices;                  
                //-------------------------------------
                /// @build edge-sharing neighbor triangles 
                //-------------------------------------
                std::vector<Triangle> m_neighbourTriangles;                  
            
        };
    }
}

#endif // TRIANGLE_H_INCLUDED
