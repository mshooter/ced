#include "Triangle.hpp"

namespace ced
{
    namespace cpu
    {
        Triangle::Triangle()
        {
            m_vertices.reserve(3);
            m_neighbourTriangles.reserve(3);
        }
        //-------------------------------------
        Triangle::Triangle(Point  p1, Point  p2, Point  p3)
        {
            m_vertices.reserve(3);
            m_vertices.push_back(p1);
            m_vertices.push_back(p2);
            m_vertices.push_back(p3);
            m_neighbourTriangles.reserve(3);
        }
        //-------------------------------------
        Triangle::Triangle(std::vector<Point > _points)
        {
            m_vertices.reserve(3);
            m_vertices = std::move(_points);
            m_neighbourTriangles.reserve(3);
        }
        //-------------------------------------
        Triangle::Triangle(std::vector<Triangle > _triangles)
        {
            m_vertices.reserve(3);
            m_neighbourTriangles.reserve(3);
            m_neighbourTriangles = std::move(_triangles);
        }
        //-------------------------------------
        Triangle::Triangle(std::vector<Point > _points, std::vector<Triangle > _triangles)
        {
            m_vertices.reserve(3);
            m_vertices = std::move(_points);
            m_neighbourTriangles.reserve(3);
            m_neighbourTriangles = std::move(_triangles);
        }
        //-------------------------------------
        void Triangle::addVertex(Point  _vertex)
        {
            m_vertices.push_back(_vertex);
        }
        //-------------------------------------
        void Triangle::addTriangle(Triangle  _triangle)
        {
            m_neighbourTriangles.push_back(_triangle);
        }
        //-------------------------------------
        std::vector<Point > Triangle::getVertices()
        {
            return m_vertices;
        }
        //-------------------------------------
        std::vector<Triangle > Triangle::getNeighbourTriangles()
        {
            return m_neighbourTriangles;
        }
    }
}
