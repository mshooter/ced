#include "Triangle.hpp"

namespace ced
{
    namespace cpu
    {
        Triangle::Triangle()
        {
            m_vertices.reserve(3);
            m_triangles.reserve(3);
        }
        //-------------------------------------
        Triangle::Triangle(Point  p1, Point  p2, Point  p3)
        {
            m_vertices.reserve(3);
            m_vertices.push_back(p1);
            m_vertices.push_back(p2);
            m_vertices.push_back(p3);
            m_triangles.reserve(3);
            m_edges.push_back(Edge(p1, p2));
            m_edges.push_back(Edge(p2, p3));
            m_edges.push_back(Edge(p3, p1));
        }
        //-------------------------------------
        Triangle::Triangle(std::vector<Point > _points)
        {
            m_vertices.reserve(3);
            m_vertices = std::move(_points);
            m_triangles.reserve(3);
        }
        //-------------------------------------
        Triangle::Triangle(std::vector<Triangle > _triangles)
        {
            m_vertices.reserve(3);
            m_triangles.reserve(3);
            m_triangles = std::move(_triangles);
        }
        //-------------------------------------
        Triangle::Triangle(std::vector<Point > _points, std::vector<Triangle > _triangles)
        {
            m_vertices.reserve(3);
            m_vertices = std::move(_points);
            m_triangles.reserve(3);
            m_triangles = std::move(_triangles);
        }
        //-------------------------------------
        void Triangle::addVertex(Point  _vertex)
        {
            m_vertices.push_back(_vertex);
        }
        //-------------------------------------
        void Triangle::addTriangle(Triangle  _triangle)
        {
            m_triangles.push_back(_triangle);
        }
        //-------------------------------------
        void Triangle::addEdge(Edge _edge)
        {
            m_edges.push_back(_edge);
        }
        //-------------------------------------
        std::vector<Point > Triangle::getVertices()
        {
            return m_vertices;
        }
        //-------------------------------------
        std::vector<Triangle > Triangle::getNeighbourTriangles()
        {
            return m_triangles;
        }
        //-------------------------------------
        std::vector<Edge> Triangle::getEdges()
        {
            return m_edges;
        }
        //-------------------------------------
        bool Triangle::operator==(const Triangle& rhs) const
        {
            return ( m_vertices == rhs.m_vertices);
        }
    }
}
