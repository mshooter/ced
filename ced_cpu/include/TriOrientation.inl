template <typename T>
bool isCCW(T _p1, T _p2, T _p3)
{
    float yEdgeRho = static_cast<float>(_p2.y - _p1.y); 
    float xEdgeRho = static_cast<float>(_p2.x - _p1.x); 

    float rho = yEdgeRho / xEdgeRho;

    float yEdgeTau = static_cast<float>(_p3.y - _p2.y); 
    float xEdgeTau = static_cast<float>(_p3.x - _p2.x); 

    float tau = yEdgeTau / xEdgeTau;
    
    return ((rho-tau) < 0);    
}
