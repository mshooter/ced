template <typename T, typename U>
bool isCCW(T _p1, T _p2, T _p3)
{
    U yEdgeRho = (_p2.y - _p1.y); 
    U xEdgeRho = (_p2.x - _p1.x); 

    U rho = yEdgeRho / xEdgeRho;

    U yEdgeTau = (_p3.y - _p2.y); 
    U xEdgeTau = (_p3.x - _p2.x); 

    U tau = (yEdgeTau / xEdgeTau);
    
    return (rho<tau);    
}
