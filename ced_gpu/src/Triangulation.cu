#include "Triangulation.cuh"

#include <thrust/transform.h>

#include "ThrustFunctors.cuh"
#include "PseudoAngle.cuh"

namespace ced
{
    namespace gpu
    {
        __host__ void createFirstTri(thrust::device_vector<Point> pts, unsigned int& i0, unsigned int& i1, unsigned int& i2, Point centroid)
        {
            thrust::device_vector<float> distances; 
            thrust::transform(pts.begin(), pts.end(), distances.begin(), DistanceCP(centroid));
            // min_element gives an iterator so to get a position - the beginning of the vector
            i0 = thrust::min_element(distances.begin(), distances.end()) - distances.begin();
            // need to find what the size of the new distances is ? 
            // need to create a functor 
            thrust::transform(pts.begin(), pts.end(), distances.begin(), distances.begin(), DistanceCP(centroid));
            i1 = thrust::min_element(distances.begin(), distances.end()) - distances.begin();
            // need to create two functors 
        }

        __device__ unsigned int hashKey(Point p, point cc, unsigned int hashSize)
        {
            Point pcc = p - cc; 
            float fangle = pseudo_angle(pcc) * static_cast<float>(hashSize);
            int iangle = static_cast<int>(fangle);
            return (iangle > fangle ? --iangle : iangle)&(hashSize-1);
        }
    }
}
