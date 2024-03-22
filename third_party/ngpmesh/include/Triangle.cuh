#pragma once

#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH
#include<Common.cuh>
#include<Eigen/Eigen>

struct Triangle{
    Eigen::Vector3f a,b,c;

    Triangle(){}

    Triangle(Eigen::Vector3f a_, Eigen::Vector3f b_, Eigen::Vector3f c_){
        a = a_; b = b_; c = c_;
    }

    __host__ __device__ Eigen::Vector3f normal() const{
        return (b - a).cross(c - a).normalized();
    }

    // 1/2 * AB * AC * sin(a)
    __host__ __device__ float area() const{
        return 0.5f * Eigen::Vector3f((b - a).cross(c - a)).norm();
    }

    __host__ __device__ Eigen::Vector3f centroid() const {
        return (a + b + c)/3.0f;
    }

    __host__ __device__ float centroid(int axis) const{
        return (a[axis] + b[axis] + c[axis])/3;
    }
    // Reference Code from instant-ngp
    __host__ __device__ Eigen::Vector3f sample_uniform_position(Eigen::Vector2f sample){
        // sqrt(x)
        // alpha = 1 - sqrt(x)
        // beta = sqrt(x) * (1 - sqrt(y))
        // gamma = sqrt(x) * y
        float sqrt_x = std::sqrt(sample.x());
        float factor0 = 1.0f - sqrt_x;
        float factor1 = sqrt_x * (1.0f - sample.y());
        float factor2 = sqrt_x * sample.y();

        return factor0 * a + factor1 * b + factor2 * c;
    }

    //reference code from https://iquilezles.org/articles/triangledistance/
    __host__ __device__ float distance_from_point_2(Eigen::Vector3f &p) const{
        // prepare data    
        Eigen::Vector3f v21 = b - a; Eigen::Vector3f p1 = p - a;
        Eigen::Vector3f v32 = c - b; Eigen::Vector3f p2 = p - b;
        Eigen::Vector3f v13 = a - c; Eigen::Vector3f p3 = p - c;
        Eigen::Vector3f nor = v21.cross(v13);

        return  // inside/outside test    
                    (sign(v21.cross(nor).dot(p1)) + 
                    sign(v32.cross(nor).dot(p2)) + 
                    sign(v13.cross(nor).dot(p3))<2.0) 
                    ?
                    // 3 edges
                    std::min({
                        (v21 * clamp(v21.dot(p1) / v21.squaredNorm(), 0.0f, 1.0f)-p1).squaredNorm(),
                        (v32 * clamp(v32.dot(p2) / v32.squaredNorm(), 0.0f, 1.0f)-p2).squaredNorm(),
                        (v13 * clamp(v13.dot(p3) / v13.squaredNorm(), 0.0f, 1.0f)-p3).squaredNorm(),
                    })    
                    :
                    // 1 face   
                    nor.dot(p1) * nor.dot(p1) / nor.squaredNorm();
    }

    __host__ __device__ float distance_from_point(Eigen::Vector3f &p) const{
        return std::sqrt(distance_from_point_2(p));
    }

    //reference code from https://iquilezles.org/articles/intersectors/
    /**
     * @return distance from ro to hit point
     */ 
    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd, Eigen::Vector3f &n) const{
        Eigen::Vector3f v10 = b - a, 
                        v20 = c - a, 
                        vo0 = ro - a;
        n = v10.cross(v20);
        Eigen::Vector3f q = vo0.cross(rd);
        float d = 1.0/rd.dot(n);
        float u = d*(-q).dot(v20);
        float v = d*q.dot(v10);
        float t = d*(-n).dot(vo0);
        if( u<0.0 || v<0.0 || (u+v)>1.0 || t<0.0f ) t = std::numeric_limits<float>::max();
        return t;
    }

    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd) const{
        Eigen::Vector3f n;
        return ray_intersect(ro, rd, n);
    }

    __host__ __device__ bool point_in_triangle(const Eigen::Vector3f &p) const{
        Eigen::Vector3f la = a - p,
                        lb = b - p,
                        lc = c - p;
        Eigen::Vector3f u = lb.cross(lc);
        Eigen::Vector3f v = lc.cross(la);
        Eigen::Vector3f w = la.cross(lb);

        return u.dot(v) >= 0.0f && u.dot(w) >= 0.0f;
    }

    __host__ __device__ Eigen::Vector3f closest_point_to_line(const Eigen::Vector3f &a, const Eigen::Vector3f &b, Eigen::Vector3f &p) const{
        float t = (p - a).dot(b - a) / (b - a).dot(b - a);
        t = clamp(t, 0.0f, 1.0f);
        return a + t * (b - a);
    }

    // It seems to obtain an approximation
    // reference code of instant-ngp
    __host__ __device__ Eigen::Vector3f closest_point_to(Eigen::Vector3f p) const {
        Eigen::Vector3f n = normal();
        p -= n.dot(p - a) * n;

        if(point_in_triangle(p))return p;

        Eigen::Vector3f c1 = closest_point_to_line(a, b, p);
        Eigen::Vector3f c2 = closest_point_to_line(b, c, p);
        Eigen::Vector3f c3 = closest_point_to_line(c, a, p);

        float mag1 = (p - c1).squaredNorm();
        float mag2 = (p - c2).squaredNorm();
        float mag3 = (p - c3).squaredNorm();
        float min = std::min({mag1, mag2, mag3});
        if(min == mag1)return c1;
        else if(min == mag2)return c2;
        else return c3;
    }
};

#endif