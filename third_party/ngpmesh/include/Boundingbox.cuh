#pragma once
#ifndef BOUNDINGBOX_CUH
#define BOUNDINGBOX_CUH

#include<Eigen/Eigen>
#include<vector>
#include<Triangle.cuh>


struct BoundingBox{
    Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());

    __host__ __device__ BoundingBox(){}
    __host__ __device__ BoundingBox(float scale):min(Eigen::Vector3f::Constant(-scale)), max(Eigen::Vector3f::Constant(scale)){}
    __host__ __device__ BoundingBox(const Eigen::Vector3f &min_, const Eigen::Vector3f &max_):min(min_), max(max_){}
    __host__ __device__ BoundingBox(std::vector<Triangle>::iterator begin, std::vector<Triangle>::iterator end){
        min = max = begin->a;
        for(auto it=begin; it!=end; ++it){
            update(*it);
        }
    }
    __host__ __device__ BoundingBox(const std::vector<Triangle> &triangles){ for(auto tri : triangles)update(tri);}

    __host__ __device__ void clear(){
        min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
        max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
    }

    __host__ __device__ void update(const Triangle &t){
        update(t.a);
        update(t.b);
        update(t.c);
    }
    __host__ __device__ void update(const Eigen::Vector3f &p){
        min = min.cwiseMin(p);
        max = max.cwiseMax(p);
    }

    __host__ __device__ void enlarge(const float offset){
        min -= Eigen::Vector3f::Constant(offset);
        max += Eigen::Vector3f::Constant(offset);
    }
        
    __host__ __device__ Eigen::Vector3f diag() const{ return max - min;}
    __host__ __device__ Eigen::Vector3f center() const{ return (min + max) * 0.5f;}

    __host__ __device__ bool available() const{ return (max.array() < min.array()).any();}

    __host__ __device__ bool contains(Eigen::Vector3f &p) const{
        return  p.x() >= min.x() && p.x() <= max.x() &&
                p.y() >= min.y() && p.y() <= max.y() &&
                p.z() >= min.z() && p.z() <= max.z();
    }

    __host__ __device__ Eigen::Vector3f relative(const Eigen::Vector3f &p) const{ return (p - min).cwiseQuotient(diag());}

    __host__ __device__ BoundingBox intersection(const BoundingBox &bb) const{ return BoundingBox(min.cwiseMax(bb.min), max.cwiseMin(bb.max));}

        /**
         * @param o ray origin
         * @param d ray direction
         * @return {t_in, t_out}
         */
    __host__ __device__ Eigen::Vector2f ray_intersect(const Eigen::Vector3f &o, const Eigen::Vector3f &d) const{
        float t_in = (min.x() - o.x())/ d.x();
        float t_out = (max.x() - o.x())/ d.x();

        if(t_in > t_out)swap_value(t_in, t_out);
            
        float y_in = (min.y() - o.y())/ d.y();
        float y_out = (max.y() - o.y())/ d.y();

        if(y_in > y_out)swap_value(y_in, y_out);

        if(t_in > y_out || t_out < y_in){
            return {std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::min()};
        }
            
        if(y_in > t_in)t_in = y_in;
        if(y_out < t_out)t_out = y_out;

        float z_in = (min.z() - o.z())/ d.z();
        float z_out = (max.z() - o.z())/ d.z();

        if(z_in > z_out)swap_value(z_in, z_out);

        if(t_in > z_out || t_out < z_in){
            return {std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()};
        }

        if(z_in > t_in)t_in = z_in;
        if(z_out < t_out)t_out = z_out;

        return {t_in, t_out};
    }

    __host__ __device__ float distance_2(const Eigen::Vector3f &p) const{ return (min - p).cwiseMax(p - max).cwiseMax(0.0f).squaredNorm(); }

    __host__ __device__ float distance(const Eigen::Vector3f &p) const{ return std::sqrt(distance_2(p));}

    __host__ __device__ float signed_distance(const Eigen::Vector3f &p) const{
        Eigen::Vector3f q = (p - min).cwiseAbs() - diag();
        return q.cwiseMax(0.0f).norm() + std::min(std::max({q.x(), q.y(), q.z()}), 0.0f);
    }

};

#endif