#pragma once
#ifndef RAYTRACERUTILS_CUH
#define RAYTRACERUTILS_CUH
#include<TriangleBvh.cuh>
#include<Loader.cuh>

class NGPMesh{
    private:
        BoundingBox aabb;
        float aabb_offset = 1e-3f;
        std::vector<Triangle> triangles_cpu;
        GPUVector<Triangle> triangles_gpu;
        std::unique_ptr<TriangleBVH> bvh;
        cudaStream_t stream;
    
    public:
        NGPMesh(): aabb(1.0f){
            CUDA_CHECK_THROW(cudaStreamCreate(&stream));
        }
        NGPMesh(std::vector<Eigen::Matrix3f> &triangles_matrix):aabb(1.0f){
            CUDA_CHECK_THROW(cudaStreamCreate(&stream));
            load_mesh(triangles_matrix);
        }
        NGPMesh(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector<int, 3>> &faces){
            CUDA_CHECK_THROW(cudaStreamCreate(&stream));
            load_mesh(vertices, faces);
        }
        ~NGPMesh(){}

        void load_mesh(std::vector<Eigen::Matrix3f> &triangles_matrix);
        void load_mesh(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector<int, 3>> &faces);

        std::vector<float>           unsigned_distance(std::vector<Eigen::Vector3f> &positions);
        std::vector<float>           signed_distance(std::vector<Eigen::Vector3f> &positions, SDFCalcMode mode = SDFCalcMode::RAYSTAB);
        std::vector<Eigen::Vector3f> trace(std::vector<Eigen::Vector3f> &origins, std::vector<Eigen::Vector3f> &directions);
        std::vector<Eigen::Vector3f> nearest_point(std::vector<Eigen::Vector3f> &positions);
};

#endif