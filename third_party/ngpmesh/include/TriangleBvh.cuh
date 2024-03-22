#pragma once
#ifndef TRIANGLEBVH
#define TRIANGLEBVH

#include<stack>
#include<array>
#include<iostream>
#include<memory>

#include<Boundingbox.cuh>
#include<Triangle.cuh>
#include<GPUVector.cuh>
#include<RNG.cuh>


constexpr float MAX_DIST = 10.0f;
constexpr float MAX_DIST2 = MAX_DIST * MAX_DIST;

static constexpr uint32_t N_STAB_RAYS = 32;

struct DistAndIdx{
    float dist;
    size_t idx;

    __host__ __device__ bool operator<(const DistAndIdx &other){
        return dist < other.dist;
    }
};

template <typename T>
__host__ __device__ void inline compare_and_swap(T& t1, T& t2) {
    if (t1 < t2) {
        T tmp{t1}; t1 = t2; t2 = tmp;
    }
}

    // Sorting networks from http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N4L5D3
template <uint32_t N, typename T>
__host__ __device__ void sorting_network(T values[N]) {
    static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
    if (N <= 1) {
        return;
    } else if (N == 2) {
        compare_and_swap(values[0], values[1]);
    } else if (N == 3) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 4) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 5) {
        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[2], values[3]);
    } else if (N == 6) {
        compare_and_swap(values[0], values[5]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[2], values[5]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
    } else if (N == 7) {
        compare_and_swap(values[0], values[6]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[5]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[4], values[6]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    } else if (N == 8) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[4], values[6]);
        compare_and_swap(values[5], values[7]);
        compare_and_swap(values[0], values[4]);
        compare_and_swap(values[1], values[5]);
        compare_and_swap(values[2], values[6]);
        compare_and_swap(values[3], values[7]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[6], values[7]);
        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[3], values[5]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    }
}

enum class SDFCalcMode{ WATERTIGHT, RAYSTAB, PATHESCAPE};
struct TriangleBvhNode{
    BoundingBox bb;
    int first;  // negative represent leaves node
    int last;
};

// Reference code of instant-ngp
class TriangleBVH{
    protected:
        std::vector<TriangleBvhNode> m_nodes_cpu;
        GPUVector<TriangleBvhNode> m_nodes_gpu;
        TriangleBVH(){};
    public:
        virtual void build(std::vector<Triangle> &triangles, unsigned int n_primitives_per_leaf) = 0;
        virtual void build_optix(GPUVector<Triangle> &triangles, cudaStream_t stream) = 0;
        virtual void nearest_point(Eigen::Vector3f *p_gpu, Eigen::Vector3f *p_out, unsigned int num, Triangle *triangles_gpu, cudaStream_t stream) = 0;
        virtual void unsigned_distance_gpu(Eigen::Vector3f *p_gpu, float *d_gput, unsigned int num, Triangle *triangles_gpu, cudaStream_t stream) = 0;
        virtual void signed_distance_gpu(Eigen::Vector3f *p_gpu, float *d_gpu, unsigned int num, Triangle *triangles_gpu, SDFCalcMode mode, bool use_existing_distances_as_upper_bound, cudaStream_t stream) = 0;
        virtual void ray_trace_gpu(Eigen::Vector3f *p_gpu, Eigen::Vector3f *d_gpu, uint32_t num, Triangle *triangles_gpu, cudaStream_t stream) = 0;
        static std::unique_ptr<TriangleBVH> create();
};

#endif