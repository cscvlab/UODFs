#include<TriangleBvh.cuh>
#include<stdio.h>

#ifdef GEO_OPTIX

#include<optix.h>
#include<optix_stubs.h>
#include<optix_function_table_definition.h>
#include<optix_stack_size.h>

#include "optix/ray_trace.h"
#include "optix/raystab.h"
#include "optix/pathescape.h"

#include<utils/m_optix_program.h>

#define RAYTRACE_PTX "./build/CMakeFiles/optix_program.dir/src/optix/ray_trace.ptx"
#define RAYSTAB_PTX "./build/CMakeFiles/optix_program.dir/src/optix/raystab.ptx"
#define PATHESCAPE_PTX "./build/CMakeFiles/optix_program.dir/src/optix/pathescape.ptx"

#endif

__global__ void signed_distance_watertight_kernel(Eigen::Vector3f *points, float *distances, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes, bool use_existing_distances_as_upper_bounds);
__global__ void signed_distance_raystab_kernel(Eigen::Vector3f *points, float *distances, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes, bool use_existing_distances_as_upper_bounds);
__global__ void unsigned_distance_kernel(Eigen::Vector3f *points, float *distances, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes, bool use_existing_distances_as_upper_bounds);
__global__ void raytrace_kernel(Eigen::Vector3f *points, Eigen::Vector3f *directions, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes);
__global__ void nearest_point_kernel(Eigen::Vector3f *points, Eigen::Vector3f *npoints, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes);
template<size_t BRANCH_FACTOR>
class TriangleBVHWithBranchFactor : public TriangleBVH{
    public:
        void build(std::vector<Triangle> &triangles, unsigned int n_primitives_per_leaf) override{
            m_nodes_cpu.clear();
            m_nodes_cpu.emplace_back();
            m_nodes_cpu.front().bb = BoundingBox(triangles.begin(), triangles.end());
            // std::cout << "Bounding of Mesh: " << 
            //         m_nodes_cpu.front().bb.min.x() << m_nodes_cpu.front().bb.min.y() << m_nodes_cpu.front().bb.min.z() <<
            //         m_nodes_cpu.front().bb.max.x() << m_nodes_cpu.front().bb.max.y() << m_nodes_cpu.front().bb.max.z() << std::endl;
            struct BuildNode{
                int node_idx;
                std::vector<Triangle>::iterator begin;
                std::vector<Triangle>::iterator end;
            };
            std::stack<BuildNode> build_stack;
            build_stack.push({0, std::begin(triangles), std::end(triangles)});
            while(!build_stack.empty()){
                const BuildNode &current = build_stack.top();
                size_t node_idx = current.node_idx;
                std::array<BuildNode, BRANCH_FACTOR> children;
                children[0].begin = current.begin;
                children[0].end = current.end;
                build_stack.pop();
                int n_children = 1;
                // Distribute triangles
                while(n_children < BRANCH_FACTOR){
                    for(int i=n_children - 1; i>=0; --i){
                        auto &child = children[i];
                        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
                        for(auto it = child.begin; it != child.end; ++it){
                            mean += it->centroid();
                        }
                        mean /= (float)std::distance(child.begin, child.end);
                        Eigen::Vector3f variance = Eigen::Vector3f::Zero();
                        for(auto it = child.begin; it != child.end; ++it){
                            Eigen::Vector3f diff = it->centroid() - mean;
                            variance += diff.cwiseProduct(diff);
                        }
                        variance /= (float)std::distance(child.begin, child.end);
                        Eigen::Vector3f::Index axis;
                        variance.maxCoeff(&axis);
                        auto mid = child.begin + std::distance(child.begin, child.end)/2;
                        // This contains sort
                        std::nth_element(child.begin, mid, child.end, [&](const Triangle &tri1, const Triangle &tri2){return tri1.centroid(axis) < tri2.centroid(axis);});
                        children[i*2].begin = children[i].begin;
                        children[i*2+1].end = children[i].end;
                        children[i*2].end = children[i*2+1].begin = mid;
                    }
                    n_children *= 2;
                }
                // Create next build nodes
                m_nodes_cpu[node_idx].first = (int)m_nodes_cpu.size();
                for(int i=0; i<BRANCH_FACTOR; i++){
                    auto &child = children[i];
                    assert(child.begin != child.end);
                    child.node_idx = (int)m_nodes_cpu.size();
                    m_nodes_cpu.emplace_back();
                    m_nodes_cpu.back().bb = BoundingBox(child.begin, child.end);
                    if(std::distance(child.begin, child.end) <= n_primitives_per_leaf){
                        m_nodes_cpu.back().first = -(int)std::distance(std::begin(triangles), child.begin) - 1;
                        m_nodes_cpu.back().last = -(int)std::distance(std::begin(triangles), child.end) - 1;
                    }else{
                        build_stack.push(child);
                    }
                }
                m_nodes_cpu[node_idx].last = (int)m_nodes_cpu.size();
            }
            
            m_nodes_gpu.resize_and_memcpy_from_vector(m_nodes_cpu);
            std::cout << "Build Triangle BVH Success. The number of nodes = " << m_nodes_gpu.size() 
                      << ". Allocate Memory: " << bytes_to_string(m_nodes_cpu.size()*sizeof(TriangleBvhNode)) << std::endl; 
        }
        void build_optix(GPUVector<Triangle> &triangles, cudaStream_t stream) override{
#ifdef GEO_OPTIX
            m_optix.available = initialize_optix();
            if(m_optix.available){
                std::string input_ptx;
                m_optix.gas = std::make_unique<Gas>(triangles, g_optix, stream);
                read_ptx_file(input_ptx, RAYTRACE_PTX);
                m_optix.raytrace = std::make_unique<Program<RayTrace>>((const char*)input_ptx.c_str(), input_ptx.size(), g_optix);
                read_ptx_file(input_ptx, RAYSTAB_PTX);
                m_optix.raystab = std::make_unique<Program<Raystab>>((const char*)input_ptx.c_str(), input_ptx.size(), g_optix);
                read_ptx_file(input_ptx, PATHESCAPE_PTX);
                m_optix.pathescape = std::make_unique<Program<PathEscape>>((const char*)input_ptx.c_str(), input_ptx.size(), g_optix);
                std::cout << "OptiX Program Compilation Success" << std::endl;
            }else{
                std::cout << "OptiX Build Failed" << std::endl;
            }
#else
                std::cout << "OptiX was not Built." << std::endl;
#endif
        }
        void signed_distance_gpu(Eigen::Vector3f *p_gpu, float *d_gpu, unsigned int num, Triangle *triangles_gpu, SDFCalcMode mode, bool use_existing_distances_as_upper_bound, cudaStream_t stream) override{
            unsigned int threads = 128;
            unsigned int blocks = (num % threads) ? (num / threads + 1) : num / threads;
            if(mode == SDFCalcMode::WATERTIGHT){
                signed_distance_watertight_kernel<<<blocks, threads, 0, stream>>>(
                    p_gpu, d_gpu, num, triangles_gpu, m_nodes_gpu.ptr(), use_existing_distances_as_upper_bound
                );
            }else{
#ifdef GEO_OPTIX
                if(m_optix.available){
                    unsigned_distance_kernel<<<blocks, threads, 0, stream>>>(
                        p_gpu, d_gpu, num, triangles_gpu, m_nodes_gpu.ptr(), use_existing_distances_as_upper_bound
                    );

                    if(mode == SDFCalcMode::RAYSTAB){
                        m_optix.raystab->invoke({p_gpu, d_gpu, m_optix.gas->handle()}, {num, 1, 1}, stream);
                    }else if(mode == SDFCalcMode::PATHESCAPE){
                        m_optix.pathescape->invoke({p_gpu, triangles_gpu, d_gpu, m_optix.gas->handle()}, {num, 1, 1}, stream);
                    }
                }else
#endif  // GEO_OPTIX
                {
                    if(mode == SDFCalcMode::RAYSTAB){
                        signed_distance_raystab_kernel<<<blocks, threads, 0, stream>>>(
                            p_gpu, d_gpu, num, triangles_gpu, m_nodes_gpu.ptr(), use_existing_distances_as_upper_bound
                        );
                    }else if(mode == SDFCalcMode::PATHESCAPE){
                        throw std::runtime_error("PATHESCAPE MODE only available while optix is enabled.");
                    }
                }
            }
        }
        void ray_trace_gpu(Eigen::Vector3f *p_gpu, Eigen::Vector3f *d_gpu, uint32_t num, Triangle *triangles_gpu, cudaStream_t stream) override{
#ifdef GEO_OPTIX
            if(m_optix.available){
                m_optix.raytrace->invoke({p_gpu, d_gpu, triangles_gpu, m_optix.gas->handle()}, {num, 1, 1}, stream);
            }else
#endif  // GEO_OPTIX
            {
                unsigned int threads = 128;
                unsigned int blocks = (num % threads) ? (num / threads + 1) : num / threads;
                raytrace_kernel<<<blocks, threads, 0, stream>>>(p_gpu, d_gpu, num, triangles_gpu, m_nodes_gpu.ptr());
                cudaDeviceSynchronize();
            }
        }
        void nearest_point(Eigen::Vector3f *p_gpu, Eigen::Vector3f *p_out, unsigned int num, Triangle *triangles_gpu, cudaStream_t stream) override{
            unsigned int threads = 128;
            unsigned int blocks = (num % threads) ? (num / threads + 1) : num / threads;
            nearest_point_kernel<<<blocks, threads, 0, stream>>>(
                p_gpu, p_out, num, triangles_gpu, m_nodes_gpu.ptr()
            );

        }
        void unsigned_distance_gpu(Eigen::Vector3f *p_gpu, float *d_gpu, unsigned int num, Triangle *triangles_gpu, cudaStream_t stream) override{
            unsigned int threads = 128;
            unsigned int blocks = (num % threads) ? (num / threads + 1) : num / threads;
            unsigned_distance_kernel<<<blocks, threads, 0, stream>>>(
                p_gpu, d_gpu, num, triangles_gpu, m_nodes_gpu.ptr(), false
            );
        }
        
        __host__ __device__ static std::pair<int, float> closest_triangle(Eigen::Vector3f &p, Triangle *triangles, TriangleBvhNode *nodes, float max_distance2 = MAX_DIST2){
                FixedIntStack query_stack;
                query_stack.push(0);
                
                float shortest_distance2 = max_distance2;
                int shorest_idx = -1;

                while(!query_stack.empty()){
                    int idx = query_stack.pop();
                    const TriangleBvhNode &node = nodes[idx];

                    if(node.first < 0){ // leaf node
                        int end = -node.last - 1;
                        for(int i= -node.first-1; i<end; ++i){
                            float dist2 = triangles[i].distance_from_point_2(p);
                            if(dist2 <= shortest_distance2){
                                shortest_distance2 = dist2;
                                shorest_idx = i;
                            }
                        }
                    }else{
                        DistAndIdx children[BRANCH_FACTOR];

                        unsigned int first = node.first;

                        for(unsigned int i = 0; i < BRANCH_FACTOR; ++i){
                            children[i] = {nodes[i+first].bb.distance_2(p), i+first};
                        }

                        sorting_network<BRANCH_FACTOR, DistAndIdx>(children);

                        for(unsigned int i=0; i<BRANCH_FACTOR; ++i){
                            if(children[i].dist <= shortest_distance2){
                                query_stack.push(children[i].idx);
                            }
                        }

                    }
                }

                if(shorest_idx == -1){  // This must a bug
                    shorest_idx = 0;
                    shortest_distance2 = 0.0f;
                }

                return {shorest_idx, std::sqrt(shortest_distance2)};
            }
        __host__ __device__ static Eigen::Vector3f avg_normal_around_points(Eigen::Vector3f &p, Triangle *triangles, TriangleBvhNode *nodes){
                FixedIntStack query_stack;
                query_stack.push(0);

                static constexpr float EPISILON = 1e-6f;
                float total_weight = 0;
                Eigen::Vector3f result = Eigen::Vector3f::Zero();

                while(!query_stack.empty()){
                    int idx = query_stack.pop();
                    const TriangleBvhNode &node = nodes[idx];

                    if(node.first < 0){ //leaf node
                        int end = -node.last -1;
                        for(int i=-node.first-1; i< end; ++i){
                            if(triangles[i].distance_from_point_2(p) < EPISILON){
                                result += triangles[i].normal();
                                total_weight += 1;
                            }
                        }
                    }else{
                        unsigned int first_child = node.first;
                        
                        for(unsigned int i=0; i < BRANCH_FACTOR; ++i){
                            if(nodes[i+first_child].bb.distance_2(p) < EPISILON){
                                query_stack.push(i+first_child);
                            }
                        }
                    }
                }

                return result / total_weight;
            }
        __host__ __device__ static float unsigned_distance(Eigen::Vector3f &p, Triangle *triangles, TriangleBvhNode *nodes){
                return closest_triangle(p, triangles, nodes).second;
            }
        __host__ __device__ static std::pair<int, float> ray_intersect(const Eigen::Vector3f &o, const Eigen::Vector3f &d, const Triangle *triangles, const TriangleBvhNode *nodes){
                FixedIntStack query_stack;
                query_stack.push(0);

                float mint = MAX_DIST;
                int shortest_idx = -1;
                
                while(!query_stack.empty()){
                    int idx = query_stack.pop();

                    const TriangleBvhNode &node = nodes[idx];

                    if(node.first < 0){
                        int end = -node.last - 1;
                        for(int i= -node.first-1; i<end; ++i){
                            float t = triangles[i].ray_intersect(o, d);
                            if(t < mint){
                                mint = t;
                                shortest_idx = i;
                            }
                        }
                    }else{
                        DistAndIdx children[BRANCH_FACTOR];
                        int first_child = node.first;
                        
                        for(unsigned int i = 0; i < BRANCH_FACTOR; ++i){
                            children[i] = {nodes[i+first_child].bb.ray_intersect(o, d).x(), i+first_child};
                        }

                        sorting_network<BRANCH_FACTOR, DistAndIdx>(children);

                        for(unsigned int i=0; i<BRANCH_FACTOR; ++i){
                            if(children[i].dist < mint){
                                query_stack.push(children[i].idx);
                            }
                        }
                    }
                }

                return {shortest_idx, mint};
            }
        __host__ __device__ static float signed_distance_watertight(Eigen::Vector3f &p, Triangle *triangles, TriangleBvhNode *nodes, float max_distance2 = MAX_DIST2){
                auto pair = closest_triangle(p, triangles, nodes, max_distance2);
                const Triangle &tri = triangles[pair.first];
                Eigen::Vector3f closest_point = tri.closest_point_to(p);
                Eigen::Vector3f avg_normal = avg_normal_around_points(closest_point, triangles, nodes);
                return std::copysignf(pair.second, avg_normal.dot(p - closest_point));
            }
        __host__ __device__ static float signed_distance_raystab(Eigen::Vector3f &p, Triangle *triangles, TriangleBvhNode *nodes, float max_distance2 = MAX_DIST2, default_rng_t rng={}){
                float distance = unsigned_distance(p, triangles, nodes);

                Eigen::Vector2f offset = {rng.nextFloat(), rng.nextFloat()};

                for(unsigned int i = 0; i < N_STAB_RAYS; ++i){
                    Eigen::Vector3f d = fibonacci_dir<N_STAB_RAYS>(i, offset);
                    // Exist one ray does not hit even a triangle, judge as outside
                    if(ray_intersect(p, d, triangles, nodes).first < 0 || ray_intersect(p, -d, triangles, nodes).first < 0){
                        return distance;
                    }
                }
                return -distance;
            }

#ifdef GEO_OPTIX
        private:
            struct{
                std::unique_ptr<Gas> gas;
                std::unique_ptr<Program<RayTrace>> raytrace;
                std::unique_ptr<Program<Raystab>> raystab;
                std::unique_ptr<Program<PathEscape>> pathescape;
                bool available = false;
            } m_optix;
#endif
};
using TriangleBVH4 = TriangleBVHWithBranchFactor<4>;
std::unique_ptr<TriangleBVH> TriangleBVH::create(){
    return std::unique_ptr<TriangleBVH>(new TriangleBVH4());
}
__global__ void signed_distance_watertight_kernel(Eigen::Vector3f *p_gpu, float *d_gpu, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes, bool use_existing_distances_as_upper_bounds){
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= num)return;
    float max_distance = use_existing_distances_as_upper_bounds ? d_gpu[i] : MAX_DIST;
    d_gpu[i] = TriangleBVH4::signed_distance_watertight(p_gpu[i], triangles, nodes, max_distance*max_distance);
}
__global__ void signed_distance_raystab_kernel(Eigen::Vector3f *p_gpu, float *d_gpu, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes, bool use_existing_distances_as_upper_bounds){
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= num)return;
    float max_distance = use_existing_distances_as_upper_bounds ? d_gpu[i] : MAX_DIST;
    default_rng_t rng;
    rng.advance(i * 2);
    d_gpu[i] = TriangleBVH4::signed_distance_raystab(p_gpu[i], triangles, nodes, max_distance*max_distance, rng);
}
__global__ void unsigned_distance_kernel(Eigen::Vector3f *points, float *distances, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes, bool use_existing_distances_as_upper_bounds){
    const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= num)return;
    // float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;
    distances[i] = TriangleBVH4::unsigned_distance(points[i], triangles, nodes);
}
__global__ void raytrace_kernel(Eigen::Vector3f *p_gpu, Eigen::Vector3f *d_gpu, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes){
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= num)return;
    auto pos = p_gpu[i];
    auto dir = d_gpu[i];
    auto p = TriangleBVH4::ray_intersect(pos, dir, triangles, nodes);
    p_gpu[i] = pos + p.second * dir;
    if(p.first >= 0){
        Eigen::Vector3f normal = triangles[p.first].normal();
        d_gpu[i] = dir.dot(normal) < 0 ? normal : -normal;
    }
}
__global__ void nearest_point_kernel(Eigen::Vector3f *points, Eigen::Vector3f *npoints, unsigned int num, Triangle *triangles, TriangleBvhNode *nodes){
    const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= num)return;
    Eigen::Vector3f p = points[i];
    int idx = TriangleBVH4::closest_triangle(p, triangles, nodes).first;
    Triangle t = triangles[idx];
    npoints[i] = t.closest_point_to(p);
}
