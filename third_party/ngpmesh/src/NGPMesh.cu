#include<NGPMesh.cuh>

inline __device__ __host__ uint32_t sample_idx(float prob, float *distri_int, uint32_t length){
    return binary_search(prob, distri_int, length);
}

inline __device__ __host__ Eigen::Matrix<float, 3, 4> get_camera_matrix(Eigen::Vector3f cam_pos, Eigen::Vector3f cam_to){
    Eigen::Matrix<float, 3, 4> camera_matrix;
    camera_matrix.col(3) = cam_pos;
    camera_matrix.col(2) = (cam_to - cam_pos).normalized();
    camera_matrix.col(1) = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    camera_matrix.col(0) = camera_matrix.col(2).cross(camera_matrix.col(1)).normalized();
    camera_matrix.col(1) = camera_matrix.col(0).cross(camera_matrix.col(2)).normalized();
    return camera_matrix;
}

inline __device__ __host__ Eigen::Vector2f calc_focal_length(Eigen::Vector2i win_res, float fov) {
    return Eigen::Vector2f::Constant((0.5f * 1.0f / tanf(0.5f * fov * 3.14159265f/180))) * win_res[0];
}

inline __host__ __device__ Eigen::Vector2f ld_random_pixel_offset(const uint32_t /*x*/, const uint32_t /*y*/) {
    Eigen::Vector2f offset = Eigen::Vector2f::Constant(0.5f);
    offset.x() = fractf(offset.x());
    offset.y() = fractf(offset.y());
    return offset;
}

inline __host__ __device__ Eigen::Vector3f init_rays_direction(
    const Eigen::Vector2i& pixel,
    const Eigen::Vector2i& resolution,
    const Eigen::Vector2f& focal_length,
    const Eigen::Matrix<float, 3, 4>& camera_matrix,
    Eigen::Vector2f screen_center
) {
    Eigen::Vector2f offset = ld_random_pixel_offset(pixel.x(), pixel.y());
    Eigen::Vector2f uv = (pixel.cast<float>() + offset).cwiseQuotient(resolution.cast<float>());
    Eigen::Vector3f dir;
    
    dir = {
        (uv.x() - screen_center.x()) * (float)resolution.x() / focal_length.x(),
        (uv.y() - screen_center.y()) * (float)resolution.y() / focal_length.y(),
        1.0f
    };
    dir = camera_matrix.block<3, 3>(0, 0) * dir;
    return dir;
}

__global__ void sample_uniform_on_triangle_kernel(
    Eigen::Vector3f *positions,
    uint32_t num,
    float *distri_int,
    uint32_t length,
    Triangle *triangles
){
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i>=num)return;

    Eigen::Vector3f sample = positions[i];
    uint32_t tri_idx = sample_idx(sample.x(), distri_int, length);

    positions[i] = triangles[tri_idx].sample_uniform_position(sample.tail<2>());
}

__global__ void sample_uniform_on_aabb_kernel(
    Eigen::Vector3f *positions,
    uint32_t num,
    BoundingBox aabb
){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= num)return;

    positions[idx] = aabb.min + positions[idx].cwiseProduct(aabb.diag());
}

__global__ void sample_perturbation_near_triangle_kernel(
    Eigen::Vector3f *positions,
    Eigen::Vector3f *perturb,
    uint32_t num
){
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= num)return;

    positions[i] = positions[i] + perturb[i];
}

__global__ void init_rays_from_camera_kernel(
    Eigen::Vector3f *positions,
    Eigen::Vector3f *directions,
    Eigen::Vector2i resolution,
    Eigen::Vector2f focal_length,
    Eigen::Matrix<float, 3, 4> camera_matrix,
    BoundingBox aabb
){
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= resolution.x() || y >= resolution.y()) return;

    uint32_t idx = x + resolution.x() * y;

    Eigen::Vector3f origin = camera_matrix.col(3);
    Eigen::Vector3f direction = init_rays_direction({x, y}, resolution, focal_length, camera_matrix, {0.5, 0.5});
    direction = direction.normalized();

    float t = std::max(aabb.ray_intersect(origin, direction).x(), 0.0f);
    origin = origin + (t + 1e-6f) * direction;
    positions[idx] = origin;
    directions[idx] = direction;

}

inline std::vector<float> triangle_weights(std::vector<Triangle> &triangles){
    std::vector<float> weights(triangles.size());
    float total_area = 0.0f;
    for(size_t i=0; i < triangles.size(); i++){
        weights[i] = triangles[i].area();
        total_area += weights[i];
    }
    float inv_total_area = 1.0f / total_area;
    for(size_t i=0; i < triangles.size(); i++){
        weights[i] *= inv_total_area;
    }
    return weights;
}

inline std::vector<float> triangle_weights_integration(std::vector<float> &weights){
    std::vector<float> integrate(weights.size());
    float accumulate = 0.0f;
    for(size_t i=0; i < weights.size(); i++){
        accumulate += weights[i];
        integrate[i] = accumulate;
    }
    integrate.back() = 1.0f;
    return integrate;
}

void NGPMesh::load_mesh(std::vector<Eigen::Matrix3f> &triangles_matrix){
    triangles_cpu = loadTriangles(triangles_matrix);

    bvh = TriangleBVH::create();
    bvh->build(triangles_cpu, 8);

    triangles_gpu.resize_and_memcpy_from_vector(triangles_cpu);
    bvh->build_optix(triangles_gpu, stream);
}

void NGPMesh::load_mesh(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector<int, 3>> &faces){
    triangles_cpu = loadTriangles(vertices, faces);

    bvh = TriangleBVH::create();
    bvh->build(triangles_cpu, 8);

    triangles_gpu.resize_and_memcpy_from_vector(triangles_cpu);
    bvh->build_optix(triangles_gpu, stream);
}

std::vector<float> NGPMesh::unsigned_distance(std::vector<Eigen::Vector3f> &positions){
    GPUVector<Eigen::Vector3f> positions_gpu(positions.size());
    std::vector<float> distances_cpu(positions.size());
    GPUVector<float> distances_gpu(positions.size());
    positions_gpu.memcpyfrom(&positions[0], positions.size(), stream);
    clock_t start, end;
    start = clock();
    bvh->unsigned_distance_gpu(positions_gpu.ptr(), distances_gpu.ptr(), positions.size(), triangles_gpu.ptr(), stream);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Calculate UDF Cost " << (float(end - start)/CLOCKS_PER_SEC) << "s." << std::endl;
    distances_gpu.memcpyto(&distances_cpu[0], positions.size(), stream);
    return distances_cpu;
}

std::vector<float> NGPMesh::signed_distance(std::vector<Eigen::Vector3f> &positions, SDFCalcMode mode){
    GPUVector<Eigen::Vector3f> positions_gpu(positions.size());
    std::vector<float> distances_cpu(positions.size());
    GPUVector<float> distances_gpu(positions.size());
    positions_gpu.memcpyfrom(&positions[0], positions.size(), stream);
    clock_t start, end;
    start = clock();
    bvh->signed_distance_gpu(positions_gpu.ptr(), distances_gpu.ptr(), positions.size(), triangles_gpu.ptr(), mode, false, stream);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Calculate SDF Cost " << (float(end - start)/CLOCKS_PER_SEC) << "s." << std::endl;
    distances_gpu.memcpyto(&distances_cpu[0], positions.size(), stream);
    return distances_cpu;
}

std::vector<Eigen::Vector3f> NGPMesh::trace(std::vector<Eigen::Vector3f> &positions, std::vector<Eigen::Vector3f> &directions){
    GPUVector<Eigen::Vector3f> positions_gpu;
    GPUVector<Eigen::Vector3f> directions_gpu;
    std::vector<Eigen::Vector3f> res(positions.size());
    positions_gpu.resize_and_memcpy_from_vector(positions, stream);
    directions_gpu.resize_and_memcpy_from_vector(directions, stream);

    bvh->ray_trace_gpu(positions_gpu.ptr(), directions_gpu.ptr(), positions.size(), triangles_gpu.ptr(), stream);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    positions_gpu.memcpyto(res.data(), positions.size(), stream);
    return res;
}

std::vector<Eigen::Vector3f> NGPMesh::nearest_point(std::vector<Eigen::Vector3f> &positions){
    GPUVector<Eigen::Vector3f> positions_gpu;
    GPUVector<Eigen::Vector3f> target_gpu(positions.size());
    std::vector<Eigen::Vector3f> target_cpu(positions.size());
    positions_gpu.resize_and_memcpy_from_vector(positions, stream);

    bvh->nearest_point(positions_gpu.ptr(), target_gpu.ptr(), positions.size(), triangles_gpu.ptr(), stream);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    target_gpu.memcpyto(target_cpu.data(), positions.size(), stream);
    return target_cpu;
}