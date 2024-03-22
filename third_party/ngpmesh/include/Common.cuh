#pragma once
#ifndef COMMON_CUH
#define COMMON_CUH

#include<Eigen/Eigen>

static constexpr float GOLDEN_RATIO = 1.6180339887498948482045868343656f;
static constexpr float PI = 3.14159265358979323846f;

__host__ __device__ inline float sign(float x){
	return copysignf(1.0f, x);
}

template<typename T>
__host__ __device__ inline T clamp(T x, T min, T max){
	if(x<min)return min;
	else if(x > max)return max;
	else return x;
}

template<typename T, int Cols>
__host__ __device__ inline Eigen::Vector<T, Cols> clamp(const Eigen::Vector<T, Cols> &x, T min, T max){
    auto max_ = Eigen::Vector<T, Cols>::Constant(max);
    auto min_ = Eigen::Vector<T, Cols>::Constant(min);
    Eigen::Vector<T, Cols> ret = x;
    ret = ret.cwiseMin(max_);
    ret = ret.cwiseMax(min_);
	return ret;
}

template<typename T>
__host__ __device__ inline void swap_value(T &a, T &b){
	T c(a);
	a = b;
	b = c;
}

template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

__host__ __device__ inline float fractf(float x){
	return x - floorf(x);
}

__host__ __device__ inline float logit(const float x) {	// 1e-9 < x < 1-1e-9
	// -log(1/x) - 1
	return -logf(1.0f / (fminf(fmaxf(x, 1e-9f), 1.0f - 1e-9f)) - 1.0f);
}

__host__ __device__ inline uint32_t binary_search(float val, const float* data, uint32_t length) {
	if (length == 0) {
		return 0;
	}

	uint32_t it;
	uint32_t count, step;
	count = length;

	uint32_t first = 0;
	while (count > 0) {
		it = first;
		step = count / 2;
		it += step;
		if (data[it] < val) {
			first = ++it;
			count -= step + 1;
		} else {
			count = step;
		}
	}

	return std::min(first, length-1);
}

__host__ __device__ inline Eigen::Vector3f cylindrical_to_dir(const Eigen::Vector2f& p) {
	const float cos_theta = -2.0f * p.x() + 1.0f;
	const float phi = 2.0f * PI * (p.y() - 0.5f);

	const float sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));
	float sin_phi, cos_phi;
	sincosf(phi, &sin_phi, &cos_phi);

	return {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
}

// Reference Code from instant-ngp
template<unsigned int N_DIRS>
__host__ __device__ inline Eigen::Vector3f fibonacci_dir(unsigned int i, Eigen::Vector2f offset = Eigen::Vector2f::Zero()){
	float epsilon;
	if (N_DIRS >= 11000) {
		epsilon = 27;
	} else if (N_DIRS >= 890) {
		epsilon = 10;
	} else if (N_DIRS >= 177) {
		epsilon = 3.33;
	} else if (N_DIRS >= 24) {
		epsilon = 1.33;
	} else {
		epsilon = 0.33;
	}

	return cylindrical_to_dir(
        Eigen::Vector2f{
            fractf((i+epsilon) / (N_DIRS-1+2*epsilon) + offset.x()), 
			fractf(i / GOLDEN_RATIO + offset.y())
            }
        );
}




inline void split(std::string s, std::vector<std::string> &results, char delimeter){
	results.clear();
	if(delimeter == ' '){	// it may appear to skip several contiuous space
		std::string temp = "";
		for(int i=0; i<=s.length(); i++){
			if(s[i] == ' ' || s[i] == '\0'){
				if(temp.length())results.push_back(temp);
			}else{
				temp += s[i];
			}
		}
	}else{
		std::stringstream ss(s);
		std::string temp;
		while(getline(ss, temp, delimeter)){
			results.push_back(temp);
		}
	}
}

inline bool startWith(std::string s, std::string start){
	for(int i=0; start[i] != '\0'; i++){
		if(s[i] != start[i])return false;
	}
	return true;
}

inline std::string bytes_to_string(size_t bytes) {
	std::array<std::string, 7> suffixes = {{ "B", "KB", "MB", "GB", "TB", "PB", "EB" }};

	double count = (double)bytes;
	uint32_t i = 0;
	for (; i < suffixes.size() && count >= 1024; ++i) {
		count /= 1024;
	}

	std::ostringstream oss;
	oss.precision(3);
	oss << count << " " << suffixes[i];
	return oss.str();
}




#endif // COMMON_CUH