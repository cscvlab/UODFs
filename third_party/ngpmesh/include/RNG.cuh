// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
#pragma once
#ifndef RNG_CUH
#define RNG_CUH

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL

#include<vector>

struct pcg32_random_t{
    size_t state; 
    size_t inc;
    __host__ __device__ pcg32_random_t(): state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM){}
    __host__ __device__ pcg32_random_t(size_t init_state, size_t init_inc = 1u): state(init_state), inc(init_inc){}
    __host__ __device__ void seed(size_t init_state, size_t init_inc = 1){
        state = 0U;
        inc = (init_inc << 1u) | 1u;
        nextUInt();
        state += init_state;
        nextUInt();
    }
    // Reference code from pcg-random.org
    __host__ __device__ unsigned int nextUInt(){
        size_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + (inc|1);
        unsigned int xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        unsigned int rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    // Reference Code from instant-ngp
    __host__ __device__ float nextFloat(){
        /* Trick from MTGP: generate an uniformly distributed
            single precision number in [1,2) and subtract 1. */
        union {
            unsigned int u;
            float f;
        } x;
        x.u = (nextUInt() >> 9) | 0x3f800000u;
        return x.f - 1.0f;
    }

    /**
     * \brief Multi-step advance function (jump-ahead, jump-back)
     *
     * Reference Code from instant-ngp
     * 
     * The method used here is based on Brown, "Random Number Generation
     * with Arbitrary Stride", Transactions of the American Nuclear
     * Society (Nov. 1994). The algorithm is very similar to fast
     * exponentiation.
     *
     * The default value of 2^32 ensures that the PRNG is advanced
     * sufficiently far that there is (likely) no overlap with
     * previously drawn random numbers, even if small advancements.
     * are made inbetween.
     */
    __host__ __device__ void advance(int64_t delta_ = (1ll<<32)) {
        size_t
            cur_mult = PCG32_MULT,
            cur_plus = inc,
            acc_mult = 1u,
            acc_plus = 0u;

        /* Even though delta is an unsigned integer, we can pass a signed
            integer to go backwards, it just goes "the long way round". */
        size_t delta = (size_t) delta_;

        while (delta > 0) {
            if (delta & 1) {
                acc_mult *= cur_mult;
                acc_plus = acc_plus * cur_mult + cur_plus;
            }
            cur_plus = (cur_mult + 1) * cur_plus;
            cur_mult *= cur_mult;
            delta /= 2;
        }
        state = acc_mult * state + acc_plus;
    }
};

typedef pcg32_random_t default_rng_t;

template<typename T, size_t N_TO_GENERATE, typename F>
__global__ void generate_random_kernel(T *arr, size_t num, pcg32_random_t rng, F transform){
    const size_t i = (threadIdx.x + blockIdx.x * blockDim.x) * N_TO_GENERATE;
    rng.advance(i);
    #pragma unroll
    for(size_t j=0; j<N_TO_GENERATE; j++){
        if(i+j >= num)return;
        arr[i+j] = transform((T)rng.nextFloat());
    }
}

template<typename T, typename F>
void generate_random(T *arr, size_t num, pcg32_random_t &rng, F &&transform, cudaStream_t stream){
    const size_t N_TO_GENERATE = 8;
    const uint32_t blocks = div_round_up(num, 128*N_TO_GENERATE);
    generate_random_kernel<T, N_TO_GENERATE><<<blocks, 128, 0, stream>>>(arr, num, rng, transform);
    rng.advance(num);
}

template<typename T>
void generate_random_uniform(T *arr, size_t num, pcg32_random_t &rng, T min = (T)0.0f, T max = (T)1.0f, cudaStream_t stream = nullptr){
    return generate_random<T>(arr, num, rng, [max, min] __device__(T val){ return val * (max - min) + min;}, stream);
}

template<typename T>
void generate_random_logistic(T *arr, size_t num, pcg32_random_t &rng, T mean = (T)0.0f, T stddev = (T)1.0f, cudaStream_t stream = nullptr){
    return generate_random<T>(arr, num, rng, [mean, stddev] __device__(T val){ return (T)logit(val) * stddev * 0.551328895f + mean; }, stream);
}

#endif
