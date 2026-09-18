// Проверка flash::qk_int8_scores — ровно та форма вызова, что будет в ядре
// (Q/K int8, масштабы множителями, f32-аккумулятор). Эталон считается на CPU.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
// Преамбула как в flash_fwd_kernel.h: utils.h не самодостаточен.
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"

using namespace cute;
using flash::qk_int8_scores;

constexpr int M = 64, N = 32, K = 256, THREADS = 128;

__global__ void helper_kernel(const int8_t* __restrict__ q, const int8_t* __restrict__ k,
                              const __half* __restrict__ qs, const __half* __restrict__ ks,
                              float* __restrict__ out) {
    __shared__ int8_t sQ[M * K];
    __shared__ int8_t sK[N * K];
    for (int i = threadIdx.x; i < M * K; i += THREADS) sQ[i] = q[i];
    for (int i = threadIdx.x; i < N * K; i += THREADS) sK[i] = k[i];
    __syncthreads();
    using LQ  = Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>;
    using LK  = Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>;
    using LQS = Layout<Shape<Int<M>, Int<1>>, Stride<Int<1>, Int<1>>>;
    using LKS = Layout<Shape<Int<N>>, Stride<Int<1>>>;
    using LC  = Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>;
    auto tQ  = make_tensor(make_smem_ptr(sQ), LQ{});
    auto tK  = make_tensor(make_smem_ptr(sK), LK{});
    auto gQS = make_tensor(make_gmem_ptr(qs), LQS{});
    auto gKS = make_tensor(make_gmem_ptr(ks), LKS{});
    using TiledMmaS8 = decltype(make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{}, Layout<Shape<Int<4>, Int<1>, Int<1>>>{}));
    TiledMmaS8 mma8;
    auto thr = mma8.get_thread_slice(threadIdx.x);
    Tensor acc = partition_fragment_C(mma8, Shape<Int<M>, Int<N>>{});
    qk_int8_scores<M, N, K, 4>(tQ, gQS, tK, gKS, acc, threadIdx.x, 0, 0, M, 1.0f);
    auto gC = make_tensor(make_gmem_ptr(out), LC{});
    copy(acc, thr.partition_C(gC));
}


// Инлайновый вариант — ровно тот код, что дал 0 несовпадений в спайке §91.
__global__ void inline_kernel(const int8_t* __restrict__ q, const int8_t* __restrict__ k,
                              float* __restrict__ out) {
    __shared__ int8_t sQ[M * K];
    __shared__ int8_t sK[N * K];
    for (int i = threadIdx.x; i < M * K; i += THREADS) sQ[i] = q[i];
    for (int i = threadIdx.x; i < N * K; i += THREADS) sK[i] = k[i];
    __syncthreads();
    using LQ  = Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>;
    using LK  = Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>;
    using LC  = Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>;
    auto tQ = make_tensor(make_smem_ptr(sQ), LQ{});
    auto tK = make_tensor(make_smem_ptr(sK), LK{});
    using TiledMmaS8 = decltype(make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{}, Layout<Shape<Int<4>, Int<1>, Int<1>>>{}));
    TiledMmaS8 mma8;
    auto thr = mma8.get_thread_slice(threadIdx.x);
    auto tAsQ = thr.partition_A(tQ);
    auto tBsK = thr.partition_B(tK);
    auto tArQ = thr.partition_fragment_A(tQ);
    auto tBrK = thr.partition_fragment_B(tK);
    auto acc32 = partition_fragment_C(mma8, Shape<Int<M>, Int<N>>{});
    clear(acc32);
    #pragma unroll
    for (int kb = 0; kb < K / 32; ++kb) {
        copy(tAsQ(_, _, kb), tArQ(_, _, kb));
        copy(tBsK(_, _, kb), tBrK(_, _, kb));
        gemm(mma8, tArQ(_, _, kb), tBrK(_, _, kb), acc32);
    }
    Tensor accf = make_tensor<float>(acc32.layout());
    #pragma unroll
    for (int i = 0; i < size(acc32); ++i) accf(i) = static_cast<float>(acc32(i));
    auto gC = make_tensor(make_gmem_ptr(out), LC{});
    copy(accf, thr.partition_C(gC));
}

int main() {
    static int8_t q[M * K], k[N * K];
    static __half qs[M], ks[N];
    static float out[M * N];
    for (int i = 0; i < M * K; ++i) q[i] = (int8_t)((i * 37 % 255) - 127);
    for (int i = 0; i < N * K; ++i) k[i] = (int8_t)((i * 53 % 255) - 127);
    for (int i = 0; i < M; ++i) qs[i] = __float2half(1.0f);
    for (int i = 0; i < N; ++i) ks[i] = __float2half(1.0f);
    int8_t *dq, *dk; __half *dqs, *dks; float* dout;
    cudaMalloc(&dq, M * K); cudaMalloc(&dk, N * K);
    cudaMalloc(&dqs, sizeof(__half) * M); cudaMalloc(&dks, sizeof(__half) * N);
    cudaMalloc(&dout, sizeof(float) * M * N);
    cudaMemcpy(dq, q, M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k, N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dqs, qs, sizeof(__half) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dks, ks, sizeof(__half) * N, cudaMemcpyHostToDevice);
    helper_kernel<<<1, THREADS>>>(dq, dk, dqs, dks, dout);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(out, dout, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    int bad = 0; float maxd = 0.0f;
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float ref = 0.0f;
        for (int kk = 0; kk < K; ++kk) ref += (float)q[m*K+kk] * (float)k[n*K+kk];
        float d = out[m*N+n] - ref; if (d < 0) d = -d;
        if (d > 0.5f) bad++; if (d > maxd) maxd = d;
    }
    printf("qk_int8_scores: несовпадений=%d из %d, max|diff|=%.2f\n", bad, M * N, maxd);
    {   // дифференциальный тест: инлайновый вариант против хелпера
        static float outi[M * N];
        float* douti; cudaMalloc(&douti, sizeof(float) * M * N);
        inline_kernel<<<1, THREADS>>>(dq, dk, douti);
        cudaError_t e2 = cudaDeviceSynchronize();
        if (e2 != cudaSuccess) printf("inline: CUDA error %s\n", cudaGetErrorString(e2));
        cudaMemcpy(outi, douti, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        int bad2 = 0; float maxd2 = 0.0f;
        for (int i = 0; i < M * N; ++i) {
            float dd = out[i] - outi[i]; if (dd < 0) dd = -dd;
            if (dd > 0.5f) bad2++; if (dd > maxd2) maxd2 = dd;
        }
        printf("helper vs inline: несовпадений=%d из %d, max|diff|=%.2f\n", bad2, M * N, maxd2);
        cudaFree(douti);
    }
    int shown = 0;
    for (int m = 0; m < M && shown < 8; ++m) for (int n = 0; n < N && shown < 8; ++n) {
        float ref = 0.0f;
        for (int kk = 0; kk < K; ++kk) ref += (float)q[m*K+kk] * (float)k[n*K+kk];
        float d = out[m*N+n] - ref; if (d < 0) d = -d;
        if (d > 0.5f) { printf("  mismatch m=%d n=%d got=%.0f ref=%.0f\n", m, n, out[m*N+n], ref); shown++; }
    }
    return bad ? 2 : 0;
}
