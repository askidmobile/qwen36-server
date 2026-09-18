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
    int shown = 0;
    for (int m = 0; m < M && shown < 8; ++m) for (int n = 0; n < N && shown < 8; ++n) {
        float ref = 0.0f;
        for (int kk = 0; kk < K; ++kk) ref += (float)q[m*K+kk] * (float)k[n*K+kk];
        float d = out[m*N+n] - ref; if (d < 0) d = -d;
        if (d > 0.5f) { printf("  mismatch m=%d n=%d got=%.0f ref=%.0f\n", m, n, out[m*N+n], ref); shown++; }
    }
    return bad ? 2 : 0;
}
