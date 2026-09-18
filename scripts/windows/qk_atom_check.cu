// Минимальная проверка: ровно один атом m16n8k32, один варп.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
using namespace cute;
constexpr int M = 16, N = 8, K = 32;
__global__ void atom_kernel(const int8_t* q, const int8_t* k, const __half* qs, const __half* ks,
                            float* out_copy, float* out_coord) {
    __shared__ int8_t sQ[M * K]; __shared__ int8_t sK[N * K];
    if (threadIdx.x < M * K) sQ[threadIdx.x] = q[threadIdx.x];
    if (threadIdx.x < N * K) sK[threadIdx.x] = k[threadIdx.x];
    __syncthreads();
    using LQ = Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>;
    using LK = Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>;
    using LQS = Layout<Shape<Int<M>, Int<1>>, Stride<Int<1>, Int<1>>>;
    using LKS = Layout<Shape<Int<N>>, Stride<Int<1>>>;
    using LC = Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>;
    auto tQ = make_tensor(make_smem_ptr(sQ), LQ{});
    auto tK = make_tensor(make_smem_ptr(sK), LK{});
    auto gQS = make_tensor(make_gmem_ptr(qs), LQS{});
    auto gKS = make_tensor(make_gmem_ptr(ks), LKS{});
    using TiledMmaS8 = decltype(make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{}, Layout<Shape<Int<1>, Int<1>, Int<1>>>{}));
    TiledMmaS8 mma8;
    auto thr = mma8.get_thread_slice(threadIdx.x);
    Tensor acc = partition_fragment_C(mma8, Shape<Int<M>, Int<N>>{});
    flash::qk_int8_scores<M, N, K, 1>(tQ, gQS, tK, gKS, acc, threadIdx.x, 0, 0, M, 1.0f);
    auto gC = make_tensor(make_gmem_ptr(out_copy), LC{});
    copy(acc, thr.partition_C(gC));
    auto tScS = thr.partition_C(make_identity_tensor(Shape<Int<M>, Int<N>>{}));
    for (int i = 0; i < size(acc); ++i) {
        auto rc = tScS(i);
        out_coord[get<0>(rc) * N + get<1>(rc)] = acc(i);
    }
}
int main() {
    static int8_t q[M * K], k[N * K]; static __half qs[M], ks[N];
    static float oc[M * N], od[M * N];
    for (int i = 0; i < M * K; ++i) q[i] = (int8_t)((i * 37 % 255) - 127);
    for (int i = 0; i < N * K; ++i) k[i] = (int8_t)((i * 53 % 255) - 127);
    for (int i = 0; i < M; ++i) qs[i] = __float2half(1.0f);
    for (int i = 0; i < N; ++i) ks[i] = __float2half(1.0f);
    int8_t *dq, *dk; __half *dqs, *dks; float *dc, *dd;
    cudaMalloc(&dq, M * K); cudaMalloc(&dk, N * K);
    cudaMalloc(&dqs, sizeof(__half) * M); cudaMalloc(&dks, sizeof(__half) * N);
    cudaMalloc(&dc, sizeof(float) * M * N); cudaMalloc(&dd, sizeof(float) * M * N);
    cudaMemcpy(dq, q, M * K, cudaMemcpyHostToDevice); cudaMemcpy(dk, k, N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dqs, qs, sizeof(__half) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dks, ks, sizeof(__half) * N, cudaMemcpyHostToDevice);
    cudaMemset(dc, 0, sizeof(float) * M * N);
    atom_kernel<<<1, 32>>>(dq, dk, dqs, dks, dc, dd);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(oc, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(od, dd, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    int badc = 0, badd = 0; float maxc = 0, maxd = 0;
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float ref = 0; for (int kk = 0; kk < K; ++kk) ref += (float)q[m*K+kk] * (float)k[n*K+kk];
        float dc_ = oc[m*N+n] - ref; if (dc_ < 0) dc_ = -dc_;
        float dd_ = od[m*N+n] - ref; if (dd_ < 0) dd_ = -dd_;
        if (dc_ > 0.5f) badc++; if (dc_ > maxc) maxc = dc_;
        if (dd_ > 0.5f) badd++; if (dd_ > maxd) maxd = dd_;
    }
    printf("atom copy-write: несовпадений=%d/%d max=%.1f | coord-write: несовпадений=%d/%d max=%.1f\n",
           badc, M*N, maxc, badd, M*N, maxd);
    return 0;
}
