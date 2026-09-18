// Какой раскладкой должен лежать K для s8-атома m16n8k32?
// Вариант A: тензор (N,K) со stride (K,1) — K-непрерывная (как в пуле).
// Вариант B: тензор (N,K) со stride (1,N) — N-непрерывная (K как колонки).
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

template <typename LK>
__global__ void atom_kernel(const int8_t* __restrict__ q, const int8_t* __restrict__ k,
                            float* __restrict__ out) {
    __shared__ int8_t sQ[M * K];
    __shared__ int8_t sK[N * K];
    for (int i = threadIdx.x; i < M * K; i += 32) sQ[i] = q[i];
    for (int i = threadIdx.x; i < N * K; i += 32) sK[i] = k[i];
    __syncthreads();
    auto tQ = make_tensor(make_smem_ptr(sQ), Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>{});
    auto tK = make_tensor(make_smem_ptr(sK), LK{});
    using TiledMmaS8 = decltype(make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{}, Layout<Shape<Int<1>, Int<1>, Int<1>>>{}));
    TiledMmaS8 mma8;
    auto thr = mma8.get_thread_slice(threadIdx.x);
    auto tAsQ = thr.partition_A(tQ);
    auto tArQ = thr.partition_fragment_A(tQ);
    auto tBsK = thr.partition_B(tK);
    auto tBrK = thr.partition_fragment_B(tK);
    auto acc32 = partition_fragment_C(mma8, Shape<Int<M>, Int<N>>{});
    clear(acc32);
    copy(tAsQ(_, _, 0), tArQ(_, _, 0));
    copy(tBsK(_, _, 0), tBrK(_, _, 0));
    gemm(mma8, tArQ(_, _, 0), tBrK(_, _, 0), acc32);
    auto gC = make_tensor(make_gmem_ptr(out), Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>{});
    auto accf = make_tensor<float>(acc32.layout());
    for (int i = 0; i < size(acc32); ++i) accf(i) = static_cast<float>(acc32(i));
    copy(accf, thr.partition_C(gC));
}

int main() {
    static int8_t q[M * K], kA[N * K], kB[N * K];
    static float oA[M * N], oB[M * N];
    for (int i = 0; i < M * K; ++i) q[i] = (int8_t)((i * 37 % 255) - 127);
    int8_t klog[N][K];
    for (int n = 0; n < N; ++n) for (int kk = 0; kk < K; ++kk)
        klog[n][kk] = (int8_t)(((n * K + kk) * 53 % 255) - 127);
    for (int n = 0; n < N; ++n) for (int kk = 0; kk < K; ++kk) {
        kA[n * K + kk] = klog[n][kk];   // K-непрерывная
        kB[kk * N + n] = klog[n][kk];   // N-непрерывная
    }
    int8_t *dq, *dkA, *dkB; float *doA, *doB;
    cudaMalloc(&dq, M * K); cudaMalloc(&dkA, N * K); cudaMalloc(&dkB, N * K);
    cudaMalloc(&doA, sizeof(float) * M * N); cudaMalloc(&doB, sizeof(float) * M * N);
    cudaMemcpy(dq, q, M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dkA, kA, N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dkB, kB, N * K, cudaMemcpyHostToDevice);
    using LA = Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>;
    using LB = Layout<Shape<Int<N>, Int<K>>, Stride<Int<1>, Int<N>>>;
    atom_kernel<LA><<<1, 32>>>(dq, dkA, doA);
    atom_kernel<LB><<<1, 32>>>(dq, dkB, doB);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(oA, doA, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(oB, doB, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    int badA = 0, badB = 0; float maxA = 0, maxB = 0;
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float ref = 0;
        for (int kk = 0; kk < K; ++kk) ref += (float)q[m * K + kk] * (float)klog[n][kk];
        float a = oA[m * N + n] - ref; if (a < 0) a = -a;
        float b = oB[m * N + n] - ref; if (b < 0) b = -b;
        if (a > 0.5f) badA++; if (a > maxA) maxA = a;
        if (b > 0.5f) badB++; if (b > maxB) maxB = b;
    }
    printf("A (K-непрерывная): несовпадений=%d/%d max=%.1f | B (N-непрерывная): несовпадений=%d/%d max=%.1f\n",
           badA, M * N, maxA, badB, M * N, maxB);
    return 0;
}
