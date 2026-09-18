// Тот же каркас, но на f16-атоме: M=64, N=32, K=32, 4 варпа.
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
constexpr int M = 64, N = 32, K = 32, TW = 4;

__global__ void f16_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                           float* __restrict__ out) {
    __shared__ __half sQ[M * K];
    __shared__ __half sK[N * K];
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) sQ[i] = q[i];
    for (int i = threadIdx.x; i < N * K; i += blockDim.x) sK[i] = k[i];
    __syncthreads();
    auto tQ = make_tensor(make_smem_ptr(sQ), Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>{});
    auto tK = make_tensor(make_smem_ptr(sK), Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>{});
    using TiledMma = decltype(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                            Layout<Shape<Int<TW>, Int<1>, Int<1>>>{}));
    TiledMma mma;
    auto thr = mma.get_thread_slice(threadIdx.x);
    auto tArQ = thr.partition_fragment_A(tQ);
    auto tBrK = thr.partition_fragment_B(tK);
    auto acc = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
    clear(acc);
    copy(thr.partition_A(tQ), tArQ);
    copy(thr.partition_B(tK), tBrK);
    gemm(mma, tArQ, tBrK, acc);
    auto gC = make_tensor(make_gmem_ptr(out), Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>{});
    copy(acc, thr.partition_C(gC));
}

int main() {
    static __half q[M * K], k[N * K]; static float o[M * N];
    static float fq[M * K], fk[N * K];
    for (int i = 0; i < M * K; ++i) { fq[i] = (float)((i % 7) - 3); q[i] = __float2half(fq[i]); }
    for (int i = 0; i < N * K; ++i) { fk[i] = (float)((i % 5) - 2); k[i] = __float2half(fk[i]); }
    __half *dq, *dk; float* dout;
    cudaMalloc(&dq, sizeof(__half) * M * K); cudaMalloc(&dk, sizeof(__half) * N * K);
    cudaMalloc(&dout, sizeof(float) * M * N);
    cudaMemcpy(dq, q, sizeof(__half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k, sizeof(__half) * N * K, cudaMemcpyHostToDevice);
    f16_kernel<<<1, 32 * TW>>>(dq, dk, dout);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(o, dout, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    int bad = 0; float maxd = 0;
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float ref = 0; for (int kk = 0; kk < K; ++kk) ref += fq[m*K+kk] * fk[n*K+kk];
        float d = o[m*N+n] - ref; if (d < 0) d = -d;
        if (d > 0.01f) bad++; if (d > maxd) maxd = d;
    }
    printf("f16-аналог (M=64,N=32,K=32,4варпа): несовпадений=%d/%d max=%.3f\n", bad, M*N, maxd);
    printf("  o[0]=%.1f o[1]=%.1f o[31]=%.1f o[32]=%.1f\n", o[0], o[1], o[31], o[32]);
    return 0;
}
