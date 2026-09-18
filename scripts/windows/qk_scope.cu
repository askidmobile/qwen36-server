// Разложение многофрагментного случая на два фактора:
//   case 1: M=16, N=8, K=32, 1 варп, 1 k-блок  (базовый, уже проверен)
//   case 2: M=64, N=8, K=32, 4 варпа, 1 k-блок (проверяет tiling по M)
//   case 3: M=16, N=8, K=64, 1 варп, 2 k-блока (проверяет цикл по k)
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
constexpr int N = 8;

template <int TM, int TK, int TW>
__global__ void tiled_kernel(const int8_t* __restrict__ q, const int8_t* __restrict__ k,
                             float* __restrict__ out) {
    __shared__ int8_t sQ[TM * TK];
    __shared__ int8_t sK[N * TK];
    for (int i = threadIdx.x; i < TM * TK; i += blockDim.x) sQ[i] = q[i];
    for (int i = threadIdx.x; i < N * TK; i += blockDim.x) sK[i] = k[i];
    __syncthreads();
    auto tQ = make_tensor(make_smem_ptr(sQ), Layout<Shape<Int<TM>, Int<TK>>, Stride<Int<TK>, Int<1>>>{});
    auto tK = make_tensor(make_smem_ptr(sK), Layout<Shape<Int<N>, Int<TK>>, Stride<Int<TK>, Int<1>>>{});
    using TiledMmaS8 = decltype(make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},
                                              Layout<Shape<Int<TW>, Int<1>, Int<1>>>{}));
    TiledMmaS8 mma8;
    auto thr = mma8.get_thread_slice(threadIdx.x);
    auto tAsQ = thr.partition_A(tQ);
    auto tArQ = thr.partition_fragment_A(tQ);
    auto tBsK = thr.partition_B(tK);
    auto tBrK = thr.partition_fragment_B(tK);
    auto acc32 = partition_fragment_C(mma8, Shape<Int<TM>, Int<N>>{});
    clear(acc32);
    #pragma unroll
    for (int kb = 0; kb < TK / 32; ++kb) {
        copy(tAsQ(_, _, kb), tArQ(_, _, kb));
        copy(tBsK(_, _, kb), tBrK(_, _, kb));
        gemm(mma8, tArQ(_, _, kb), tBrK(_, _, kb), acc32);
    }
    auto gC = make_tensor(make_gmem_ptr(out), Layout<Shape<Int<TM>, Int<N>>, Stride<Int<N>, Int<1>>>{});
    auto accf = make_tensor<float>(acc32.layout());
    for (int i = 0; i < size(acc32); ++i) accf(i) = static_cast<float>(acc32(i));
    copy(accf, thr.partition_C(gC));
}

template <int TM, int TK, int TW>
static void run_case(const char* name) {
    int8_t *hq = new int8_t[TM * TK], *hk = new int8_t[N * TK];
    float* ho = new float[TM * N];
    for (int i = 0; i < TM * TK; ++i) hq[i] = (int8_t)((i * 37 % 255) - 127);
    for (int i = 0; i < N * TK; ++i) hk[i] = (int8_t)((i * 53 % 255) - 127);
    int8_t *dq, *dk; float* dout;
    cudaMalloc(&dq, TM * TK); cudaMalloc(&dk, N * TK); cudaMalloc(&dout, sizeof(float) * TM * N);
    cudaMemcpy(dq, hq, TM * TK, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk, N * TK, cudaMemcpyHostToDevice);
    tiled_kernel<TM, TK, TW><<<1, 32 * TW>>>(dq, dk, dout);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("%s: CUDA error %s\n", name, cudaGetErrorString(e)); return; }
    cudaMemcpy(ho, dout, sizeof(float) * TM * N, cudaMemcpyDeviceToHost);
    int bad = 0; float maxd = 0.0f;
    for (int m = 0; m < TM; ++m) for (int n = 0; n < N; ++n) {
        float ref = 0;
        for (int kk = 0; kk < TK; ++kk) ref += (float)hq[m * TK + kk] * (float)hk[n * TK + kk];
        float d = ho[m * N + n] - ref; if (d < 0) d = -d;
        if (d > 0.5f) bad++; if (d > maxd) maxd = d;
    }
    printf("%s: несовпадений=%d/%d max=%.1f\n", name, bad, TM * N, maxd);
    cudaFree(dq); cudaFree(dk); cudaFree(dout);
    delete[] hq; delete[] hk; delete[] ho;
}

int main() {
    run_case<16, 32, 1>("case1 M=16 K=32 1варп 1блок");
    run_case<64, 32, 4>("case2 M=64 K=32 4варпа 1блок");
    run_case<16, 64, 1>("case3 M=16 K=64 1варп 2блока");
    return 0;
}
