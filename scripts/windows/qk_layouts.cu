// Печатаем раскладки фрагментов для f16- и s8-атомов в одном и том же каркасе
// (M=64, N=32, K=32, 4 варпа вдоль M).
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

__global__ void layouts_kernel() {
    __shared__ int8_t sQ8[M * K];
    __shared__ int8_t sK8[N * K];
    __shared__ __half sQf[M * K];
    __shared__ __half sKf[N * K];
    if (threadIdx.x == 0) {
        // f16
        auto tQf = make_tensor(make_smem_ptr(sQf), Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>{});
        auto tKf = make_tensor(make_smem_ptr(sKf), Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>{});
        using MmaF = decltype(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<Int<TW>, Int<1>, Int<1>>>{}));
        MmaF mf; auto thf = mf.get_thread_slice(threadIdx.x);
        auto af = thf.partition_fragment_A(tQf);
        auto bf = thf.partition_fragment_B(tKf);
        auto cf = partition_fragment_C(mf, Shape<Int<M>, Int<N>>{});
        printf("f16: A size=%d B size=%d C size=%d\n", (int)size(af), (int)size(bf), (int)size(cf));
        print("  f16 A layout: "); print(af.layout()); print("\n");
        print("  f16 B layout: "); print(bf.layout()); print("\n");
        print("  f16 C layout: "); print(cf.layout()); print("\n");
        // s8
        auto tQ8 = make_tensor(make_smem_ptr(sQ8), Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, Int<1>>>{});
        auto tK8 = make_tensor(make_smem_ptr(sK8), Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, Int<1>>>{});
        using MmaS8 = decltype(make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{}, Layout<Shape<Int<TW>, Int<1>, Int<1>>>{}));
        MmaS8 ms; auto ths = ms.get_thread_slice(threadIdx.x);
        auto as = ths.partition_fragment_A(tQ8);
        auto bs = ths.partition_fragment_B(tK8);
        auto cs = partition_fragment_C(ms, Shape<Int<M>, Int<N>>{});
        printf("s8 : A size=%d B size=%d C size=%d\n", (int)size(as), (int)size(bs), (int)size(cs));
        print("  s8 A layout: "); print(as.layout()); print("\n");
        print("  s8 B layout: "); print(bs.layout()); print("\n");
        print("  s8 C layout: "); print(cs.layout()); print("\n");
    }
}

int main() {
    layouts_kernel<<<1, 32 * TW>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 0;
}
