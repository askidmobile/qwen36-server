// Спайк: QK-произведение на int8-тензорных ядрах (m16n8k32 s8) из обычных
// row-major smem-тайлов. Проверяем, что:
//   * фрагменты A/B собираются прямо из [n, k]-раскладки (k подряд);
//   * аккумулятор s32 совпадает с эталоном (целочисленно точно);
//   * сколько тактов стоит такая сборка по сравнению с f16-путём.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm80.hpp>

using namespace cute;

constexpr int M = 64;   // строки запроса (kBlockM)
constexpr int N = 32;   // ключей в тайле (kBlockN)
constexpr int K = 256;  // head_dim
constexpr int THREADS = 128;

// 4 варпа вдоль M (как в FA-ядре: каждый варп берёт свои 16 строк тайла)
using TiledMmaS8 = decltype(make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{}, Layout<Shape<_4, _1, _1>>{}));

__global__ void qk_s8_kernel(const int8_t* __restrict__ q, const int8_t* __restrict__ k,
                             int32_t* __restrict__ out, unsigned long long* cycles) {
    __shared__ int8_t sQ[M * K];
    __shared__ int8_t sK[N * K];
    for (int i = threadIdx.x; i < M * K; i += THREADS) sQ[i] = q[i];
    for (int i = threadIdx.x; i < N * K; i += THREADS) sK[i] = k[i];
    __syncthreads();

    TiledMmaS8 tiled_mma;
    auto thr = tiled_mma.get_thread_slice(threadIdx.x);
    auto tQ = make_tensor(make_smem_ptr(sQ), Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, _1>>{});
    auto tK = make_tensor(make_smem_ptr(sK), Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, _1>>{});
    // Смем-партиции (источник) и регистровые фрагменты (приёмник) — как в FA-ядре:
    // копируем smem -> registers, затем gemm по регистрам.
    auto tAsQ = thr.partition_A(tQ);
    auto tBsK = thr.partition_B(tK);
    auto tArA = thr.partition_fragment_A(tQ);
    auto tBrB = thr.partition_fragment_B(tK);
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
    clear(acc);

    unsigned long long t0 = clock64();
    #pragma unroll
    for (int kb = 0; kb < K / 32; ++kb) {
        copy(tAsQ(_, _, kb), tArA(_, _, kb));
        copy(tBsK(_, _, kb), tBrB(_, _, kb));
        gemm(tiled_mma, tArA(_, _, kb), tBrB(_, _, kb), acc);
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) cycles[0] = t1 - t0;

    // Выгрузка аккумулятора (s32) в gmem построчно.
    auto thr_c = tiled_mma.get_thread_slice(threadIdx.x);
    auto gC = make_tensor(make_gmem_ptr(out), Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, _1>>{});
    auto tCgC = thr_c.partition_C(gC);
    copy(acc, tCgC);
}


static void run_probe(const char* name, const int8_t* q, const int8_t* k, int32_t* out) {
    int8_t *dq, *dk; int32_t* dout; unsigned long long* cyc;
    cudaMalloc(&dq, M * K); cudaMalloc(&dk, N * K);
    cudaMalloc(&dout, M * N * sizeof(int32_t)); cudaMalloc(&cyc, sizeof(unsigned long long));
    cudaMemcpy(dq, q, M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k, N * K, cudaMemcpyHostToDevice);
    qk_s8_kernel<<<1, THREADS>>>(dq, dk, dout, cyc);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("%s: CUDA error %s\n", name, cudaGetErrorString(err)); }
    cudaMemcpy(out, dout, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(dout); cudaFree(cyc);
}

int main() {
    static int8_t q[M * K], k[N * K];
    static int32_t out[M * N];
    // Проба A: Q=1, K=1 -> ожидаем 256 везде
    for (int i = 0; i < M * K; ++i) q[i] = 1;
    for (int i = 0; i < N * K; ++i) k[i] = 1;
    run_probe("A", q, k, out);
    printf("A: все 1 -> ожидаем 256: out[0]=%d out[1*N+0]=%d out[63*N+31]=%d\n", out[0], out[1*N], out[63*N+31]);
    // Проба B: Q[m][k] = (m==0), K=1 -> 256 в строке 0
    for (int i = 0; i < M * K; ++i) q[i] = 0;
    for (int kk = 0; kk < K; ++kk) q[0 * K + kk] = 1;
    run_probe("B", q, k, out);
    printf("B: строка 0 -> 256, остальные 0: out[0]=%d out[0*N+31]=%d out[1*N+0]=%d out[63*N+0]=%d\n",
           out[0], out[0*N+31], out[1*N+0], out[63*N+0]);
    // Проба C: K[n][k] = (n==0), Q=1 -> 256 в колонке 0
    for (int i = 0; i < N * K; ++i) k[i] = 0;
    for (int kk = 0; kk < K; ++kk) k[0 * K + kk] = 1;
    for (int i = 0; i < M * K; ++i) q[i] = 1;
    run_probe("C", q, k, out);
    printf("C: колонка 0 -> 256: out[0]=%d out[63*N+0]=%d out[0*N+1]=%d out[63*N+1]=%d\n",
           out[0], out[63*N+0], out[0*N+1], out[63*N+1]);
    // Проба D: Q[m][k] = (k==0)?1:0, K=1 -> 1 везде
    for (int i = 0; i < M * K; ++i) q[i] = 0;
    for (int m = 0; m < M; ++m) q[m * K + 0] = 1;
    for (int i = 0; i < N * K; ++i) k[i] = 1;
    run_probe("D", q, k, out);
    printf("D: Q=e0 по k, K=1 -> 1 везде: out[0]=%d out[10*N+5]=%d out[63*N+31]=%d\n", out[0], out[10*N+5], out[63*N+31]);
    return 0;
}
