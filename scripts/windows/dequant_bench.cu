// Микробенчмарк декванта int8 -> fp16 для smem->smem пути FA2.
// Вопрос: сколько тактов стоит распаковка тайла 32x256 (8192 элемента) на CTA
// и сколько из них снимает векторная загрузка + half2-математика.
#include <cstdio>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int ROWS = 32;
constexpr int COLS = 256;      // 256 значений головы
constexpr int THREADS = 256;
constexpr int ELEMS = ROWS * COLS;

// V0: текущий цикл из utils.h (скалярно, с масштабом на строку, как в ядре)
__global__ void k_v0(const int8_t* __restrict__ src, const half* __restrict__ scale, half* __restrict__ dst, int rounds, unsigned long long* out) {
    __shared__ int8_t s8[ELEMS];
    __shared__ __half sf[ROWS];
    __shared__ __half sd[ELEMS];
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) s8[i] = src[i];
    if (threadIdx.x < ROWS) sf[threadIdx.x] = scale[threadIdx.x];
    __syncthreads();
    unsigned long long t0 = clock64();
    for (int r = 0; r < rounds; ++r) {
        const int per = ELEMS / THREADS;              // 32 элемента на поток
        for (int e = 0; e < per; ++e) {
            const int idx = e * THREADS + threadIdx.x; // как в ядре: сгруппировано по строкам
            const int row = idx / COLS;
            const int col = idx % COLS;
            const float s = __half2float(sf[row]);
            sd[idx] = __float2half(static_cast<float>(s8[idx]) * s);
        }
        __syncthreads();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[0] = t1 - t0;
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) dst[i] = sd[i];
}

// V1: векторная загрузка uint4 (16 байт) + half2-масштаб
__global__ void k_v1(const int8_t* __restrict__ src, const half* __restrict__ scale, half* __restrict__ dst, int rounds, unsigned long long* out) {
    __shared__ int8_t s8[ELEMS];
    __shared__ __half sf[ROWS];
    __shared__ __half sd[ELEMS];
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) s8[i] = src[i];
    if (threadIdx.x < ROWS) sf[threadIdx.x] = scale[threadIdx.x];
    __syncthreads();
    unsigned long long t0 = clock64();
    for (int r = 0; r < rounds; ++r) {
        // 8192 элемента / 256 потоков = 32 элемента = 2 вектора по 16 байт
        #pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int base = (v * THREADS + threadIdx.x) * 16;
            const int row = base / COLS;
            const half2 s2 = __half2half2(sf[row]);
            const uchar4* p4 = reinterpret_cast<const uchar4*>(s8 + base);
            __half2* d2 = reinterpret_cast<__half2*>(sd + base);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uchar4 u = p4[q];
                const int a = static_cast<int>(static_cast<int8_t>(u.x));
                const int b = static_cast<int>(static_cast<int8_t>(u.y));
                const int c = static_cast<int>(static_cast<int8_t>(u.z));
                const int d = static_cast<int>(static_cast<int8_t>(u.w));
                const __half2 h0 = __hmul2(__floats2half2_rn((float)a, (float)b), s2);
                const __half2 h1 = __hmul2(__floats2half2_rn((float)c, (float)d), s2);
                d2[2 * q] = h0;
                d2[2 * q + 1] = h1;
            }
        }
        __syncthreads();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[0] = t1 - t0;
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) dst[i] = sd[i];
}

// V2: то же, но масштаб НЕ применяется (его сворачивают в score/P отдельно)
__global__ void k_v2(const int8_t* __restrict__ src, const half* __restrict__ scale, half* __restrict__ dst, int rounds, unsigned long long* out) {
    __shared__ int8_t s8[ELEMS];
    __shared__ __half sd[ELEMS];
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) s8[i] = src[i];
    __syncthreads();
    unsigned long long t0 = clock64();
    for (int r = 0; r < rounds; ++r) {
        #pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int base = (v * THREADS + threadIdx.x) * 16;
            const uchar4* p4 = reinterpret_cast<const uchar4*>(s8 + base);
            __half2* d2 = reinterpret_cast<__half2*>(sd + base);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uchar4 u = p4[q];
                d2[2 * q] = __floats2half2_rn((float)(int)(int8_t)u.x, (float)(int)(int8_t)u.y);
                d2[2 * q + 1] = __floats2half2_rn((float)(int)(int8_t)u.z, (float)(int)(int8_t)u.w);
            }
        }
        __syncthreads();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[0] = t1 - t0;
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) dst[i] = sd[i];
}

// V3: векторно, но через int16-пары и __short22half2_rn (1 cvt на пару)
__device__ __forceinline__ __half2 p8_2h2(int a, int b) {
    // cvt.rn.f16.s32 на каждый элемент + сборка пары
    return __halves2half2(__int2half_rn(a), __int2half_rn(b));
}

__global__ void k_v3(const int8_t* __restrict__ src, const half* __restrict__ scale, half* __restrict__ dst, int rounds, unsigned long long* out) {
    __shared__ int8_t s8[ELEMS];
    __shared__ __half sf[ROWS];
    __shared__ __half sd[ELEMS];
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) s8[i] = src[i];
    if (threadIdx.x < ROWS) sf[threadIdx.x] = scale[threadIdx.x];
    __syncthreads();
    unsigned long long t0 = clock64();
    for (int r = 0; r < rounds; ++r) {
        #pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int base = (v * THREADS + threadIdx.x) * 16;
            const int row = base / COLS;
            const half2 s2 = __half2half2(sf[row]);
            const uchar4* p4 = reinterpret_cast<const uchar4*>(s8 + base);
            __half2* d2 = reinterpret_cast<__half2*>(sd + base);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uchar4 u = p4[q];
                d2[2 * q] = __hmul2(p8_2h2((int)(int8_t)u.x, (int)(int8_t)u.y), s2);
                d2[2 * q + 1] = __hmul2(p8_2h2((int)(int8_t)u.z, (int)(int8_t)u.w), s2);
            }
        }
        __syncthreads();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[0] = t1 - t0;
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) dst[i] = sd[i];
}

template <typename F>
void run(const char* name, F kernel, const int8_t* src, const half* sc, half* dst, int rounds, unsigned long long* out) {
    kernel<<<1, THREADS>>>(src, sc, dst, rounds, out);
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("%s: CUDA error %s\n", name, cudaGetErrorString(err)); return; }
    unsigned long long cyc = 0;
    cudaMemcpy(&cyc, out, sizeof(cyc), cudaMemcpyDeviceToHost);
    const double per_tile = double(cyc) / rounds;
    printf("%-6s cycles/tile(32x256) = %10.0f   ~%.2f цикл/элемент\n", name, per_tile, per_tile / ELEMS);
}

int main() {
    int8_t* src; half* sc; half* dst; unsigned long long* out;
    cudaMalloc(&src, ELEMS); cudaMalloc(&sc, ROWS * sizeof(half)); cudaMalloc(&dst, ELEMS * sizeof(half)); cudaMalloc(&out, sizeof(unsigned long long));
    int8_t host[ELEMS]; for (int i = 0; i < ELEMS; ++i) host[i] = (int8_t)((i % 255) - 127);
    cudaMemcpy(src, host, ELEMS, cudaMemcpyHostToDevice);
    int rounds = 2000;
    run("V0", k_v0, src, sc, dst, rounds, out);
    run("V1", k_v1, src, sc, dst, rounds, out);
    run("V2", k_v2, src, sc, dst, rounds, out);
    run("V3", k_v3, src, sc, dst, rounds, out);
    return 0;
}
