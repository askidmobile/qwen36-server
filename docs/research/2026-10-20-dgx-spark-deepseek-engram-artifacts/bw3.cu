// Real decode access pattern: random expert-sized spans inside a big blob,
// plus a dense matrix read. Measures achievable GB/s, not peak.
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
__global__ void rd_span(const uint4 *__restrict__ p, unsigned long long n4, float *sink) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long step = (unsigned long long)gridDim.x * blockDim.x;
    uint4 a = make_uint4(0,0,0,0);
    for (; i < n4; i += step) { uint4 v = __ldg(&p[i]); a.x^=v.x; a.y^=v.y; a.z^=v.z; a.w^=v.w; }
    if ((a.x^a.y^a.z^a.w)==0xdeadbeefu) *sink=1.f;
}
int main(int argc,char**argv){
    const unsigned long long blob = 2770ull<<20;   // one layer's VQ blob
    const unsigned long long span = (7ull<<20);    // one expert
    const int nexperts = 6;
    unsigned char *p=nullptr; cudaMalloc(&p,blob); cudaMemset(p,1,blob);
    float *sink; cudaMalloc(&sink,4); cudaMemset(sink,0,4);
    int nsm=48; cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0);
    const unsigned long long nspan = blob/span;
    srand(7);
    for (int bpsm : {1,2,4}) for (int w : {8,16,32}) {
        int grid=nsm*bpsm, thr=w*32;
        // warm
        for (int e=0;e<nexperts;e++) rd_span<<<grid,thr>>>(reinterpret_cast<uint4*>(p+e*span),(span)/16,sink);
        cudaDeviceSynchronize();
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        // 20 rounds of 6 random experts = the decode's per-layer work
        unsigned long long offs[20][6];
        for(int r=0;r<20;r++) for(int e=0;e<nexperts;e++) offs[r][e]=(rand()%nspan)*span;
        cudaEventRecord(a);
        for(int r=0;r<20;r++) for(int e=0;e<nexperts;e++)
            rd_span<<<grid,thr>>>(reinterpret_cast<uint4*>(p+offs[r][e]),span/16,sink);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        double bytes=(double)20*nexperts*span;
        printf("random-span bpsm=%d warps=%2d  %.1f ms  %.1f GB/s\n",bpsm,w,ms,bytes/(ms*1e6));
    }
    // dense read of the same volume for comparison
    {
        unsigned long long bytes=(unsigned long long)20*nexperts*span;
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        rd_span<<<nsm,1024>>>(reinterpret_cast<uint4*>(p),bytes/16,sink);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        printf("dense same-bytes     %.1f ms  %.1f GB/s\n",ms,bytes/(ms*1e6));
    }
    return 0;
}
