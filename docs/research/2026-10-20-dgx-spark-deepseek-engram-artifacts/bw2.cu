#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
__global__ void rd(const uint4 *__restrict__ p, unsigned long long n4, float *sink) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long step = (unsigned long long)gridDim.x * blockDim.x;
    uint4 a = make_uint4(0,0,0,0);
    for (; i < n4; i += step) { uint4 v = __ldg(&p[i]); a.x^=v.x; a.y^=v.y; a.z^=v.z; a.w^=v.w; }
    if ((a.x^a.y^a.z^a.w)==0xdeadbeefu) *sink=1.f;
}
// 4 independent streams per thread: more MLP per warp
__global__ void rd4(const uint4 *__restrict__ p, unsigned long long n4, float *sink) {
    const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long step = (unsigned long long)gridDim.x * blockDim.x;
    uint4 a[4];
    #pragma unroll
    for (int j=0;j<4;j++) a[j]=make_uint4(0,0,0,0);
    unsigned long long i = tid;
    for (; i + 3*step < n4; i += 4*step) {
        #pragma unroll
        for (int j=0;j<4;j++) { uint4 v=__ldg(&p[i+j*step]); a[j].x^=v.x; a[j].y^=v.y; a[j].z^=v.z; a[j].w^=v.w; }
    }
    for (; i < n4; i += step) { uint4 v=__ldg(&p[i]); a[0].x^=v.x; }
    unsigned x=0; 
    #pragma unroll
    for (int j=0;j<4;j++) x^=a[j].x^a[j].y^a[j].z^a[j].w;
    if (x==0xdeadbeefu) *sink=1.f;
}
int main(int argc,char**argv){
    unsigned long long gib = argc>1? atoi(argv[1]) : 6;
    unsigned long long bytes = gib<<30;
    uint4 *p=nullptr; cudaMalloc(&p,bytes);
    cudaMemset(p,1,bytes);
    float *sink; cudaMalloc(&sink,4); cudaMemset(sink,0,4);
    int nsm=48; cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0);
    printf("size=%llu GiB SMs=%d\n",gib,nsm);
    unsigned long long n4=bytes/16;
    for (int kind=0;kind<2;kind++)
      for (int bpsm : {1,2,4})
        for (int w : {16,32}) {
          int grid=nsm*bpsm, thr=w*32;
          cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
          auto launch=[&]{ if(kind==0) rd<<<grid,thr>>>(p,n4,sink); else rd4<<<grid,thr>>>(p,n4,sink); };
          launch(); cudaDeviceSynchronize();
          cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
          float ms=0; cudaEventElapsedTime(&ms,a,b);
          printf("%s bpsm=%d warps=%2d  %.1f ms  %.1f GB/s\n", kind?"multi4":"single", bpsm,w,ms,(double)bytes/(ms*1e6));
        }
    return 0;
}
