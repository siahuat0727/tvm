#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
// #include <stdint.h>
#include <stdio.h>
// #include <thread>

using namespace std;
using clock_value_t = long long;

extern "C" static __device__ __inline__ uint __mysmid() {
    uint smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

extern "C" __device__ void sleepForever()
{
    // clock_value_t sleep_cycles = 800000000000LL;  // TODO can we sleep forever?
    clock_value_t sleep_cycles = 100000000000LL;
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do {
        cycles_elapsed = clock64() - start;
    } while (cycles_elapsed < sleep_cycles);

    printf("Never reach here!");
}

extern "C" __device__ void smSleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do {
        cycles_elapsed = clock64() - start;
    } while (cycles_elapsed < sleep_cycles);
}

extern "C" __global__ void sleepKernel(int target_cores_num) {
    uint smid = __mysmid();
    if (smid >= target_cores_num) {
        printf("My SM ID is %d, sleep forever\n", smid);
        sleepForever();
    } else {
        printf("My SM ID is %d, take a snap about 2 s\n", smid);
        smSleep(5000000000LL);  // TODO can convert to seconds?
    }
    printf("My SM ID is %d, wake up!\n", smid);
}
