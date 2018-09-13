
#include <stdio.h>
#include <cuda.h>

//#include <cudaMalloc.h>

__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main(void)
{
    int a, b, c;
    int *pa, *pb, *pc;

    int size = sizeof(int);

    cudaMalloc((void **)&pa, size);
    cudaMalloc((void **)&pb, size);
    cudaMalloc((void **)&pc, size);

    a = 7;
    b = 8;

    cudaMemcpy(pa, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(pb, &b, size, cudaMemcpyHostToDevice);

    add<<<1,1>>>(pa, pb, pc);

    cudaMemcpy(&c, pc, size, cudaMemcpyDeviceToHost);
    cudaFree(pa);
    cudaFree(pb);
    cudaFree(pc);

    printf("GPU computed value of c (a+b): %d\n", c);

    return 0;
}
