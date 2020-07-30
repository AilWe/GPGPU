#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N (1<<24)
#define THREADS_PER_BLOCK 512
#define BLOCK_NUM (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK // 1<<15 block 

void random_floats(float *x, int Num);
__global__ void kernel1(float *a, float *b, float *out, int n);
__global__ void kernel2WithAtomicOp(float *a, float *b, float *out, int n);

int main(void){
  printf("N: %d, block num: %d\n", N, BLOCK_NUM);
  // initialization
  float *a, *b, *reduce, *sum;
  a = (float *)malloc(N * sizeof(float));
  random_floats(a, N);
  b = (float *)malloc(N * sizeof(float));
  random_floats(b, N);
  reduce = (float *)malloc(BLOCK_NUM * sizeof(float));
  sum = (float *)malloc(sizeof(float));
  *sum = 0;
  // cudaEvent initialization
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // create space on gpu side
  float *da, *db, *dreduce, *dsum;
  cudaMalloc((void **)&da, N * sizeof(float));
  cudaMalloc((void **)&db, N * sizeof(float));
  cudaMalloc((void **)&dreduce, BLOCK_NUM * sizeof(float));
  cudaMalloc((void **)&dsum, sizeof(float));

  // copy from cpu to gpu
  cudaMemcpy(da, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dsum, sum, sizeof(float), cudaMemcpyHostToDevice);

  // kernel1: shared memory + parallel reduction
  cudaEventRecord(start);
  kernel1<<<BLOCK_NUM, THREADS_PER_BLOCK>>>(da, db, dreduce, N);
  cudaEventRecord(stop);
  //// copy back to cpu
  cudaMemcpy(reduce, dreduce, BLOCK_NUM*sizeof(float), cudaMemcpyDeviceToHost);
  //// add up all elements in reduce
  *sum = 0;
  for (int i = 0; i < BLOCK_NUM; i++)
    *sum += reduce[i];
  printf("result from Kernel1 with sum on CPU side: %f\n", *sum);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("kernel 1 execution time: %f milliseconds\n\n", milliseconds);


  // kernel2: shared memory + parallel reduction + atomic operation
  cudaEventRecord(start);
  kernel2WithAtomicOp<<<BLOCK_NUM, THREADS_PER_BLOCK>>>(da, db, dsum, N);
  cudaEventRecord(stop);
  //// copy back to cpu
  cudaMemcpy(sum, dsum, sizeof(float), cudaMemcpyDeviceToHost);
  printf("result from Kernel2 with sum on GPU side: %f\n", *sum);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("kernel 2 execution time: %f milliseconds\n\n", milliseconds);

  free(a);  free(b);  free(reduce);  free(sum);
  cudaFree(da);  cudaFree(db);  cudaFree(dreduce);  cudaFree(dsum);
  cudaEventDestroy(start);  cudaEventDestroy(stop);
  return 0;
}

void random_floats(float *x, int Num)
{
  for (int i = 0; i < Num; i++)
  {
    x[i] = (float)rand() / RAND_MAX;
  }
}

__global__ void kernel1(float *a, float *b, float *out, int n){
  __shared__ float sdata[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  sdata[tid] = 0.0;
  if (index < n)
    sdata[tid] = a[index] * b[index];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2){
    int ix = 2 * s * tid;
    if (ix < blockDim.x)
      sdata[ix] += sdata[ix+s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0]; 
}

__global__ void kernel2WithAtomicOp(float *a, float *b, float *out, int n){
  __shared__ float sdata[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  sdata[tid] = 0.0;
  if (index < n)
    sdata[tid] = a[index] * b[index];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2){
    int ix = 2 * s * tid;
    if (ix < blockDim.x)
      sdata[ix] += sdata[ix+s];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(out, sdata[0]);
}
