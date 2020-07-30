#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1024*1024
#define THREADS_PER_BLOCK 512

__global__ void multiply(float *a, float *b, int n);
void random_floats(float *x, int Num);
float* CPU_big_dot(float *a, float *b, int Num);
float* GPU_big_dot(float *A, float *B, int Num);
long long start_timer();
long long stop_timer(long long start_time, char *name);

int main(void) {
  float *a, *b; // host copies of a, b, c
  int size = N * sizeof(float);
  
  // Alloc space for host copies of a, b, c and setup input values
  a = (float *) malloc(size);
  random_floats(a, N);
  b = (float *) malloc(size);
  random_floats(b, N);

  float *result_cpu, *result_gpu;
  long long cpu_start, cpu_time, gpu_start, gpu_time;
  char cpu_task_name[] = "CPU time usage";
  char gpu_task_name[] = "GPU time usage";

  cpu_start = start_timer();
  result_cpu = CPU_big_dot(a, b, N);
  cpu_time = stop_timer(cpu_start, cpu_task_name);
  gpu_start = start_timer();
  result_gpu = GPU_big_dot(a, b, N);
  gpu_time = stop_timer(gpu_start, gpu_task_name);
  printf("\ncpu result: %f\n", *result_cpu);
  printf("gpu result: %f\n", *result_gpu);
  float diff = *result_cpu - *result_gpu;
  if (diff <= 1.0e-6)
    printf("difference between 2 results is: %f < 1.0e-6 ===> correct.\n", diff);
  else{
    printf("difference between 2 results is: %f > 1.0e-6 ===> incorrect.\nExit Now!\n", diff);
    exit(-1);
  }
  printf("\nCPU/GPU speedup: %f\n", 1.0 * cpu_time / gpu_time);
  free(a);
  free(b);
  free(result_cpu);
  free(result_gpu);
  return 0;
}
__global__ void multiply(float *a, float *b, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    a[index] = a[index] * b[index];
}

void random_floats(float *x, int Num)
{
  for (int i = 0; i < Num; i++)
  {
    x[i] = (float)rand() / RAND_MAX;
  }
}

float* CPU_big_dot(float *a, float *b, int Num)
{
  float *sum;
  sum = (float *) malloc(sizeof(float));
  (*sum) = 0;
  for (int i = 0; i < Num; i++)
  {
    (*sum) += a[i] * b[i];
  }
  return sum;
}

float* GPU_big_dot(float *A, float *B, int Num)
{
  float *sum;
  sum = (float *) malloc(sizeof(float));
  (*sum) = 0;
  float *d_A, *d_B; // device copies of A, B
  int size = Num * sizeof(float);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **) &d_A, size);
  cudaMalloc((void **) &d_B, size);

  // Copy inputs to device
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU with N threads 
  multiply<<<(Num + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_A, d_B, Num);

  // Copy result back to host
  cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < Num; i++) 
    (*sum) += A[i];

  // Cleanup
  cudaFree(d_A); 
  cudaFree(d_B);

  return sum;
}

long long start_timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec; 
}

long long stop_timer(long long start_time, char *name) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
  printf("%s: %.5f sec\n", name, ((float)(end_time-start_time))/(1000*1000)); 
  return end_time - start_time;
}
