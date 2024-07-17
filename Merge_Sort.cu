#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/merge.h>

// Define constants
#define NUM_SETS 100000
#define DSIZE 100
typedef int mytype;

// Macro for ascending sorted data comparison
#define cmp(X,Y) ((X)<(Y))
#define THREADS_PER_BLOCK 512 // Threads per block
#define BLOCKS_PER_GRID 128 // Blocks per grid

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL // Microseconds per second

// Function to get current time in microseconds
long long time_usec(unsigned long long start){
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

// Merge function for sorted arrays on host and device
template <typename T>
__host__ __device__ void merge_arrays(const T * __restrict__  arr1, const T * __restrict__ arr2, T * __restrict__ result, const unsigned len_arr1, const unsigned len_arr2, const unsigned stride_arr1 = 1, const unsigned stride_arr2 = 1, const unsigned stride_result = 1){

  unsigned len_result = len_arr1 + len_arr2;
  unsigned index_result = 0;
  unsigned index_arr1 = 0;
  unsigned index_arr2 = 0;
  unsigned finished_arr1 = (len_arr2 == 0);
  unsigned finished_arr2 = (len_arr1 == 0);
  T next_arr1 = arr1[0];
  T next_arr2 = arr2[0];
  
  while (index_result < len_result){
    if (finished_arr1) {
      result[stride_result * index_result++] = next_arr1;
      index_arr1++;
      next_arr1 = arr1[stride_arr1 * index_arr1];
    }
    else if (finished_arr2) {
      result[stride_result * index_result++] = next_arr2;
      index_arr2++;
      next_arr2 = arr2[stride_arr2 * index_arr2];
    }
    else if (cmp(next_arr1, next_arr2)) {
      result[stride_result * index_result++] = next_arr1;
      index_arr1++;
      if (index_arr1 == len_arr1) finished_arr2++;
      else next_arr1 = arr1[stride_arr1 * index_arr1];
    }
    else {
      result[stride_result * index_result++] = next_arr2;
      index_arr2++;
      if (index_arr2 == len_arr2) finished_arr1++;
      else next_arr2 = arr2[stride_arr2 * index_arr2];
    }
  }
}

// Kernel function to perform row-major merge test on device
template <typename T>
__global__ void row_major_merge(const T * __restrict__  arr1, const T * __restrict__ arr2, T * __restrict__  result, int num_arrays, int array_length){

  int index = threadIdx.x + blockDim.x * blockIdx.x;

  while (index < num_arrays){
    int selected_index = index * array_length;
    merge_arrays(arr1 + selected_index, arr2 + selected_index, result + (2 * selected_index), array_length, array_length);
    index += blockDim.x * gridDim.x;
  }
}

// Kernel function to perform column-major merge test on device
template <typename T>
__global__ void column_major_merge(const T * __restrict__ arr1, const T * __restrict__ arr2, T * __restrict__ result, int num_arrays, int array_length, int stride_arr1, int stride_arr2, int stride_result){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  while (index < num_arrays){
    merge_arrays(arr1 + index, arr2 + index, result + index, array_length, array_length, stride_arr1, stride_arr2, stride_result);
    index += blockDim.x * gridDim.x;
  }
}

// Function to validate row-major merge results
template <typename T>
int validate_row_major(T *arr1, T *arr2, T *result, int num_arrays, int array_length){

  T *validation_array = (T *)malloc(2 * array_length * sizeof(T));
  for (int i = 0; i < num_arrays; i++){
    thrust::merge(arr1 + (i * array_length), arr1 + ((i + 1) * array_length), arr2 + (i * array_length), arr2 + ((i + 1) * array_length), validation_array);
#ifndef TIMING
    for (int j = 0; j < array_length * 2; j++)
      if (validation_array[j] != result[(i * 2 * array_length) + j]) {
        printf("row-major mismatch i: %d, j: %d, was: %d, should be: %d\n", i, j, result[(i * 2 * array_length) + j], validation_array[j]);
        return 0;
      }
#endif
  }
  return 1;
}

// Function to validate column-major merge results
template <typename T>
int validate_column_major(const T *result1, const T *result2, int num_arrays, int array_length){
  for (int i = 0; i < num_arrays; i++)
    for (int j = 0; j < 2 * array_length; j++)
      if (result1[i * (2 * array_length) + j] != result2[j * (num_arrays) + i]) {
        printf("column-major mismatch i: %d, j: %d, was: %d, should be: %d\n", i, j, result2[j * (num_arrays) + i], result1[i * (2 * array_length) + j]);
        return 0;
      }
  return 1;
}

// Main function
int main(){
  // Allocate host and device memory
  mytype *host_array1, *host_array2, *host_result, *device_array1, *device_array2, *device_result;
  host_array1 = (mytype *)malloc(DSIZE * NUM_SETS * sizeof(mytype));
  host_array2 = (mytype *)malloc(DSIZE * NUM_SETS * sizeof(mytype));
  host_result = (mytype *)malloc(DSIZE * NUM_SETS * sizeof(mytype) * 2);
  cudaMalloc(&device_array1, (DSIZE * NUM_SETS + 1) * sizeof(mytype));
  cudaMalloc(&device_array2, (DSIZE * NUM_SETS + 1) * sizeof(mytype));
  cudaMalloc(&device_result, DSIZE * NUM_SETS * sizeof(mytype) * 2);

  // Test "row-major" storage
  for (int i = 0; i < DSIZE * NUM_SETS; i++){
    host_array1[i] = rand();
    host_array2[i] = rand();
  }
  thrust::sort(host_array1, host_array1 + DSIZE * NUM_SETS);
  thrust::sort(host_array2, host_array2 + DSIZE * NUM_SETS);
  cudaMemcpy(device_array1, host_array1, DSIZE * NUM_SETS * sizeof(mytype), cudaMemcpyHostToDevice);
  cudaMemcpy(device_array2, host_array2, DSIZE * NUM_SETS * sizeof(mytype), cudaMemcpyHostToDevice);
  unsigned long gpu_time = time_usec(0);
  row_major_merge<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(device_array1, device_array2, device_result, NUM_SETS, DSIZE);
  cudaDeviceSynchronize();
  gpu_time = time_usec(gpu_time);
  cudaMemcpy(host_result, device_result, DSIZE * NUM_SETS * 2 * sizeof(mytype), cudaMemcpyDeviceToHost);
  unsigned long cpu_time = time_usec(0);
  if (!validate_row_major(host_array1, host_array2, host_result, NUM_SETS, DSIZE)) {
    printf("row-major validation failed!\n");
    return 1;
  }
  cpu_time = time_usec(cpu_time);
  printf("CPU time: %f, GPU row-major time: %f\n", cpu_time / (float)USECPSEC, gpu_time / (float)USECPSEC);

  // Test "column-major" storage
  mytype *host_array_col1, *host_array_col2, *host_result_col;
  host_array_col1 = (mytype *)malloc(DSIZE * NUM_SETS * sizeof(mytype));
  host_array_col2 = (mytype *)malloc(DSIZE * NUM_SETS * sizeof(mytype));
  host_result_col = (mytype *)malloc(DSIZE * NUM_SETS * sizeof(mytype));
  for (int i = 0; i < NUM_SETS; i++)
    for (int j = 0; j < DSIZE; j++){
      host_array_col1[j * NUM_SETS + i] = host_array1[i * DSIZE + j];
      host_array_col2[j * NUM_SETS + i] = host_array2[i * DSIZE + j];
    }
  cudaMemcpy(device_array1, host_array_col1, DSIZE * NUM_SETS * sizeof(mytype), cudaMemcpyHostToDevice);
  cudaMemcpy(device_array2, host_array_col2, DSIZE * NUM_SETS * sizeof(mytype), cudaMemcpyHostToDevice);
  gpu_time = time_usec(0);
  column_major_merge<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(device_array1, device_array2, device_result, NUM_SETS, DSIZE, NUM_SETS, NUM_SETS, NUM_SETS);
  cudaDeviceSynchronize();
  gpu_time = time_usec(gpu_time);
  cudaMemcpy(host_result_col, device_result, DSIZE * NUM_SETS * 2 * sizeof(mytype), cudaMemcpyDeviceToHost);
  if (!validate_column_major(host_result, host_result_col, NUM_SETS, DSIZE)) {
    printf("column-major validation failed!\n");
    return 1;
  }

  printf("GPU column-major time: %f\n", gpu_time / (float)USECPSEC);
  return 0;
}
