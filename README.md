# CUDA Code Explanation

This repository contains CUDA code for performing parallel merge operations on sorted arrays using both row-major and column-major storage formats. The code demonstrates how to utilize GPU acceleration to merge large sets of data efficiently.

## Code Structure

### 1. Constants and Macros

- `NUM_SETS`: Number of sets of arrays to merge.
- `DSIZE`: Size of each array in a set.
- `mytype`: Typedef for the data type of array elements.
- `cmp(X,Y)`: Macro for comparison of elements for ascending order.
- `THREADS_PER_BLOCK`: Number of threads per CUDA block.
- `BLOCKS_PER_GRID`: Number of CUDA blocks per grid.
- `USECPSEC`: Microseconds per second for timing calculations.

### 2. Helper Functions

- `time_usec(unsigned long long start)`: Calculates the current time in microseconds relative to a start time using `gettimeofday`.
- `merge_arrays(const T * __restrict__ arr1, const T * __restrict__ arr2, T * __restrict__ result, const unsigned len_arr1, const unsigned len_arr2, const unsigned stride_arr1 = 1, const unsigned stride_arr2 = 1, const unsigned stride_result = 1)`: Merges two sorted arrays `arr1` and `arr2` into `result` using a merge sort algorithm. Can be executed on both host and device (CUDA) environments.

### 3. CUDA Kernels

- `row_major_merge(const T * __restrict__ arr1, const T * __restrict__ arr2, T * __restrict__ result, int num_arrays, int array_length)`: CUDA kernel for performing row-major merge of sorted arrays on the GPU. Each thread block handles one set of arrays, merging corresponding elements from `arr1` and `arr2` into `result`.
  
- `column_major_merge(const T * __restrict__ arr1, const T * __restrict__ arr2, T * __restrict__ result, int num_arrays, int array_length, int stride_arr1, int stride_arr2, int stride_result)`: CUDA kernel for performing column-major merge of sorted arrays on the GPU. Each thread block handles one element from each set of arrays, merging elements from `arr1` and `arr2` into `result`.

### 4. Validation Functions

- `validate_row_major(T *arr1, T *arr2, T *result, int num_arrays, int array_length)`: Validates the results of row-major merges against CPU-based merges using Thrust library functions. Compares each element of `result` against the expected merged result.
  
- `validate_column_major(const T *result1, const T *result2, int num_arrays, int array_length)`: Validates the results of column-major merges against row-major merges. Compares each element of `result1` against `result2` to ensure consistency across storage formats.

### 5. Main Function (`main()`)

- Allocates memory for host and device arrays.
- Generates random data and sorts arrays in row-major format.
- Executes row-major merge on the GPU, measures execution time, and validates results.
- Transposes arrays into column-major format, executes column-major merge on the GPU, measures execution time, and validates results.
- Outputs timing information and validation status.

## Usage

- Ensure CUDA Toolkit is installed and configured.
- Compile the code using `nvcc`.
- Run the executable to perform row-major and column-major merge operations on sorted arrays.


