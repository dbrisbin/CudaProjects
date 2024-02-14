/// @file spmv_driver.cu
/// @brief Definition of driver function declared in spmv_driver.h.

#include <stdio.h>
#include <algorithm>
#include "spmv.h"
#include "spmv_driver.h"
#include "spmv_utils.h"
#include "types/constants.h"

float SpMVDriver(const float* mat_h, const float* vec_h, float* result_h, const int m, const int n,
                 const int iters, const SpmvKernel kernel_to_use)
{
    float *values_d{}, *vec_d{}, *result_d{}, *values_h{};
    int *col_indices_d{}, *row_indices_d{}, *row_pointers_d{}, *nnz_per_row_d{}, *col_indices_h{},
        *row_indices_h{}, *nnz_per_row_h{}, *row_pointers_h{};
    const int length = m * n;
    int max_nnz_per_row_ell;
    int number_of_nnz_elts =
        std::count_if(mat_h, mat_h + length, [](float i) { return std::fabs(i) > FLOAT_EPS; });

    dim3 dim_block{}, dim_grid{};

    cudaMalloc((void**)&vec_d, n * sizeof(float));
    cudaMalloc((void**)&result_d, m * sizeof(float));

    cudaMemcpy(vec_d, vec_h, n * sizeof(float), cudaMemcpyHostToDevice);

    switch (kernel_to_use)
    {
        case SpmvKernel::kCooSpmv:
        {
            // Convert to COO format
            values_h = new float[number_of_nnz_elts];
            col_indices_h = new int[number_of_nnz_elts];
            row_indices_h = new int[number_of_nnz_elts];
            UncompressedToCOO(mat_h, m, n, values_h, row_indices_h, col_indices_h);

            // Allocate device memory
            cudaMalloc((void**)&values_d, number_of_nnz_elts * sizeof(float));
            cudaMalloc((void**)&col_indices_d, number_of_nnz_elts * sizeof(int));
            cudaMalloc((void**)&row_indices_d, number_of_nnz_elts * sizeof(int));

            // Copy data to device
            cudaMemcpy(values_d, values_h, number_of_nnz_elts * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(col_indices_d, col_indices_h, number_of_nnz_elts * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(row_indices_d, row_indices_h, number_of_nnz_elts * sizeof(int),
                       cudaMemcpyHostToDevice);

            // delete host memory
            delete[] row_indices_h;
            delete[] col_indices_h;
            delete[] values_h;
        }
        break;
        case SpmvKernel::kCsrSpmv:
        {
            // Convert to CSR format
            values_h = new float[number_of_nnz_elts];
            col_indices_h = new int[number_of_nnz_elts];
            row_pointers_h = new int[m + 1];
            UncompressedToCSR(mat_h, m, n, values_h, col_indices_h, row_pointers_h);

            // Allocate device memory
            cudaMalloc((void**)&values_d, number_of_nnz_elts * sizeof(float));
            cudaMalloc((void**)&col_indices_d, number_of_nnz_elts * sizeof(int));
            cudaMalloc((void**)&row_pointers_d, (m + 1) * sizeof(int));

            // Copy data to device
            cudaMemcpy(values_d, values_h, number_of_nnz_elts * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(col_indices_d, col_indices_h, number_of_nnz_elts * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(row_pointers_d, row_pointers_h, (m + 1) * sizeof(int),
                       cudaMemcpyHostToDevice);

            // delete host memory
            delete[] row_pointers_h;
            delete[] col_indices_h;
            delete[] values_h;
        }
        break;
        case SpmvKernel::kEllSpmv:
        {
            // Convert to ELL format
            const auto max_nnz_per_row = MaxNnzPerRow(mat_h, m, n);
            values_h = new float[max_nnz_per_row * m];
            col_indices_h = new int[max_nnz_per_row * m];
            nnz_per_row_h = new int[m + 1];
            UncompressedToELL(mat_h, m, n, values_h, col_indices_h, nnz_per_row_h, max_nnz_per_row);

            // Allocate device memory
            cudaMalloc((void**)&values_d, max_nnz_per_row * m * sizeof(float));
            cudaMalloc((void**)&col_indices_d, max_nnz_per_row * m * sizeof(int));
            cudaMalloc((void**)&nnz_per_row_d, (m + 1) * sizeof(int));

            // Copy data to device
            cudaMemcpy(values_d, values_h, max_nnz_per_row * m * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(col_indices_d, col_indices_h, max_nnz_per_row * m * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(nnz_per_row_d, nnz_per_row_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

            // delete host memory
            delete[] nnz_per_row_h;
            delete[] col_indices_h;
            delete[] values_h;
        }
        break;
        case SpmvKernel::kEllCooSpmv:
        {
            // Convert to ELL-COO format
            const auto max_nnz_per_row = MaxNnzPerRow(mat_h, m, n);
            values_h = new float[number_of_nnz_elts];
            col_indices_h = new int[number_of_nnz_elts];
            row_indices_h = new int[number_of_nnz_elts];
            nnz_per_row_h = new int[m + 1];
            float* ell_values_h = new float[max_nnz_per_row * m];
            int* ell_col_indices_h = new int[max_nnz_per_row * m];

            UncompressedToELLCOO(mat_h, m, n, ell_values_h, ell_col_indices_h, nnz_per_row_h,
                                 values_h, row_indices_h, col_indices_h, &max_nnz_per_row_ell,
                                 &number_of_nnz_elts);

            // Allocate device memory
            cudaMalloc((void**)&values_d, max_nnz_per_row_ell * m * sizeof(float));
            cudaMalloc((void**)&col_indices_d, max_nnz_per_row_ell * m * sizeof(int));
            cudaMalloc((void**)&nnz_per_row_d, (m + 1) * sizeof(int));

            // Copy data to device
            cudaMemcpy(values_d, ell_values_h, max_nnz_per_row_ell * m * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(col_indices_d, ell_col_indices_h, max_nnz_per_row_ell * m * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(nnz_per_row_d, nnz_per_row_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

            // delete host memory
            delete[] ell_col_indices_h;
            delete[] ell_values_h;
            delete[] nnz_per_row_h;
        }
        break;
        case SpmvKernel::kNumKernels:
        default:
            printf("Invalid kernel selected! Try again!\n");
            return -1.0;
    }

    float time{};
    float total_time{};
    cudaEvent_t start, stop;
    cudaError_t err;

    for (int iter = 0; iter < iters; ++iter)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        switch (kernel_to_use)
        {
            case SpmvKernel::kCooSpmv:
                dim_block = dim3(SECTION_SIZE, 1, 1);
                dim_grid = dim3(ceil(static_cast<double>(number_of_nnz_elts) / SECTION_SIZE), 1, 1);
                CooSpmvKernel<<<dim_grid, dim_block>>>(values_d, col_indices_d, row_indices_d,
                                                       vec_d, result_d, number_of_nnz_elts);

                err = cudaMemcpy(result_h, result_d, m * sizeof(float), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
                }
                break;
            case SpmvKernel::kCsrSpmv:
                dim_block = dim3(SECTION_SIZE, 1, 1);
                dim_grid = dim3(ceil(static_cast<double>(m) / SECTION_SIZE), 1, 1);
                CsrSpmvKernel<<<dim_grid, dim_block>>>(values_d, col_indices_d, row_pointers_d,
                                                       vec_d, result_d, m);

                err = cudaMemcpy(result_h, result_d, m * sizeof(float), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
                }
                break;
            case SpmvKernel::kEllSpmv:
                dim_block = dim3(SECTION_SIZE, 1, 1);
                dim_grid = dim3(ceil(static_cast<double>(m) / SECTION_SIZE), 1, 1);
                EllSpmvKernel<<<dim_grid, dim_block>>>(values_d, col_indices_d, nnz_per_row_d,
                                                       vec_d, result_d, m);

                err = cudaMemcpy(result_h, result_d, m * sizeof(float), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
                }
                break;
            case SpmvKernel::kEllCooSpmv:
            {
                dim_block = dim3(SECTION_SIZE, 1, 1);
                dim_grid = dim3(ceil(static_cast<double>(m) / SECTION_SIZE), 1, 1);
                // Compute the ELL part
                EllSpmvKernel<<<dim_grid, dim_block>>>(values_d, col_indices_d, nnz_per_row_d,
                                                       vec_d, result_d, m);
                err = cudaMemcpy(result_h, result_d, m * sizeof(float), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    printf("%s in %s at line %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
                }
                // Compute the COO part on host
                float* coo_result_h = new float[m];
                CooSpmvCPU(values_h, row_indices_h, col_indices_h, vec_h, coo_result_h,
                           number_of_nnz_elts);

                // Add the ELL and COO results
                std::transform(result_h, result_h + m, coo_result_h, result_h, std::plus<float>());
            }
            break;
            case SpmvKernel::kNumKernels:
            default:
                printf("Invalid kernel selected! Try again!\n");
                return -1.0;
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        total_time += time;
    }

    cudaFree(row_indices_d);
    cudaFree(col_indices_d);
    cudaFree(result_d);
    cudaFree(vec_d);
    cudaFree(values_d);

    delete[] row_indices_h;
    delete[] col_indices_h;
    delete[] values_h;

    return total_time;
}
