// Copyright 2023 Vinokurov Ivan
#include <random>
#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <cmath>
#include "task_2/vinokurov_i_seidel_iteration_method/seidel_iteration_method.h"

std::vector<double> funcSystemSolveSeidelMPI(const std::vector<std::vector<double>>& _mtxA,
                                             const std::vector<double>& _vectorB,
                                             int _numRows, double _eps) {
    int size, _rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int k = 0;

    int block_size = _numRows / size;
    int remaining_rows = _numRows % size;

    int start_row = _rank * block_size + std::min(_rank, remaining_rows);
    int end_row = start_row + block_size + (_rank < remaining_rows ? 1 : 0);

    std::vector<double> x(_numRows, 0.0);
    std::vector<double> xNew(_numRows, 0.0);

    bool converged = false;
    while (!converged) {
        k++;
        if (k > 100000) {
            // having this many cycle repetitions means there are no roots for this system
            return std::vector<double>(_numRows, 0.0);
        }
        for (int i = start_row; i < end_row; ++i) {
            double sum1 = 0.0, sum2 = 0.0;
            for (int j = 0; j < i; ++j) {
                sum1 += _mtxA[i][j] * xNew[j];
            }
            for (int j = i + 1; j < _numRows; ++j) {
                sum2 += _mtxA[i][j] * x[j];
            }
            xNew[i] = (_vectorB[i] - sum1 - sum2) / _mtxA[i][i];
        }

        MPI_Allgatherv(&xNew[start_row], end_row - start_row - 1, MPI_DOUBLE, &xNew[0],
            &block_size, &start_row, MPI_DOUBLE, MPI_COMM_WORLD);

        double local_max_diff = 0.0;
        for (int i = 0; i < _numRows; ++i) {
            double diff = std::abs(xNew[i] - x[i]);
            if (diff > local_max_diff) {
                local_max_diff = diff;
            }
        }

        double global_max_diff;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (global_max_diff < _eps) {
            converged = true;
        }

        x = xNew;
    }
    return x;
}
