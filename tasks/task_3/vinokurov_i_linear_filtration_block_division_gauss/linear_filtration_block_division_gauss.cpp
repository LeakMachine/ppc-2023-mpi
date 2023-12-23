// Copyright 2023 Vinokurov Ivan
#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <cmath>
#include "task_3/vinokurov_i_linear_filtration_block_division_gauss/linear_filtration_block_division_gauss.h"

int funcClamp(int _min, int _max, int _input) {
    if (_input < _min) {
        return 0;
    } else if (_input > _max) {
        return _max;
    } else {
        return _input;
    }
}

void applyGaussianFilter(const std::vector<std::vector<int>>& _input, std::vector<std::vector<int>> _output) {
    float kernel[3][3] = {
        {0.0625, 0.125, 0.0625},
        {0.125, 0.25, 0.125},
        {0.0625, 0.125, 0.0625}
    };

    int rows = _input.size();
    int cols = _input[0].size();

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            float sum = 0.0f;
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    sum += _input[i + k][j + l] * kernel[k + 1][l + 1];
                }
            }
            _output[i][j] = funcClamp(0, 255, static_cast<int>(sum));
        }
    }
}

std::vector<std::vector<int>> applyFilterMPI(const std::vector<std::vector<int>>& _image) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = _image.size();
    int cols = _image[0].size();

    std::vector<std::vector<int>> tempImage = _image;

    int blockRows = rows / size;
    int blockStart = rank * blockRows;
    int blockEnd = (rank == size - 1) ? rows : blockStart + blockRows;

    std::vector<std::vector<int>> localImage(blockEnd - blockStart, std::vector<int>(cols));
    std::vector<std::vector<int>> localOutput(blockEnd - blockStart, std::vector<int>(cols));

    if (rank != 0) {
        MPI_Send(&tempImage[blockStart - 1][0], (blockRows + 2) * cols, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < size; ++i) {
            MPI_Recv(&localImage[0][0], (blockRows + 2) * cols, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            applyGaussianFilter(localImage, localOutput);
            MPI_Send(&localOutput[0][0], blockRows * cols, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        applyGaussianFilter(tempImage, localOutput);
    } else {
        MPI_Recv(&localOutput[0][0], blockRows * cols, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank != 0) {
        MPI_Send(&localOutput[0][0], blockRows * cols, MPI_INT, 0, 1, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < size; ++i) {
            MPI_Recv(&tempImage[i * blockRows][0], blockRows * cols, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    return tempImage;
}
