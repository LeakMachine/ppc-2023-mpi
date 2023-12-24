// Copyright 2023 Vinokurov Ivan
#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <cmath>
#include "task_3/vinokurov_i_linear_filtration_block_division_gauss/linear_filtration_block_division_gauss.h"

int funcClamp(int _max, int _min, int _input) {
    if (_input < _min) {
        return 0;
    } else if (_input > _max) {
        return _max;
    } else {
        return _input;
    }
}

float kernel[3][3] = {
    {1.0f / 16, 2.0f / 16, 1.0f / 16},
    {2.0f / 16, 4.0f / 16, 2.0f / 16},
    {1.0f / 16, 2.0f / 16, 1.0f / 16}
};

unsigned char funcProcessPixel(int _x, int _y, const std::vector<std::vector<unsigned char>>& _image) {
    int radiusX = 1;
    int radiusY = 1;
    float result = 0;

    int rows = _image.size();
    int cols = _image[0].size();
    unsigned char neighborColor;

    for (int l = -radiusY; l <= radiusY; l++) {
        for (int k = -radiusX; k <= radiusX; k++) {
            int idX = funcClamp(_x + k, 0, rows - 1);
            int idY = funcClamp(_y + l, 0, cols - 1);
            neighborColor = _image[idX][idY];
            result += neighborColor * kernel[k + radiusX][l + radiusY];
        }
    }
    return static_cast<unsigned char>(result);
}

std::vector<std::vector<unsigned char>> applyFilter(const std::vector<std::vector<unsigned char>>& _image) {
    int rows = _image.size();
    int cols = _image[0].size();

    std::vector<std::vector<unsigned char>> temp(rows, std::vector<unsigned char>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            temp[i][j] = funcProcessPixel(i + 1, j + 1, _image);
        }
    }

    return std::vector<std::vector<unsigned char>>(temp);
}

std::vector<std::vector<unsigned char>> applyFilterMPI(const std::vector<std::vector<unsigned char>>& _image) {
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows = _image.size();
    int cols = _image[0].size();

    int blockRows = rows / size;
    int blockStart = rank * blockRows;
    int blockEnd = (rank == size - 1) ? rows : blockStart + blockRows;

    std::vector<std::vector<unsigned char>> localImage(blockEnd - blockStart, std::vector<unsigned char>(cols));
    std::vector<std::vector<unsigned char>> localOutput(blockEnd - blockStart, std::vector<unsigned char>(cols));

    for (int i = blockStart; i < blockEnd; i++) {
        for (int j = 0; j < cols; j++) {
            localImage[i][j] = _image[i][j];
        }
    }

    for (int i = 0; i < blockEnd - blockStart; i++) {
        for (int j = 0; j < cols; j++) {
            localOutput[i][j] = funcProcessPixel(i + 1, j + 1, localImage);
        }
    }

    if (rank == 0 && size == 1) {
        return localOutput;
    }

    std::vector<std::vector<unsigned char>> localOutput2(rows, std::vector<unsigned char>(cols));
    MPI_Allgather(&localOutput[0][0], (blockEnd - blockStart) * cols, MPI_UNSIGNED_CHAR,
        &localOutput2[0][0], (blockEnd - blockStart) * cols, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

    return localOutput2;
}
