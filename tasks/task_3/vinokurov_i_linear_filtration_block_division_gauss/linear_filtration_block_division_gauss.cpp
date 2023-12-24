// Copyright 2023 Vinokurov Ivan
#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <cmath>
#include "task_3/vinokurov_i_linear_filtration_block_division_gauss/linear_filtration_block_division_gauss.h"

int funcClamp(int _input, int _min, int _max) {
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

std::vector<std::vector<unsigned char>> applyFilter(const std::vector<std::vector<unsigned char>>& _image) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = _image.size();
    int cols = _image[0].size();
    int blockRows = rows / size;

    std::vector<std::vector<unsigned char>> localImage(blockRows, std::vector<unsigned char>(cols));
    std::vector<std::vector<unsigned char>> localOutput(blockRows, std::vector<unsigned char>(cols));
    std::vector<std::vector<unsigned char>> finalOutput(rows, std::vector<unsigned char>(cols));

    std::vector<int> sendcounts(size, blockRows * cols);
    std::vector<int> displs(size, 0);

    MPI_Scatter(&_image[0][0], blockRows * cols, MPI_UNSIGNED_CHAR,
        &localImage[0][0], blockRows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < blockRows; ++i) {
        for (int j = 0; j < cols; ++j) {
            localOutput[i][j] = funcProcessPixel(i + rank * blockRows + 1, j + 1, _image);
        }
    }

    MPI_Gatherv(&localOutput[0][0], blockRows * cols, MPI_UNSIGNED_CHAR,
        &finalOutput[0][0], sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    return finalOutput;
}
