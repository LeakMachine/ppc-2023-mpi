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

unsigned char funcProcessPixelFlat(int _x, int _y, const std::vector<unsigned char>& flattenImage, int cols) {
    int radiusX = 1;
    int radiusY = 1;
    float result = 0;

    unsigned char neighborColor;

    for (int l = -radiusY; l <= radiusY; l++) {
        for (int k = -radiusX; k <= radiusX; k++) {
            int idX = _x + k;
            int idY = _y + l;
            idX = (idX < 0) ? 0 : ((idX >= cols) ? cols - 1 : idX);
            idY = (idY < 0) ? 0 : ((idY >= cols) ? cols - 1 : idY);
            neighborColor = flattenImage[idX * cols + idY];
            result += neighborColor * kernel[k + radiusX][l + radiusY];
        }
    }
    return static_cast<unsigned char>(result);
}

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

std::vector<std::vector<unsigned char>> applyFilterMPI(const std::vector<std::vector<unsigned char>>& image) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = image.size();
    int cols = image[0].size();

    int block_size = rows / size;
    int block_start = rank * block_size;
    int block_end = (rank == size - 1) ? rows : block_start + block_size;

    std::vector<std::vector<unsigned char>> localImage(image.begin() + block_start, image.begin() + block_end);
    std::vector<std::vector<unsigned char>> localResult(localImage.size(), std::vector<unsigned char>(cols));

    for (int i = 0; i < localImage.size(); ++i) {
        for (int j = 0; j < cols; ++j) {
            int x = block_start + i;
            int y = j;

            localResult[i][j] = funcProcessPixel(x, y, image);
        }
    }

    std::vector<std::vector<unsigned char>> gatheredResult(rows, std::vector<unsigned char>(cols));
    MPI_Allgather(localResult.data(), localResult.size() * cols, MPI_UNSIGNED_CHAR,
        gatheredResult.data(), localResult.size() * cols, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

    return gatheredResult;
}
