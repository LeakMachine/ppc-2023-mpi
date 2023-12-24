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

std::vector<std::vector<unsigned char>> applyFilterMPI(const std::vector<std::vector<unsigned char>>& _image) {
    int rank, size, blockRows, blockStart, blockEnd;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = _image.size();
    int cols = _image[0].size();

    blockRows = rows / size;
    blockStart = rank * blockRows;
    blockEnd = (rank == size - 1) ? rows : blockStart + blockRows;

    std::vector<std::vector<unsigned char>> localImage(blockEnd - blockStart, std::vector<unsigned char>(cols));
    std::vector<std::vector<unsigned char>> localOutput(blockEnd - blockStart, std::vector<unsigned char>(cols));
    std::vector<std::vector<unsigned char>> finalOutput(rows, std::vector<unsigned char>(cols));

    for (int i = 0; i < blockEnd - blockStart; i++) {
        for (int j = 0; j < cols; j++) {
            localImage[i][j] = _image[i + blockStart][j];
        }
    }

    for (int i = 0; i < blockEnd - blockStart; i++) {
        for (int j = 0; j < cols; j++) {
            localOutput[i][j] = funcProcessPixel(i + 1, j + 1, localImage);
        }
    }

    std::vector<unsigned char> flattenLocalOutput;
    for (const auto& vec : localOutput) {
        flattenLocalOutput.insert(flattenLocalOutput.end(), vec.begin(), vec.end());
    }

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size - 1; ++i) {
        sendcounts[i] = blockRows * cols;
        displs[i] = i * blockRows * cols;
    }
    sendcounts[size - 1] = (rows - (size - 1) * blockRows) * cols;
    displs[size - 1] = (size - 1) * blockRows * cols;

    std::vector<unsigned char> all_pixels(rows * cols);
    MPI_Gatherv(&flattenLocalOutput[0], flattenLocalOutput.size(), MPI_UNSIGNED_CHAR,
        &all_pixels[0], sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    auto itr = all_pixels.begin();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            finalOutput[i][j] = *itr++;
        }
    }

    return finalOutput;
}
