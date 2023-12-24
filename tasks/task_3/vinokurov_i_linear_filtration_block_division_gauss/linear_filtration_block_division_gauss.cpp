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

std::vector<std::vector<unsigned char>> applyFilter(const std::vector<std::vector<unsigned char>>& _image) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = _image.size();
    int cols = _image[0].size();
    int blockRows = rows / size;

    std::vector<std::vector<unsigned char>> localOutput(blockRows, std::vector<unsigned char>(cols));
    std::vector<std::vector<unsigned char>> finalOutput(rows, std::vector<unsigned char>(cols));

    if (rank == 0) {
        std::vector<unsigned char> flattenImage;
        flattenImage.reserve(rows * cols);
        for (const auto& row : _image) {
            flattenImage.insert(flattenImage.end(), row.begin(), row.end());
        }

        for (int dest = 1; dest < size; ++dest) {
            MPI_Send(flattenImage.data(), rows * cols, MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
        }

        for (int i = 0; i < blockRows; ++i) {
            for (int j = 0; j < cols; ++j) {
                localOutput[i][j] = funcProcessPixelFlat(i, j, flattenImage, cols);
            }
        }
    } else {
        std::vector<unsigned char> recvBuffer(rows * cols);
        MPI_Recv(recvBuffer.data(), rows * cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < blockRows; ++i) {
            for (int j = 0; j < cols; ++j) {
                localOutput[i][j] = funcProcessPixelFlat(i + rank * blockRows, j, recvBuffer, cols);
            }
        }
    }

    if (rank == 0) {
        for (int src = 1; src < size; ++src) {
            MPI_Recv(&finalOutput[src * blockRows][0], blockRows * cols, MPI_UNSIGNED_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        finalOutput = localOutput;
    } else {
        MPI_Send(&localOutput[0][0], blockRows * cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    return finalOutput;
}
