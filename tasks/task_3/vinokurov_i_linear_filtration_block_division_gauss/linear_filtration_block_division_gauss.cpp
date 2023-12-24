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

std::vector<unsigned char> convertTo1D(const std::vector<std::vector<unsigned char>>& image) {
    std::vector<unsigned char> flattenedImage;
    for (const auto& row : image) {
        flattenedImage.insert(flattenedImage.end(), row.begin(), row.end());
    }
    return flattenedImage;
}

std::vector<std::vector<unsigned char>> convertTo2D(const std::vector<unsigned char>& flattenedImage,
                                        int rows, int cols) {
    std::vector<std::vector<unsigned char>> image;
    for (int i = 0; i < rows; ++i) {
        image.emplace_back(flattenedImage.begin() + i * cols, flattenedImage.begin() + (i + 1) * cols);
    }
    return image;
}

std::vector<std::vector<unsigned char>> applyFilterMPI(const std::vector<std::vector<unsigned char>>& image) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = image.size();
    int cols = image[0].size();

    std::vector<unsigned char> flattenedImage = convertTo1D(image);

    if (image.empty()) {
        throw std::runtime_error("Cannot work with an empty image");
    }

    int block_size = rows / size;

    if (block_size == 0) {
        throw std::runtime_error("Cannot work with wrong parameters");
    }

    int block_start = rank * block_size;
    int block_end = (rank == size - 1) ? rows : block_start + block_size;

    std::vector<unsigned char> localFlattenedImage(flattenedImage.begin() + block_start * cols,
        flattenedImage.begin() + block_end * cols);

    std::vector<unsigned char> localResult(localFlattenedImage.size());

    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < cols; ++j) {
            int x = block_start + i;
            int y = j;

            localResult[i * cols + j] = funcProcessPixelFlat(x, y, flattenedImage, cols);
        }
    }

    std::vector<unsigned char> gatheredResult(rows * cols);
    MPI_Allgather(localResult.data(), localResult.size(), MPI_UNSIGNED_CHAR,
        gatheredResult.data(), localResult.size(), MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

    return convertTo2D(gatheredResult, rows, cols);
}
