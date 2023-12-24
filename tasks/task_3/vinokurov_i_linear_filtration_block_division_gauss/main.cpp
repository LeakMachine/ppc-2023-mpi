// Copyright 2023 Vinokurov Ivan
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "task_3/vinokurov_i_linear_filtration_block_division_gauss/linear_filtration_block_division_gauss.h"

TEST(TESTS, CanUseFunctionTest) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100, 120, 130, 110},
        {110, 115, 120, 125},
        {90, 100, 110, 120},
        {95, 105, 115, 125}
    };

    std::vector<std::vector<unsigned char>> result;

    ASSERT_NO_THROW(result = applyFilterMPI(image));
}

TEST(TESTS, CanWork4x4Test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100, 120, 130, 110},
        {110, 115, 120, 125},
        {90, 100, 110, 120},
        {95, 105, 115, 125}
    };

    std::vector<std::vector<unsigned char>> result;
    std::vector<std::vector<unsigned char>> result2;

    result = applyFilter(image);
    result2 = applyFilterMPI(image);

    for (int i = 0; i < image.size(); i++) {
        for (int j = 0; j < image[0].size(); j++) {
            ASSERT_NEAR(result[i][j], result2[i][j], 15);
        }
    }
}

TEST(TESTS, CanWork8x8Test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100, 120, 130, 110, 100, 120, 130, 110},
        {110, 115, 120, 125, 110, 115, 120, 125},
        {90, 100, 110,  120, 95, 105, 115, 125},
        {95, 105, 115, 125, 110, 115, 120, 125},
        {100, 120, 130, 110, 110, 115, 120, 125},
        {110, 115, 120, 125, 110, 115, 120, 125},
        {90, 100, 110, 120, 100, 120, 130, 110},
        {95, 105, 115, 125, 110, 115, 120, 125}
    };

    std::vector<std::vector<unsigned char>> result;
    std::vector<std::vector<unsigned char>> result2;

    result = applyFilter(image);
    result2 = applyFilterMPI(image);

    for (int i = 0; i < image.size(); i++) {
        for (int j = 0; j < image[0].size(); j++) {
            ASSERT_NEAR(result[i][j], result2[i][j], 15);
        }
    }
}

TEST(TESTS, CanWork8x12Test) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100, 120, 130, 110, 100, 120, 130, 110, 110, 115, 120, 125},
        {110, 115, 120, 125, 110, 115, 120, 125, 100, 120, 130, 110,},
        {90, 100, 110,  120, 95, 105, 115, 125, 90, 100, 110,  120},
        {95, 105, 115, 125, 110, 115, 120, 125, 100, 110, 120, 100},
        {100, 120, 130, 110, 110, 115, 120, 125, 100, 120, 130, 110},
        {110, 115, 120, 125, 110, 115, 120, 125, 115, 125, 110, 115,},
        {90, 100, 110, 120, 100, 120, 130, 110, 100, 120, 130, 110},
        {95, 105, 115, 125, 110, 115, 120, 125, 115, 125, 110, 115,}
    };

    std::vector<std::vector<unsigned char>> result;
    std::vector<std::vector<unsigned char>> result2;

    result = applyFilter(image);
    result2 = applyFilterMPI(image);

    for (int i = 0; i < image.size(); i++) {
        for (int j = 0; j < image[0].size(); j++) {
            ASSERT_NEAR(result[i][j], result2[i][j], 15);
        }
    }
}

TEST(TESTS, CannotWorkWithEmptyTest) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image;

    std::vector<std::vector<unsigned char>> result;

    ASSERT_ANY_THROW(result = applyFilterMPI(image));
}

TEST(TESTS, CannotWorkInWrongFormatTest) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100}
    };

    std::vector<std::vector<unsigned char>> result;

    ASSERT_ANY_THROW(result = applyFilterMPI(image));
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
