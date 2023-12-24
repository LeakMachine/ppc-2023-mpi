// Copyright 2023 Vinokurov Ivan
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "task_3/vinokurov_i_linear_filtration_block_division_gauss/linear_filtration_block_division_gauss.h"

TEST(TESTS, CanUseFunctionTest) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100, 120, 130, 110, 120},
        {110, 115, 120, 125, 105},
        {90, 100, 110, 120, 130},
        {95, 105, 115, 125, 135},
        {120, 130, 140, 150, 110}
    };

    std::vector<std::vector<unsigned char>> result;
    std::vector<std::vector<unsigned char>> result2;

    ASSERT_NO_THROW(result = applyFilter(image));
    for (const auto& row : result) {
        for (const auto& pixel : row) {
            std::cout << static_cast<int>(pixel) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    ASSERT_NO_THROW(result2 = applyFilterMPI(image));
    for (const auto& row : result2) {
        for (const auto& pixel : row) {
            std::cout << static_cast<int>(pixel) << " ";
        }
        std::cout << std::endl;
    }
    EXPECT_EQ(result, result2);
}

TEST(TESTS, CanUseFunctionTest2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100, 120, 130, 110, 120},
        {110, 115, 120, 125, 105},
        {90, 100, 110, 120, 130},
        {95, 105, 115, 125, 135},
        {120, 130, 140, 150, 110}
    };

    std::vector<std::vector<unsigned char>> result;

    ASSERT_NO_THROW(result = applyFilterMPI(image));
}

TEST(TESTS, CanUseFunctionTest3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100, 120, 130, 110, 120},
        {110, 115, 120, 125, 105},
        {90, 100, 110, 120, 130},
        {95, 105, 115, 125, 135},
        {120, 130, 140, 150, 110}
    };

    std::vector<std::vector<unsigned char>> result;

    ASSERT_NO_THROW(result = applyFilterMPI(image));
}

TEST(TESTS, CanUseFunctionTest4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100, 120, 130, 110, 120},
        {110, 115, 120, 125, 105},
        {90, 100, 110, 120, 130},
        {95, 105, 115, 125, 135},
        {120, 130, 140, 150, 110}
    };

    std::vector<std::vector<unsigned char>> result;

    ASSERT_NO_THROW(result = applyFilterMPI(image));
}

TEST(TESTS, CanUseFunctionTest5) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<unsigned char>> image = {
        {100, 120, 130, 110, 120},
        {110, 115, 120, 125, 105},
        {90, 100, 110, 120, 130},
        {95, 105, 115, 125, 135},
        {120, 130, 140, 150, 110}
    };

    std::vector<std::vector<unsigned char>> result;

    ASSERT_NO_THROW(result = applyFilterMPI(image));
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
