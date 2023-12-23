// Copyright 2023 Vinokurov Ivan
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "task_3/vinokurov_i_linear_filtration_block_division_gauss/linear_filtration_block_division_gauss.h"

TEST(TESTS, CanUseFunctionTest) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<int>> image = {
        {100, 120, 130, 110, 120},
        {110, 115, 120, 125, 105},
        {90, 100, 110, 120, 130},
        {95, 105, 115, 125, 135},
        {120, 130, 140, 150, 110}
    };

    std::vector<std::vector<int>> result;

    ASSERT_NO_THROW(result = applyFilterMPI(image));
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
