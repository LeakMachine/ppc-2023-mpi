// Copyright 2023 Vinokurov Ivan
#ifndef TASKS_TASK_3_VINOKUROV_I_LINEAR_FILTRATION_BLOCK_DIVISION_GAUSS_LINEAR_FILTRATION_BLOCK_DIVISION_GAUSS_H_
#define TASKS_TASK_3_VINOKUROV_I_LINEAR_FILTRATION_BLOCK_DIVISION_GAUSS_LINEAR_FILTRATION_BLOCK_DIVISION_GAUSS_H_

#include <mpi.h>
#include <vector>
#include <cmath>

std::vector<std::vector<int>> applyFilterMPI(
    const std::vector<std::vector<int>>& _image);

#endif  //  TASKS_TASK_3_VINOKUROV_I_LINEAR_FILTRATION_BLOCK_DIVISION_GAUSS_LINEAR_FILTRATION_BLOCK_DIVISION_GAUSS_H_