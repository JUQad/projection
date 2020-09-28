#ifndef GUARD_budgetMaintenance_h
#define GUARD_budgetMaintenance_h

#include "loadData.h"
#include "kernel.h"
#include <tuple>
#include <vector>
#include</Users/saharqaadan/Documents/issues2020/project_INI2020/eigen/Eigen/Dense>
//#include</Users/saharqaadan/Documents/issues2020/project_INI2020/eigen/Eigen/
#include "omp.h"
//#include </Users/saharqaadan/Documents/issues2020/project_INI2020/Shark/include/shark/LinAlg/BLAS/matrix_expression.hpp>
//#include "shark/LinAlg/BLAS/matrix_expression.hpp"

//#include "Remora/include/remora/remora.hpp"
//#include "Remora/include/remora/solve.hpp"
//#include "Remora/include/remora/assignment.hpp"
//#include "Remora/include/remora/decompositions.hpp"

using LookupTable = std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>;
using Heuristic = std::tuple<INDEX, INDEX>(std::vector<double>&, sparseData&, Kernel const&, LookupTable const&, double);

Heuristic mergeHeuristicWD;
Heuristic mergeHeuristicRandom;
Heuristic mergeHeuristicKernel;
Heuristic mergeHeuristicRandomWD;
Heuristic mergeHeuristicMinAlpha;
Heuristic mergeHeuristicMintwoAlphas;
Heuristic mergeHeuristic59plusWD;
Heuristic mergeHeuristicReprocessLASVM;
Heuristic projection_smallestalpha;


using HeuristicWithmoreVectors = std::tuple<INDEX, INDEX, std::vector<double>>(std::vector<double>&, sparseData&, Kernel const&, LookupTable const&, double, std::vector<double>);

HeuristicWithmoreVectors mergeHeuristicWDVector;

int mergeAndDeleteSVwithmoreVectors(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, HeuristicWithmoreVectors heuristicWithmoreVectors);

int mergeAndDeleteSV(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, Heuristic heuristic);


int projectSV(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic);

std::vector<double> projectSVLinearEquations(std::vector<double>& dual_variables_notpseudo, sparseData& dataset_notpseudo, std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic);

double  mergeAndDeleteSV_pVector(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, Heuristic heuristic, std::vector<double>);

int projectAndDeleteSV(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic);


int multimergeAndDeleteSV(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic);

int DeleteSV(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic);

std::tuple<int,  double , std::vector<SE>, char > mergeDeleteAdd(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic);

#endif
