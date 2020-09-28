#ifndef GUARD_solver_h
#define GUARD_solver_h

#include "loadData.h"
#include "kernel.h"
#include "budgetMaintenance.h"
#include "svm.h"


//primal solver with budget
SVM BSGD(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

//primal solver with budget
SVM BDCA(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

SVM acfBDCA(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

SVM SBSCA(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

SVM BDCA_pVector(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

SVM BMVPSMO(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

SVM BMVPSMOSimplified(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

#endif
