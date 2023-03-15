#ifndef SPARSE_LU_SOLVER_HPP
#define SPARSE_LU_SOLVER_HPP

#include <dynamicLaplacianSolver/DynamicLaplacianSolver.hpp>

#include <Eigen/Sparse>
#include <networkit/graph/Graph.hpp>

class SparseLUSolver : public DynamicLaplacianSolver<
                           Eigen::SparseMatrix<double>,
                           Eigen::SparseLU<Eigen::SparseMatrix<double>,
                                           Eigen::COLAMDOrdering<int>>> {
  virtual void setup_solver() override;
public:
  SparseLUSolver& operator=(const SparseLUSolver& other);
};

#endif