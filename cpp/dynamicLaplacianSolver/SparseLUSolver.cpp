#include <dynamicLaplacianSolver/SparseLUSolver.hpp>

void SparseLUSolver::setup_solver() {
  this->solverLaplacian = this->laplacian;
  this->solverLaplacian.makeCompressed();

  this->solver.setPivotThreshold(0.1);
  this->solver.compute(this->solverLaplacian);

  if (this->solver.info() != Eigen::Success) {
    throw std::logic_error("Solver Setup failed.");
  }
  this->solverAge = this->round;
}