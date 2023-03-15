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

SparseLUSolver& SparseLUSolver::operator=(const SparseLUSolver& other) {
  if (this == &other) return *this;

  this->colAge = other.colAge;
  this->cols = other.cols;
  this->updateVec = other.updateVec;
  this->updateW = other.updateW;
  this->n = other.n;
  this->round = other.round;
  this->rhs = other.rhs;
  this->laplacian = other.laplacian;
  this->solverLaplacian = other.solverLaplacian;
  // swap(first.solver, second.solver);
  this->rChange = other.rChange;
  this->computedColumns = other.computedColumns;
  this->roundsPerSolverUpdate = other.roundsPerSolverUpdate;
  this->ageToRecompute = other.ageToRecompute;
  this->solverAge = other.solverAge;

  // seems like there is no way to copy a Eigen::LUSolver .... 
  // so we just have to setup the new one with the same parameters
  // from setup_solver:
  this->solver.setPivotThreshold(0.1);
  this->solver.compute(this->solverLaplacian);
  
  if (this->solver.info() != Eigen::Success) {
    throw std::logic_error("Solver Setup failed.");
  }

  return *this;
}