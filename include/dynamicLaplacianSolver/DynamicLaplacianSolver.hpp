#ifndef DYNAMIC_LAPLACIAN_SOLVER_HPP
#define DYNAMIC_LAPLACIAN_SOLVER_HPP

#include <dynamicLaplacianSolver/IDynamicLaplacianSolver.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <networkit/graph/Graph.hpp>

template <class MatrixType, class Solver>
class DynamicLaplacianSolver : virtual public IDynamicLaplacianSolver {
public:
  virtual void setup(const Graph &g, double tolerance,
                     count eqnsPerRound = 200) override;
  Eigen::VectorXd getColumn(node i);
  Eigen::MatrixXd solve(const Eigen::MatrixXd &rhs);
  virtual void computeColumnSingle(node i);
  virtual void computeColumns(std::vector<node> nodes);
  virtual void addEdge(node u, node v) override;
  virtual count getComputedColumnCount() override;
  virtual double totalResistanceDifferenceApprox(node u, node v);
  virtual double totalResistanceDifferenceExact(node u, node v);

protected:
  virtual void setup_solver() = 0;

  std::vector<int> colAge;
  std::vector<Eigen::VectorXd> cols;
  std::vector<Eigen::VectorXd> updateVec;
  std::vector<double> updateW;

  count n;
  count round = 0;

  std::vector<Eigen::VectorXd> rhs;
  MatrixType laplacian;
  MatrixType solverLaplacian;
  Solver solver;
  double rChange = 0.;
  count computedColumns = 0;

  // Update solver every n rounds
  count roundsPerSolverUpdate = 1;
  // If a round was last computed this many rounds ago or more, compute by
  // solving instead of updating.
  count ageToRecompute;

  count solverAge = 0;
};

extern template class DynamicLaplacianSolver<
    Eigen::SparseMatrix<double>,
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>>;

#endif