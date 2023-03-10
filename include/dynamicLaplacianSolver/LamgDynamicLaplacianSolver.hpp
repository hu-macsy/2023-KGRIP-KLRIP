#ifndef LAMG_DYNAMIC_LAPLACIAN_SOLVER_HPP
#define LAMG_DYNAMIC_LAPLACIAN_SOLVER_HPP

// #include <laplacian.hpp>
#include <dynamicLaplacianSolver/IDynamicLaplacianSolver.hpp>

#include <Eigen/Dense>
#include <networkit/algebraic/Vector.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>

class LamgDynamicLaplacianSolver : virtual public IDynamicLaplacianSolver {
public:
  virtual void setup(const Graph &g, double tol,
                     count eqnsPerRound = 200) override;
  void solveSingleColumn(node u);
  std::vector<NetworKit::Vector>
  parallelSolve(std::vector<NetworKit::Vector> &rhss);
  Eigen::VectorXd &getColumn(node u);
  virtual void computeColumns(std::vector<node> nodes) override;
  virtual void addEdge(node u, node v) override;
  virtual count getComputedColumnCount() override;
  virtual double totalResistanceDifferenceApprox(node u, node v);
  virtual double totalResistanceDifferenceExact(node u, node v);

private:
  std::vector<count> colAge;
  std::vector<Eigen::VectorXd> cols;
  std::vector<Eigen::VectorXd> updateVec;
  std::vector<double> updateW;

  count n;
  count round = 0;
  CSRMatrix laplacian;

  double rChange = 0.;
  count computedColumns;

  // Update solver every n rounds
  count roundsPerSolverUpdate = 10;
  // If a round was last computed this many rounds ago or more, compute by
  // solving instead of updating.
  count ageToRecompute;

  count solverAge = 0;

  Lamg<CSRMatrix> lamg;

  double tolerance;
};

#endif