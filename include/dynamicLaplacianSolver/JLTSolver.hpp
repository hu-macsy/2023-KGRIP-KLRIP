#ifndef JLT_SOLVER_HPP
#define JLT_SOLVER_HPP

#include <dynamicLaplacianSolver/IDynamicLaplacianSolver.hpp>
#include <dynamicLaplacianSolver/SparseLUSolver.hpp>

#include <Eigen/Dense>
#include <networkit/graph/Graph.hpp>

template <class MatrixType, class DynamicSolver>
class JLTSolver : virtual public IDynamicLaplacianSolver {
public:
  JLTSolver(){};
  JLTSolver(const JLTSolver& other) = default;
  JLTSolver& operator=(const JLTSolver& other) = default;
  virtual void setup(const Graph &g, double tolerance,
                     count eqnsPerRound) override;
  virtual void computeColumns(std::vector<node> nodes) override;
  virtual void addEdge(node u, node v) override;
  virtual count getComputedColumnCount() override;
  virtual double totalResistanceDifferenceApprox(node u, node v) override;
  virtual double totalResistanceDifferenceExact(node u, node v) override;
  double effR(node u, node v);
  double phiNormSquared(node u, node v);

private:
  int jltDimension(int n, double epsilon);

  void computeIntermediateMatrices();

  DynamicSolver solver;

  count n, m;
  count l;

  Eigen::MatrixXd P_n, P_m;
  Eigen::MatrixXd PL, PBL;

  double epsilon;

  std::mt19937 gen{1};
  std::normal_distribution<> d{0, 1};

  Graph G;
  Eigen::SparseMatrix<double> incidence;
};

extern template class JLTSolver<Eigen::SparseMatrix<double>, SparseLUSolver>;
typedef JLTSolver<Eigen::SparseMatrix<double>, SparseLUSolver> JLTLUSolver;

#endif