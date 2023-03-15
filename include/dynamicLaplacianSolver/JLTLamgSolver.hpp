#ifndef JLT_LAMG_SOLVER_HPP
#define JLT_LAMG_SOLVER_HPP

#include <dynamicLaplacianSolver/IDynamicLaplacianSolver.hpp>
#include <dynamicLaplacianSolver/LamgDynamicLaplacianSolver.hpp>

#include <Eigen/Dense>
#include <networkit/graph/Graph.hpp>

class JLTLamgSolver : virtual public IDynamicLaplacianSolver {
public:
  JLTLamgSolver(){};
  JLTLamgSolver(const JLTLamgSolver& other) = default;
  JLTLamgSolver& operator=(const JLTLamgSolver& other) = default;
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
  int jltDimension(int perRound, double epsilon);

  void computeIntermediateMatrices();

  LamgDynamicLaplacianSolver solver;

  count n, m;
  count l;

  DenseMatrix PL, PBL;

  double epsilon;

  std::mt19937 gen{1};
  std::normal_distribution<> d{0, 1};

  Graph G;
  CSRMatrix incidence;
};

#endif