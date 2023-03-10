#ifndef I_DYNAMIC_LAPLACIAN_SOLVER_HPP
#define I_DYNAMIC_LAPLACIAN_SOLVER_HPP

#include <networkit/graph/Graph.hpp>

using namespace NetworKit;

class IDynamicLaplacianSolver {
public:
  virtual void setup(const Graph &g, double tolerance,
                     count eqnsPerRound = 200) = 0;
  virtual void computeColumns(std::vector<node> nodes) = 0;
  virtual void addEdge(node u, node v) = 0;
  virtual count getComputedColumnCount() = 0;
  virtual double totalResistanceDifferenceApprox(node u, node v) = 0;
  virtual double totalResistanceDifferenceExact(node u, node v) = 0;
  virtual ~IDynamicLaplacianSolver() {}
};

#endif