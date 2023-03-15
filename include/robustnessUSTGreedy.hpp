#ifndef ROBUSTNESS_UST_GREEDY_H
#define ROBUSTNESS_UST_GREEDY_H

#include <vector>

#include <dynamicLaplacianSolver/JLTLamgSolver.hpp>
#include <dynamicLaplacianSolver/JLTSolver.hpp>
#include <dynamicLaplacianSolver/LamgDynamicLaplacianSolver.hpp>
#include <dynamicLaplacianSolver/SparseLUSolver.hpp>
#include <greedy.hpp>
#include <greedy_params.hpp>

#include <networkit/centrality/ApproxElectricalCloseness.hpp>
#include <networkit/graph/Graph.hpp>

using namespace NetworKit;

struct QueueItem {
  node u, v;
  double value;
  int lastUpdated;
};

template <class DynamicLaplacianSolver = LamgDynamicLaplacianSolver>
class ColStoch final : public AbstractOptimizer<NetworKit::Edge> {
public:
  ColStoch(GreedyParams params);
  virtual void reset_focus(const node &fn) override;
  double getResultValue();
  double getOriginalValue();
  std::vector<NetworKit::Edge> getResultItems();
  bool isValidSolution();
  count numberOfNodeCandidates();
  void run();

private:
  Graph &G;
  std::vector<Edge> results;
  node focus_node;

  int n;
  bool validSolution = false;
  int round = 0;
  int k;

  double totalValue = 0.0;
  double epsilon = 0.1;
  double epsilon2;
  double solverEpsilon;
  double originalResistance = 0.;

  count similarityIterations = 100;
  std::mt19937 gen;
  std::vector<double> diag;

  std::unique_ptr<ApproxElectricalCloseness> apx;
  std::unique_ptr<ApproxElectricalCloseness> originalApx;
  DynamicLaplacianSolver solver;
  DynamicLaplacianSolver originalSolver;

  std::set<node> knownColumns;

  HeuristicType heuristic;
  CandidateSetSize candidatesize;
  bool always_use_known_columns_as_candidates = false;

  std::priority_queue<QueueItem> queue;
};

extern template class ColStoch<JLTLamgSolver>;
extern template class ColStoch<SparseLUSolver>;
extern template class ColStoch<LamgDynamicLaplacianSolver>;
extern template class ColStoch<JLTLUSolver>;

#endif
