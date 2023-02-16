

#ifndef GREEDY_PARAMS_HPP
#define GREEDY_PARAMS_HPP

#include <networkit/graph/Graph.hpp>

enum class HeuristicType { random, lpinvDiag, similarity };

enum class CandidateSetSize { small, large };

class GreedyParams {
public:
  GreedyParams(NetworKit::Graph &g, NetworKit::count k) : g(g), k(k) {
    n = g.numberOfNodes();
  }

  NetworKit::count k;
  NetworKit::count n;
  double solverEpsilon = 0.00001;
  double epsilon = 0.1;
  double epsilon2 = 0.1;
  unsigned int ne = 1;
  unsigned int updatePerRound = 1;
  unsigned int diff = 1;
  NetworKit::Graph &g;
  HeuristicType heuristic;
  CandidateSetSize candidatesize;
  NetworKit::count similarityIterations = 100;
  bool always_use_known_columns_as_candidates = false;
};

#endif
