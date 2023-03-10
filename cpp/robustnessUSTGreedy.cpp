#include <robustnessUSTGreedy.hpp>

#include <random>
#include <set>

#include <laplacian.hpp>

using namespace NetworKit;

inline bool operator<(const QueueItem &left, const QueueItem &right) {
  return left.value < right.value ||
         (right.value == left.value &&
          (left.u < right.u || (left.u == right.u && left.v < right.v)));
}

template <class DynamicLaplacianSolver>
ColStoch<DynamicLaplacianSolver>::ColStoch(GreedyParams params) : G(params.g) {
  this->originalG = params.g;
  this->n = G.numberOfNodes();
  this->k = params.k;
  this->focus_node = params.focus_node;
  this->epsilon = params.epsilon;
  this->epsilon2 = params.epsilon2;
  this->solverEpsilon = params.solverEpsilon;
  this->heuristic = params.heuristic;
  this->candidatesize = params.candidatesize;
  this->round = 0;

  if (this->k > 20)
    solver.setup(G, params.solverEpsilon, numberOfNodeCandidates());
  else
    solver.setup(G, params.solverEpsilon,
                 std::ceil(n * std::sqrt(std::log(1.0 / epsilon))));
  this->totalValue = 0.;
  this->originalResistance = 0.;
  this->always_use_known_columns_as_candidates =
      params.always_use_known_columns_as_candidates;

  gen = std::mt19937(Aux::Random::getSeed());

  if (heuristic == HeuristicType::lpinvDiag) {
    apx =
        std::make_unique<ApproxElectricalCloseness>(params.g, params.epsilon2);
    apx->run();
    this->originalApx = std::make_unique<ApproxElectricalCloseness>(*apx);
    this->diag = apx->getDiagonal();
    G.forNodes([&](node u) {
      this->totalValue -= static_cast<double>(this->n) * this->diag[u];
    });
    originalResistance = -1. * totalValue;
    // INFO("Usts: ", apx->getUstCount());
  } else {
    similarityIterations = params.similarityIterations;
    totalValue = 0.;
    // pass
  }
}

template <class DynamicLaplacianSolver>
void ColStoch<DynamicLaplacianSolver>::reset_focus(const node &fn) {
  DEBUG("ColStoch::reset_focus: removing result edges");
  for (auto edge : this->results)
    this->G.removeEdge(edge.u, edge.v);
  DEBUG("ColStoch::reset_focus: done removing edges");
  assert(this->G.numberOfEdges() == this->originalG.numberOfEdges());
  // this->G = this->originalG;
  this->focus_node = fn;
  this->round = 0;

  DEBUG("ColStoch::reset_focus: destruct and re-initialize laplacian solver");
  solver.~DynamicLaplacianSolver();
  new (&solver) DynamicLaplacianSolver();
  if (this->k > 20)
    solver.setup(G, this->solverEpsilon, numberOfNodeCandidates());
  else
    solver.setup(G, this->solverEpsilon,
                 std::ceil(n * std::sqrt(std::log(1.0 / epsilon))));
  DEBUG("ColStoch::reset_focus: laplacian solver reinitialized.");
  this->totalValue = 0.;
  this->originalResistance = 0.;

  gen = std::mt19937(Aux::Random::getSeed());

  if (heuristic == HeuristicType::lpinvDiag) {
    DEBUG("ColStoch::reset_focus: copying ApproxElectricalCloseness");
    this->apx =
        std::make_unique<ApproxElectricalCloseness>(*(this->originalApx));
    DEBUG("ColStoch::reset_focus: &G: ", &G);
    this->diag = apx->getDiagonal();
    G.forNodes([&](node u) {
      this->totalValue -= static_cast<double>(this->n) * this->diag[u];
    });
    originalResistance = -1. * totalValue;
  }

  this->hasRun = false;
  DEBUG("ColStoch::reset_focus: reset done");
}

template <class DynamicLaplacianSolver>
double ColStoch<DynamicLaplacianSolver>::getResultValue() {
  return this->totalValue * (-1.0);
}

template <class DynamicLaplacianSolver>
double ColStoch<DynamicLaplacianSolver>::getOriginalValue() {
  return this->originalResistance;
}

template <class DynamicLaplacianSolver>
std::vector<NetworKit::Edge>
ColStoch<DynamicLaplacianSolver>::getResultItems() {
  return this->results;
}

template <class DynamicLaplacianSolver>
bool ColStoch<DynamicLaplacianSolver>::isValidSolution() {
  return this->validSolution;
}

template <class DynamicLaplacianSolver>
count ColStoch<DynamicLaplacianSolver>::numberOfNodeCandidates() {
  // std::cout << this->n << ", " << this->G.degree(this->focus_node) << ", " <<
  // this->round << ", sum:" << (this->n - this->G.degree(this->focus_node) -
  // this->round) << "\n"; std::cout << std::log(1.0 / epsilon) << "\n";
  return std::ceil((this->n - this->G.degree(this->focus_node) - this->round) /
                   k * std::log(1.0 / epsilon)) +
         1;
}

template <class DynamicLaplacianSolver>
void ColStoch<DynamicLaplacianSolver>::run() {
  this->round = 0;
  this->validSolution = false;
  this->results.clear();
  struct edge_cmp {
    bool operator()(const Edge &lhs, const Edge &rhs) const {
      return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
    }
  };
  std::set<Edge, edge_cmp> resultSet;

  if (k + G.numberOfEdges() >
      (n * (n - 1) / 8 * 3)) { // 3/4 of the complete graph.
    this->hasRun = true;
    std::cout << "Bad call to TreeGreedy, adding this many edges is not "
                 "supported! Attempting to have "
              << k + G.numberOfEdges() << " edges, limit is "
              << n * (n - 1) / 8 * 3;
    return;
  }

  for (int r = 0; r < k; r++) {
    DEBUG("ColStoch::run: round ", 1);
    // std::cout << "Round: " << r << "\n";
    double bestGain = -std::numeric_limits<double>::infinity();
    Edge bestEdge;

    int it = 0;

    do {

      // std::cout << "it: " << it << "\n";
      // Collect nodes set for current round
      std::set<NetworKit::node> nodes;
      unsigned int s = numberOfNodeCandidates();
      // std::cout << "s: " << s << "\n";
      // if (r == 0) { INFO("Columns in round 1: ", s); }

      double min = std::numeric_limits<double>::infinity();

      std::vector<double> nodeWeights(n);

      // the random choice following this may fail if all the vertex pairs are
      // already present as edges, we use heuristic information the first
      // time, uniform distribution if it fails
      if (heuristic == HeuristicType::lpinvDiag && it < 2) {
        double tr = -1. * this->totalValue;
        G.forNodes([&](node u) {
          double val = static_cast<double>(n) * diag[u] + tr;
          if (val < min) {
            min = val;
          }
          nodeWeights[u] = val;
        });
        for (auto &v : nodeWeights) {
          auto u = v - min;
          v = u * u;
        }
      } else if (heuristic == HeuristicType::similarity && it < 2) {
        // TODO export the parameters ...
        count maxDegree = 0;
        G.forNodes([&](node u) {
          if (G.degree(u) > maxDegree)
            maxDegree = G.degree(u);
        });
        double phi = 1. / maxDegree;
        count iterations = similarityIterations;
        Eigen::SparseMatrix<double> phi_A = phi * adjacencyMatrix(G);
        Eigen::VectorXd d(n);
        G.forNodes([&](node i) { d(i) = 1. / G.degree(i); });
        Eigen::VectorXd u = d;
        for (int i = 0; i < iterations; i++) {
          u = phi_A * u + d;
        }
        Eigen::VectorXd summedSimilarity = u;

        // We are interested in nodes that are dissimilar to other nodes i.e.
        // the summed similarity score should be small.
        double max = -std::numeric_limits<double>::infinity();
        G.forNodes([&](node i) {
          summedSimilarity(i) /= G.degree(i);
          double val = summedSimilarity(i);
          if (val > max) {
            max = val;
          }
          nodeWeights[i] = val;
        });
        for (auto &v : nodeWeights) {
          auto w = std::abs(max - v) / max;
          v = w * w;
          // v = (w * w + (1-w)*(1-w)) * 2 - 1;
        }
      } else {
        G.forNodes([&](node u) { nodeWeights[u] = 1.; });
      }
      it++;

      std::discrete_distribution<> distribution_nodes_heuristic(
          nodeWeights.begin(), nodeWeights.end());

      nodeWeights.clear();
      for (int i = 0; i < n; i++) {
        nodeWeights.push_back(1.0);
      }
      std::discrete_distribution<> distribution_nodes_uniform(
          nodeWeights.begin(), nodeWeights.end());

      while (nodes.size() < s / 2) {
        nodes.insert(distribution_nodes_heuristic(gen));
      }

      while (nodes.size() < s) {
        nodes.insert(distribution_nodes_uniform(gen));
      }
      std::vector<node> nodesVec{nodes.begin(), nodes.end()};
      auto edgeValid = [&](node u, node v) {
        if (this->G.hasEdge(u, v) || this->G.hasEdge(v, u))
          return false;
        if (resultSet.count(Edge(u, v)) > 0 ||
            resultSet.count(Edge(v, u)) > 0) {
          return false;
        }
        return true;
      };
      solver.computeColumns(nodesVec);

      if (always_use_known_columns_as_candidates) {
        throw std::logic_error(
            "always use known columns is not implemented yet");
        for (auto u : nodesVec) {
          if (knownColumns.count(u) == 0) {
            for (auto v : knownColumns) {
              if (edgeValid(u, v)) {
                QueueItem qe = {u, v, std::numeric_limits<double>::infinity(),
                                -1};
                queue.push(qe);
              }
            }
            knownColumns.insert(u);
          }
        }
        nodesVec = std::vector(knownColumns.begin(), knownColumns.end());
        std::cout << nodesVec.size() << "\n";

        // Determine best edge between nodes from node set

        while (true) {
          if (queue.size() == 0) {
            throw std::logic_error("Added zero nodes, error. ");
          }
          auto qe = queue.top();
          if (qe.lastUpdated < round) {
            queue.pop();
            qe = {qe.u, qe.v,
                  solver.totalResistanceDifferenceApprox(qe.u, qe.v), round};
            queue.push(qe);
          } else {
            bestGain = qe.value;
            bestEdge = Edge(qe.u, qe.v);
            break;
          }
        }
      } else {
        for (int i = 0; i < nodesVec.size(); i++) {
          auto u = nodesVec[i];
          if (edgeValid(focus_node, u)) {
            double gain = solver.totalResistanceDifferenceApprox(focus_node, u);

            if (gain > bestGain) {
              bestEdge = Edge(focus_node, u);
              bestGain = gain;
            }
          }
        }
      }
    } while (bestGain == -std::numeric_limits<double>::infinity());

    // Accept edge
    resultSet.insert(bestEdge);

    auto u = bestEdge.u;
    auto v = bestEdge.v;
    G.addEdge(u, v);
    DEBUG("ColStoch::run: edge added to G.");
    // computing final result on exact value
    bestGain = solver.totalResistanceDifferenceExact(u, v);

    this->totalValue += bestGain;

    if (this->round < k - 1) {

      if (heuristic == HeuristicType::lpinvDiag) {
        solver.addEdge(u, v);
        DEBUG("ColStoch::run: edge added to solver.");
        apx->edgeAdded(u, v);
        this->diag = apx->getDiagonal();
      } else {
        solver.addEdge(u, v);
      }
    } else {
      this->validSolution = true;
    }
    this->round++;
  }
  this->hasRun = true;
  this->results = {resultSet.begin(), resultSet.end()};
  // INFO("Computed columns: ", solver.getComputedColumnCount());
}

template class ColStoch<JLTLamgSolver>;
template class ColStoch<SparseLUSolver>;
template class ColStoch<LamgDynamicLaplacianSolver>;
template class ColStoch<JLTLUSolver>;
