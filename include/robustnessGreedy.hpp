#ifndef ROBUSTNESS_GREEDY_H
#define ROBUSTNESS_GREEDY_H

#include <chrono>
#include <cmath>
#include <exception>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <greedy.hpp>
#include <greedy_params.hpp>
#include <laplacian.hpp>
#include <slepc_adapter.hpp>

#include <Eigen/Dense>
#include <networkit/centrality/ApproxElectricalCloseness.hpp>
#include <networkit/components/ConnectedComponents.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/numerics/Preconditioner/DiagonalPreconditioner.hpp>

#include <networkit/auxiliary/Timer.hpp>

namespace NetworKit {
inline bool operator<(const NetworKit::Edge &lhs, const NetworKit::Edge &rhs) {
  return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
}
}; // namespace NetworKit

using namespace NetworKit;

typedef decltype(std::chrono::high_resolution_clock::now()) Time;

// The greedy algorithms in this file optimize for large -R_tot.

class RobustnessExhaustiveSearch : public virtual AbstractOptimizer<Edge> {
public:
  RobustnessExhaustiveSearch(GreedyParams params) : g(params.g), k(params.k), focus_node(params.focus_node) {}

  virtual void run() override {
    std::vector<NetworKit::Edge> es;
    std::vector<double> gain;
    int n = g.numberOfNodes();
    int num_candidates = n - g.degree(focus_node);
    if (num_candidates < k) {
      throw std::logic_error("Graph does not allow requested number of edges.");
    }

    gain.reserve(num_candidates);
    es.reserve(num_candidates);

    this->results.clear();

    this->lpinv = laplacianPseudoinverse(g);
    this->totalValue = this->lpinv.trace() * n * (-1.0);
    this->originalResistance = this->totalValue * (-1.);

    for (int round = 0; round < k; round++) {
      es.clear();
      gain.clear();
      g.forNodes([&](const NetworKit::node v) {
          if (focus_node != v && !this->g.hasEdge(focus_node, v) && !this->g.hasEdge(v, focus_node)) {
            es.push_back(Edge(focus_node, v));
            gain.push_back(
                n * (-1.0) *
                laplacianPseudoinverseTraceDifference(this->lpinv, focus_node, v));
          }
      });

      int bestIndex = std::max_element(gain.begin(), gain.end()) - gain.begin();
      double best = *std::max_element(gain.begin(), gain.end());

      updateLaplacianPseudoinverse(this->lpinv, es[bestIndex]);
      results.push_back(es[bestIndex]);
      this->totalValue += best;
    }
  }

  virtual double getResultValue() override { return this->totalValue * (-1.0); }
  virtual double getOriginalValue() override {
    return this->originalResistance;
  }
  virtual std::vector<Edge> getResultItems() override { return this->results; }
  virtual bool isValidSolution() { return true; }

private:
  Graph &g;
  count k;
  node focus_node;
  std::vector<NetworKit::Edge> results;
  double originalResistance = 0.;
  double totalValue = 0.;
  Eigen::MatrixXd lpinv;
};

class StGreedy final : public SubmodularGreedy<Edge> {
public:
  StGreedy(GreedyParams params) {
    this->g = params.g;
    this->n = g.numberOfNodes();
    this->k = params.k;
    this->focus_node = params.focus_node;

    // Compute pseudoinverse of laplacian
    Aux::Timer lpinvTimer;
    lpinvTimer.start();
    this->lpinv = laplacianPseudoinverse(g);
    lpinvTimer.stop();

    this->totalValue = this->lpinv.trace() * n * (-1.0);
    this->originalResistance = this->totalValue * (-1.);
  }

  virtual void addDefaultItems() {
    // Add edges to items of greedy
    std::vector<Edge> items;
    g.forNodes([&](const NetworKit::node v) {
      if (focus_node != v && !this->g.hasEdge(focus_node, v) && !this->g.hasEdge(v, focus_node)) {
          items.push_back(Edge(focus_node, v));
        }
    });

    this->addItems(items);
  }

  virtual double getResultValue() override { return this->totalValue * (-1.0); }

  virtual double getOriginalValue() override {
    return this->originalResistance;
  }

  virtual std::vector<NetworKit::Edge> getResultItems() override {
    return this->results;
  }

private:
  virtual double objectiveDifference(Edge e) override {
    auto i = e.u;
    auto j = e.v;
    return this->n * (-1.0) *
           laplacianPseudoinverseTraceDifference(this->lpinv, i, j);
  }

  virtual void useItem(Edge e) override {
    updateLaplacianPseudoinverse(this->lpinv, e);
    // this->g.addEdge(e.u, e.v);
  }

  Eigen::MatrixXd lpinv;

  double originalResistance = 0.;
  node focus_node;

  Graph g;
  int n;
};

class SimplStochJLT final : public AbstractOptimizer<Edge> {
public:
  SimplStochJLT(GreedyParams params) {

    this->g = params.g;
    this->n = g.numberOfNodes();
    this->focus_node = params.focus_node;
    this->k = params.k;
    this->epsilon = params.epsilon;
    gen = std::mt19937(Aux::Random::getSeed());

    // Compute pseudoinverse of laplacian
    this->lpinv = laplacianPseudoinverse(g);
    this->totalValue = this->lpinv.trace() * n * (-1.0);
    this->originalResistance = -1. * totalValue;
  }

  virtual double getResultValue() override { return this->totalValue * (-1.0); }

  virtual double getOriginalValue() override {
    return this->originalResistance;
  }

  virtual std::vector<NetworKit::Edge> getResultItems() override {
    return this->results;
  }

  virtual bool isValidSolution() override { return this->validSolution; }

  virtual void run() override {
    this->round = 0;
    this->validSolution = false;
    this->results.clear();
    struct edge_cmp {
      bool operator()(const Edge &lhs, const Edge &rhs) const {
        return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
      }
    };
    std::set<Edge, edge_cmp> resultSet;

    if (k + g.numberOfEdges() > (n * (n - 1) / 8 * 3)) {
      this->hasRun = true;
      std::cout << "Bad call to GreedySq, adding this many edges is not "
                   "supported! Attempting to have "
                << k + g.numberOfEdges() << " edges, limit is "
                << n * (n - 1) / 8 * 3;
      return;
    }

    for (int r = 0; r < k; r++) {
      double bestGain = -std::numeric_limits<double>::infinity();
      Edge bestEdge;

      int it = 0;

      do {
        // Collect nodes set for current round
        std::set<NetworKit::node> nodes;
        unsigned int s = (this->n - this->g.degree(this->focus_node) - this->round) / k * std::log(1.0 / epsilon) + 1;
        // if (s > k-round) { s = k - round; }
        if (s < 10) {
          s = 10;
        }

        // if (s < 4) { s = 4; }
        if (s > n / 2) {
          s = n / 2;
        }
        double min = std::numeric_limits<double>::infinity();

        std::vector<double> nodeWeights;
        if (it++ < 2) {
          double tr = lpinv.trace();
          for (int i = 0; i < n; i++) {
            double val = n * lpinv(i, i) + tr;
            if (val < min) {
              min = val;
            }
            nodeWeights.push_back(val);
          }
          for (auto &v : nodeWeights) {
            auto u = v - min;
            v = u * u;
          }
        } else {
          for (int i = 0; i < n; i++) {
            nodeWeights.push_back(1.0 / n);
          }
        }

        std::discrete_distribution<> distribution_nodes_heuristic(
            nodeWeights.begin(), nodeWeights.end());

        nodeWeights.clear();
        for (int i = 0; i < n; i++) {
          nodeWeights.push_back(1.0 / n);
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

        // Determine best edge between nodes from node set
        auto edgeValid = [&](node u, node v) {
          if (this->g.hasEdge(u, v))
            return false;
          if (resultSet.count(Edge(u, v)) > 0 ||
              resultSet.count(Edge(v, u)) > 0) {
            return false;
          }
          return true;
        };
        for (int i = 0; i < s; i++) {
          auto u = nodesVec[i];
          if (edgeValid(focus_node, u)) {
            double gain = objectiveDifference(Edge(focus_node, u));
            if (gain > bestGain) {
              bestEdge = Edge(focus_node, u);
              bestGain = gain;
            }
          }
        }
      } while (bestGain == -std::numeric_limits<double>::infinity());
      // Accept edge
      resultSet.insert(bestEdge);
      this->totalValue += bestGain;
      updateLaplacianPseudoinverse(this->lpinv, bestEdge);

      if (this->round == k - 1) {
        this->validSolution = true;
        break;
      }
      this->round++;
    }
    this->hasRun = true;
    this->results = {resultSet.begin(), resultSet.end()};
  }

private:
  virtual double objectiveDifference(Edge e) {
    auto i = e.u;
    auto j = e.v;
    return this->n * (-1.0) *
           laplacianPseudoinverseTraceDifference(this->lpinv, i, j);
  }

  Graph g;
  node focus_node;
  int n;
  Eigen::MatrixXd lpinv;
  std::mt19937 gen;

  bool validSolution = false;
  int round = 0;
  std::vector<Edge> results;
  double totalValue = 0.0;
  double originalResistance = 0.;
  double epsilon = 0.1;
  int k;
};

class SimplStoch : public StochasticGreedy<Edge> {
public:
  SimplStoch(GreedyParams params) {

    this->g = params.g;
    this->n = g.numberOfNodes();
    this->k = params.k;
    this->focus_node = params.focus_node;
    this->epsilon = params.epsilon;
    this->candidatesize = params.candidatesize;

    this->lpinv = laplacianPseudoinverse(g);
    this->totalValue = this->lpinv.trace() * n * (-1.0);
    this->originalResistance = -1. * this->totalValue;
  }

  virtual void addDefaultItems() override {

    // Add edges to items of greedy
    std::vector<Edge> items;

    size_t lim = std::numeric_limits<size_t>::max();
    size_t count;
    if (this->candidatesize == CandidateSetSize::small) {
      lim = std::ceil(std::log(this->n));
    } else {
    }
    for (size_t i = 0; i < this->n; i++) {
      count = 0;
      if (focus_node != i && !this->g.hasEdge(focus_node, i) && !this->g.hasEdge(i, focus_node)) {
          items.push_back(Edge(focus_node, i));
          count++;
        }
      if (count > lim) {
        break;
      }
    }

    this->addItems(items);
  }

  virtual std::vector<Edge> getResultItems() override { return this->results; }
  virtual double getResultValue() override { return this->totalValue * (-1.0); }

  virtual double getOriginalValue() override {
    return this->originalResistance;
  }

private:
  virtual double objectiveDifference(Edge e) override {
    return (-1.0) * laplacianPseudoinverseTraceDifference(lpinv, e.u, e.v) * n;
  }

  virtual void useItem(Edge e) override {
    updateLaplacianPseudoinverse(this->lpinv, e);
    // this->g.addEdge(e);
  }

  Eigen::MatrixXd lpinv;

  Graph g;
  node focus_node;
  int n;
  double originalResistance = 0.;
  CandidateSetSize candidatesize;
};

// template <class DynamicLaplacianSolver=LamgDynamicLaplacianSolver>
template <class DynamicLaplacianSolver>
class ColStoch : public StochasticGreedy<Edge> {
public:
  ColStoch(GreedyParams params) {
    this->g = params.g;
    this->n = g.numberOfNodes();
    this->focus_node = params.focus_node;
    this->k = params.k;
    this->epsilon = params.epsilon;

    this->totalValue = 0.;
    this->originalResistance = 0.;
    // solver.setup(g, 1.0e-6, std::ceil(n * std::sqrt(1. / (double)(k) *
    // std::log(1.0/epsilon))));
    if (this->k > 20)
      solver.setup(
          g, params.solverEpsilon,
          std::ceil(n * std::sqrt(1. / (double)(k)*std::log(1.0 / epsilon))));
    else
      solver.setup(g, params.solverEpsilon,
                   std::ceil(n * std::sqrt(std::log(1.0 / epsilon))));
  }

  virtual void addDefaultItems() override {
    // Add edges to items of greedy
    std::vector<Edge> items;
    g.forNodes([&](const NetworKit::node v) {
      if (focus_node != v && !this->g.hasEdge(focus_node, v) && !this->g.hasEdge(v, focus_node)) {
          items.push_back(Edge(focus_node, v));
        }
    });

    this->addItems(items);
  }

  virtual std::vector<Edge> getResultItems() override { return this->results; }
  virtual double getResultValue() override { return this->totalValue * (-1.0); }

  virtual double getOriginalValue() override {
    return this->originalResistance;
  }

  void run() override {
    this->round = 0;
    // this->totalValue = 0;
    this->validSolution = false;
    this->results.clear();

    if (this->items.size() == 0) {
      addDefaultItems();
    }

    bool candidatesLeft = true;
    std::mt19937 g(Aux::Random::getSeed());

    while (candidatesLeft) {
      this->initRound();

      std::priority_queue<ItemWrapper> R;
      unsigned int s =
          (unsigned int)(1.0 * this->N / k * std::log(1.0 / epsilon)) + 1;
      s = std::min(s, (unsigned int)this->items.size() - round);

      // Get a random subset of the items of size s.
      // Do this via selecting individual elements resp via shuffling,
      // depending on wether s is large or small.
      // ==========================================================================
      // Populating R set. What is the difference between small and large s?
      if (s > this->N / 4) { // This is not a theoretically justified estimate
        std::vector<unsigned int> allIndices =
            std::vector<unsigned int>(this->N);
        std::iota(allIndices.begin(), allIndices.end(), 0);
        std::shuffle(allIndices.begin(), allIndices.end(), g);

        auto itemCount = items.size();
        for (auto i = 0; i < itemCount; i++) {
          auto item = this->items[allIndices[i]];
          if (!item.selected) {
            R.push(item);
            if (R.size() >= s) {
              break;
            }
          }
        }
      } else {
        while (R.size() < s) {
          std::set<unsigned int> indicesSet;
          // TODO: Look into making this deterministic
          unsigned int v = std::rand() % this->N;
          if (indicesSet.count(v) == 0) {
            indicesSet.insert(v);
            auto item = this->items[v];
            if (!item.selected) {
              R.push(item);
            }
          }
        }
      }
      // ============================
      // Get top updated entry from R
      ItemWrapper c;
      while (true) {
        if (R.empty()) {
          candidatesLeft = false;
          break;
        } else {
          c = R.top();
          R.pop();
        }

        if (this->isItemAcceptable(c.item)) {
          if (c.lastUpdated == this->round) {
            break; // top updated entry found.
          } else {
            auto &item = this->items[c.index];
            c.value = this->objectiveDifference(c.item);
            item.value = c.value;
            c.lastUpdated = this->round;
            item.lastUpdated = this->round;
            R.push(c);
          }
        }
      }
      if (candidatesLeft) {
        this->results.push_back(c.item);
        this->totalValue += this->objectiveDifferenceExact(c.item); // this line is the only one changed compared to the superclass::run fn
        this->useItem(c.item);
        this->items[c.index].selected = true;

        if (this->checkSolution()) {
          this->validSolution = true;
          break;
        }
        this->round++;
      }
    }
    this->hasRun = true;
  }

private:
  virtual double objectiveDifference(Edge e) override {
    return solver.totalResistanceDifferenceApprox(e.u, e.v);
  }

  virtual double objectiveDifferenceExact(Edge e) {
    return solver.totalResistanceDifferenceExact(e.u, e.v);
  }

  virtual void useItem(Edge e) override { solver.addEdge(e.u, e.v); }

  DynamicLaplacianSolver solver;
  Graph g;
  node focus_node;
  int n;
  double originalResistance = 0.;
};

template <class DynamicLaplacianSolver>
class SpecStoch : public StochasticGreedy<Edge> {
public:
  SpecStoch(GreedyParams params) {
    this->g = params.g;
    this->n = g.numberOfNodes();
    this->focus_node = params.focus_node;
    this->k = params.k;
    this->epsilon = params.epsilon;
    this->ne = params.ne;
    this->updatePerRound = params.updatePerRound;
    this->diff = params.diff;
    this->candidatesize = params.candidatesize;

    if (this->k > 20)
      lap_solver.setup(
          g, params.solverEpsilon,
          std::ceil(n * std::sqrt(1. / (double)(k)*std::log(1.0 / epsilon))));
    else
      lap_solver.setup(g, params.solverEpsilon,
                       std::ceil(n * std::sqrt(std::log(1.0 / epsilon))));

    solver.setup(g, this->k, ne);

    solver.run_eigensolver();
    this->totalValue = 0.;
    this->originalResistance = totalValue;
    this->SpectralOriginalResistance =
        solver.SpectralToTalEffectiveResistance();
  }

  virtual void addDefaultItems() override {
    // Add edges to items of greedy
    std::vector<Edge> items;
    g.forNodes([&](const NetworKit::node v) {
      if (focus_node != v && !this->g.hasEdge(focus_node, v) && !this->g.hasEdge(v, focus_node)) {
          items.push_back(Edge(focus_node, v));
        }
    });

    this->addItems(items);
  }

  virtual std::vector<Edge> getResultItems() override { return this->results; }
  virtual double getResultValue() override { return this->totalValue * (-1.0); }

  virtual double getOriginalValue() override {
    return this->originalResistance;
  }

  double getReferenceResultValue() override {
    return this->ReferenceTotalValue;
  }

  double getReferenceOriginalResistance() override {
    return this->ReferenceOriginalResistance;
  }

  double getSpectralResultValue() override {
    this->SpectralTotalValue = solver.SpectralToTalEffectiveResistance();
    return this->SpectralTotalValue;
  }

  double getSpectralOriginalResistance() override {
    return this->SpectralOriginalResistance;
  }

  double getMaxEigenvalue() override { return solver.get_max_eigenvalue(); }

  void run() override {

    this->round = 0;
    this->validSolution = false;
    this->results.clear();

    if (this->items.size() == 0) {
      addDefaultItems();
    }
    bool candidatesLeft = true;
    std::mt19937 g(Aux::Random::getSeed());

    while (candidatesLeft) {
      this->initRound();

      std::priority_queue<ItemWrapper> R;
      unsigned int s =
          (unsigned int)(1.0 * this->N / k * std::log(1.0 / epsilon)) + 1;
      s = std::min(s, (unsigned int)this->items.size() - round);

      // Get a random subset of the items of size s.
      // Do this via selecting individual elements resp via shuffling,
      // depending on wether s is large or small.
      // ==========================================================================
      // Populating R set. What is the difference between small and large s?
      if (s > this->N / 4) { // This is not a theoretically justified estimate
        std::vector<unsigned int> allIndices =
            std::vector<unsigned int>(this->N);
        std::iota(allIndices.begin(), allIndices.end(), 0);
        std::shuffle(allIndices.begin(), allIndices.end(), g);

        auto itemCount = items.size();
        for (auto i = 0; i < itemCount; i++) {
          auto item = this->items[allIndices[i]];
          if (!item.selected) {
            R.push(item);
            if (R.size() >= s) {
              break;
            }
          }
        }
      } else {
        while (R.size() < s) {
          std::set<unsigned int> indicesSet;
          // TODO: Look into making this deterministic
          unsigned int v = std::rand() % this->N;
          if (indicesSet.count(v) == 0) {
            indicesSet.insert(v);
            auto item = this->items[v];
            if (!item.selected) {
              R.push(item);
            }
          }
        }
      }
      // ============================
      // Get top updated entry from R
      ItemWrapper c;
      while (true) {
        if (R.empty()) {
          candidatesLeft = false;
          break;
        } else {
          c = R.top();
          R.pop();
        }

        if (this->isItemAcceptable(c.item)) {
          if (c.lastUpdated == this->round) {
            break; // top updated entry found.
          } else {
            auto &item = this->items[c.index];
            c.value = this->objectiveDifference(c.item);
            item.value = c.value;
            c.lastUpdated = this->round;
            item.lastUpdated = this->round;
            R.push(c);
          }
        }
      }
      if (candidatesLeft) {
        this->results.push_back(c.item);
        this->totalValue += this->objectiveDifferenceExact(c.item);
        this->useItem(c.item);
        this->items[c.index].selected = true;

        if (this->checkSolution()) {
          this->validSolution = true;
          break;
        }
        this->round++;
      }
    }
    this->hasRun = true;
  }

private:
  virtual double objectiveDifference(Edge e) override {
    if (diff == 0)
      return solver.SpectralApproximationGainDifference1(e.u, e.v) * n;
    else if (diff == 1)
      return solver.SpectralApproximationGainDifference2(e.u, e.v) * n;
    else
      return solver.SpectralApproximationGainDifference3(e.u, e.v) * n;
  }

  virtual double objectiveDifferenceExact(Edge e) {
    return lap_solver.totalResistanceDifferenceExact(e.u, e.v);
  }

  virtual void useItem(Edge e) override {

    solver.addEdge(e.u, e.v);
    lap_solver.addEdge(e.u, e.v);

    if (!(this->round % updatePerRound))
      updateEigenpairs();
  }

  void cutOff() {
    ne = ceil(this->epsilon * n);
    assert(ne > 0 && ne <= n);
  }

  void updateEigenpairs() { solver.update_eigensolver(); }

  Graph g;
  node focus_node;
  int n;
  double originalResistance = 0.;
  SlepcAdapter solver;
  unsigned int ne = 1;
  unsigned int updatePerRound = 1;
  unsigned int diff = 1;
  CandidateSetSize candidatesize;

  // ----------------------------- //
  double ReferenceTotalValue = 0.0;
  double ReferenceOriginalResistance = 0.0;
  double SpectralTotalValue = 0.0;
  double SpectralOriginalResistance = 0.0;
  //
  DynamicLaplacianSolver lap_solver;
};

#endif // ROBUSTNESS_GREEDY_H
