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
   virtual void reset_focus(const node& fn) override {
    this->focus_node = fn;
    this->hasRun = false;
  }

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
    // this->originalG = params.g;
    this->n = g.numberOfNodes();
    this->k = params.k;
    this->focus_node = params.focus_node;

    // Compute pseudoinverse of laplacian
    Aux::Timer lpinvTimer;
    lpinvTimer.start();
    this->lpinv = laplacianPseudoinverse(g);
    lpinvTimer.stop();
    this->originalLpinv = this->lpinv;

    this->totalValue = this->lpinv.trace() * n * (-1.0);
    this->originalResistance = this->totalValue * (-1.);
  }

  virtual void reset_focus(const node& fn) override {
    // this->g = this->originalG;
    this->focus_node = fn;
    this->lpinv = this->originalLpinv;
    this->resetItems();
    this->hasRun = false;

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
  Eigen::MatrixXd originalLpinv;
  

  double originalResistance = 0.;
  node focus_node;

  Graph g;
  // Graph originalG;
  int n;
};

class SimplStoch : public StochasticGreedy<Edge> {
public:
  SimplStoch(GreedyParams params) {

    this->g = params.g;
    // this->originalG = params.g;
    this->n = g.numberOfNodes();
    this->k = params.k;
    this->focus_node = params.focus_node;
    this->epsilon = params.epsilon;
    this->candidatesize = params.candidatesize;

    this->lpinv = laplacianPseudoinverse(g);
    this->originalLpinv = this->lpinv;
    this->totalValue = this->lpinv.trace() * n * (-1.0);
    this->originalResistance = -1. * this->totalValue;
  }

  virtual void reset_focus(const node& fn) override {
    // this->g = this->originalG;
    this->focus_node = fn;
    this->lpinv = this->originalLpinv;
    this->resetItems();
    this->hasRun = false;

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
  Eigen::MatrixXd originalLpinv;

  Graph g;
  // Graph originalG;
  node focus_node;
  int n;
  double originalResistance = 0.;
  CandidateSetSize candidatesize;
};

template <class DynamicLaplacianSolver>
class simplStochDyn : public StochasticGreedy<Edge> {
public:
  simplStochDyn(GreedyParams params) {
    this->g = params.g;
    // this->originalG = params.g;
    this->n = g.numberOfNodes();
    this->focus_node = params.focus_node;
    this->k = params.k;
    this->epsilon = params.epsilon;
    this->solverEpsilon = params.solverEpsilon;

    this->totalValue = 0.;
    this->originalResistance = 0.;
    if (this->k > 20)
      solver.setup(
          g, params.solverEpsilon,
          std::ceil(n * std::sqrt(1. / (double)(k)*std::log(1.0 / epsilon))));
    else
      solver.setup(g, params.solverEpsilon,
                   std::ceil(n * std::sqrt(std::log(1.0 / epsilon))));
  }

  virtual void reset_focus(const node& fn) override {
    // this->g = this->originalG;
    this->focus_node = fn;

    solver.~DynamicLaplacianSolver();
    new (&solver) DynamicLaplacianSolver();
    if (this->k > 20)
      solver.setup(
          g, this->solverEpsilon,
          std::ceil(n * std::sqrt(1. / (double)(k)*std::log(1.0 / epsilon))));
    else
      solver.setup(g, this->solverEpsilon,
                   std::ceil(n * std::sqrt(std::log(1.0 / epsilon))));

    this->resetItems();
    this->hasRun = false;
    this->totalValue = 0.;
    this->originalResistance = 0.;
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
  Graph originalG;
  node focus_node;
  int n;
  double originalResistance = 0.;
  double solverEpsilon;
};

template <class DynamicLaplacianSolver>
class SpecStoch : public StochasticGreedy<Edge> {
public:
  SpecStoch(GreedyParams params) : 
    g(params.g),
    n(g.numberOfNodes()),
    focus_node(params.focus_node),
    solverEpsilon(params.solverEpsilon),
    ne(params.ne),
    updatePerRound(params.updatePerRound),
    diff(params.diff),
    candidatesize(params.candidatesize),
    solver(g, params.k, ne),
    originalSolver(solver)

  // double originalResistance = 0.;

  // // ----------------------------- //
  // double ReferenceTotalValue = 0.0;
  // double ReferenceOriginalResistance = 0.0;
  // double SpectralTotalValue = 0.0;
  // double SpectralOriginalResistance = 0.0;
  // //
  // DynamicLaplacianSolver lap_solver;
  {
    this->k = params.k;
    this->epsilon = params.epsilon;

    if (this->k > 20)
      lap_solver.setup(
          g, params.solverEpsilon,
          std::ceil(n * std::sqrt(1. / (double)(k)*std::log(1.0 / epsilon))));
    else
      lap_solver.setup(g, params.solverEpsilon,
                       std::ceil(n * std::sqrt(std::log(1.0 / epsilon))));

    // solver.setup(g, this->k, ne);

    // solver.run_eigensolver();

    // DEBUG("Assign solver copy");
    // this->originalSolver = solver;

    this->totalValue = 0.;
    this->originalResistance = totalValue;
    this->SpectralOriginalResistance =
        solver.SpectralToTalEffectiveResistance();
  }

  virtual void reset_focus(const node& fn) override {
    this->focus_node = fn;
    this->solver = this->originalSolver;

    lap_solver.~DynamicLaplacianSolver();
    new (&lap_solver) DynamicLaplacianSolver();
    if (this->k > 20)
      lap_solver.setup(
          g, this->solverEpsilon,
          std::ceil(n * std::sqrt(1. / (double)(k)*std::log(1.0 / epsilon))));
    else
      lap_solver.setup(g, this->solverEpsilon,
                       std::ceil(n * std::sqrt(std::log(1.0 / epsilon))));
    this->resetItems();
    this->hasRun = false;

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
  double solverEpsilon;
  DynamicLaplacianSolver lap_solver;
  SlepcAdapter solver;
  SlepcAdapter originalSolver;
};

#endif // ROBUSTNESS_GREEDY_H
