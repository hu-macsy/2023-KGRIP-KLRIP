
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <vector>

#ifndef GREEDY_H
#define GREEDY_H

#include <networkit/auxiliary/Random.hpp>
#include <networkit/base/Algorithm.hpp>
#include <networkit/graph/Graph.hpp>

#include <omp.h>

template <class T> class AbstractOptimizer : public NetworKit::Algorithm {
public:
  virtual double getResultValue() = 0;
  virtual double getOriginalValue() = 0;
  virtual std::vector<T> getResultItems() = 0;
  virtual bool isValidSolution() = 0;
  virtual double getReferenceResultValue() { return 0.0; };
  virtual double getReferenceOriginalResistance() { return 0.0; };
  virtual double getSpectralResultValue() { return 0.0; };
  virtual double getSpectralOriginalResistance() { return 0.0; };
  virtual double getMaxEigenvalue() { return 0.0; };
};

template <class Item> struct _ItemWrapperType {
  Item item;
  double value;
  int lastUpdated;
};

template <class Item>
bool operator<(const _ItemWrapperType<Item> &left,
               const _ItemWrapperType<Item> &right) {
  return left.value < right.value ||
         (right.value == left.value && left.item < right.item);
}

template <class Item> class SubmodularGreedy : public AbstractOptimizer<Item> {
public:
  virtual void run() override;
  void set_k(int k) { this->k = k; }

  // To obtain results
  int getResultSize() { return round + 1; }
  std::vector<Item> getResultItems() { return results; }
  double getTotalValue() { return totalValue; }
  virtual bool isValidSolution() override { return validSolution; }

  void summarize() {
    std::cout << "Greedy Results Summary. ";
    if (!this->hasRun) {
      std::cout << "Not executed yet!";
    }
    std::cout << "Result Size: " << this->getResultSize() << std::endl;
    if (this->getResultSize() < 1000) {
      for (auto e : this->getResultItems()) {
        std::cout << "(" << e.u << ", " << e.v << "), ";
      }
    }
    std::cout << std::endl;
    std::cout << "Total Value: " << this->getTotalValue() << std::endl;
  }

protected:
  using ItemWrapper = _ItemWrapperType<Item>;

  virtual double objectiveDifference(Item c) = 0;
  virtual void useItem(Item i) = 0;
  virtual bool checkSolution() { return this->round == this->k - 1; };
  virtual bool isItemAcceptable(Item c) { return true; }
  virtual void initRound() {}
  virtual void addDefaultItems(){};

  void addItems(std::vector<Item> items);
  void resetItems();

  std::priority_queue<ItemWrapper> itemQueue;

  bool validSolution = false;
  int round = 0;
  std::vector<Item> results;
  double totalValue = 0.0;
  int k;
};

template <class Item>
void SubmodularGreedy<Item>::addItems(std::vector<Item> items) {
  unsigned int threads = omp_get_max_threads();
  std::vector<std::vector<ItemWrapper>> items_per_thread{
      threads, std::vector<ItemWrapper>{0}};

#pragma omp parallel for
  for (unsigned int ind = 0; ind < items.size(); ind++) {
    auto i = items[ind];
    ItemWrapper it{i, objectiveDifference(i), 0};
    items_per_thread[omp_get_thread_num()].push_back(it);
  }
  for (unsigned int i = 0; i < threads; i++) {
    for (auto &it : items_per_thread[i]) {
      itemQueue.push(it);
    }
  }
}

template <class Item> void SubmodularGreedy<Item>::resetItems() {
  this->itemQueue = std::priority_queue<ItemWrapper>();
}

template <class Item> void SubmodularGreedy<Item>::run() {
  this->round = 0;
  // this->totalValue = 0;

  // DEBUG(" >>> TOTALVALUE = ", this->totalValue, " <<< ");

  this->validSolution = false;
  this->results.clear();

  if (itemQueue.size() == 0) {
    addDefaultItems();
  }

  bool candidatesLeft = true;

  while (candidatesLeft) {
    this->initRound();

    // DEBUG("AFTER POPULATING PRIORITY QUEUE.");
    // Get top updated entry from queue
    ItemWrapper c;
    while (true) {
      if (itemQueue.empty()) {
        candidatesLeft = false;
        break;
      } else {
        c = itemQueue.top();
        itemQueue.pop();
      }

      if (this->isItemAcceptable(c.item)) {
        if (c.lastUpdated == this->round) {
          break; // top updated entry found.
        } else {
          c.value = this->objectiveDifference(c.item);
          // DEBUG(" TOP new value : ", c.value, " of edge = (", c.item.u, ", ",
          // c.item.v, ")");
          c.lastUpdated = this->round;
          itemQueue.push(c);
        }
      } // Else: dont put it back
    }
    if (candidatesLeft) {
      this->results.push_back(c.item);
      this->totalValue += c.value;
      DEBUG(" >>> TOTALVALUE = ", this->totalValue, " <<< ");
      DEBUG(" SELECTED value = ", c.value, " of edge = (", c.item.u, ", ",
            c.item.v, ")");
      this->useItem(c.item);

      if (this->checkSolution()) {
        this->validSolution = true;
        break;
      }
      this->round++;
    }
  }
  this->hasRun = true;
}

// Stochastic Greedy
// Implementation of
// Baharan Mirzasoleiman, Ashwinkumar Badanidiyuru, Amin Karbasi, Jan Vondrak,
// Andreas Krause:  Lazier Than Lazy Greedy. https://arxiv.org/abs/1409.7938

template <class Item> struct _ItemWrapperType2 {
  Item item;
  double value;
  int lastUpdated;
  unsigned int index;
  bool selected;
};

template <class Item>
bool operator<(const _ItemWrapperType2<Item> &left,
               const _ItemWrapperType2<Item> &right) {
  return left.value < right.value ||
         (right.value == left.value && left.item < right.item);
}

template <class Item> class StochasticGreedy : public AbstractOptimizer<Item> {
public:
  using ItemWrapper = _ItemWrapperType2<Item>;
  void set_k(int k) { this->k = k; }
  void set_epsilon(double epsilon) {
    assert(0.0 <= epsilon && epsilon <= 1.0);
    this->epsilon = epsilon;
  }

  virtual void run() override;

  // To obtain results
  int getResultSize() { return round + 1; }
  std::vector<Item> getResultItems() { return results; }
  double getTotalValue() { return totalValue; }
  virtual bool isValidSolution() override { return validSolution; }

  void printItems() {
    DEBUG("NUMBER OF ITEMS = ", items.size());
    for (auto e = 0; e < items.size(); e++) {
      DEBUG("ITEM:(", items[e].item.u, " ,", items[e].item.v,
            ") value = ", items[e].value,
            " lastUpdated = ", items[e].lastUpdated,
            " index = ", items[e].index, " selected = ", items[e].selected);
    }
  }

  void summarize() {
    std::cout << "Stochastic Submodular Greedy Results Summary. ";
    if (!this->hasRun) {
      std::cout << "Not executed yet!";
      return;
    }
    std::cout << "Result Size: " << this->getResultSize() << std::endl;
    for (auto e : this->getResultItems()) {
      std::cout << "(" << e.u << ", " << e.v << "), ";
    }
    std::cout << std::endl;
    std::cout << "Total Value: " << this->getTotalValue() << std::endl;
  }

protected:
  virtual double objectiveDifference(Item c) = 0;
  virtual void useItem(Item i) = 0;
  virtual bool checkSolution() { return this->round + 1 == this->k; };
  virtual bool isItemAcceptable(Item c) { return true; }
  virtual void initRound() {}
  virtual void addDefaultItems(){};

  void addItems(std::vector<Item> items);
  void resetItems();

  std::vector<ItemWrapper> items;

  int N;
  bool validSolution = false;
  int round = 0;
  std::vector<Item> results;
  double totalValue = 0.0;
  int k;
  double epsilon = 0.1;
};

template <class Item>
void StochasticGreedy<Item>::addItems(std::vector<Item> items_) {
  auto size = this->items.size();
  for (auto i = 0; i < items_.size(); i++) {
    auto it = items_[i];
    ItemWrapper qe{it, std::numeric_limits<double>::infinity(), -1,
                   (unsigned int)size + (unsigned int)i, false};
    items.push_back(qe);
  }
  this->N = items.size();
}

template <class Item> void StochasticGreedy<Item>::resetItems() {
  this->items = std::vector<ItemWrapper>();
}

template <class Item> void StochasticGreedy<Item>::run() {
  this->round = 0;
  // this->totalValue = 0;
  this->validSolution = false;
  this->results.clear();

  // DEBUG(" >>> TOTALVALUE = ", this->totalValue, " <<< ");

  if (items.size() == 0) {
    DEBUG(" >>> ADDING DEFAULT ITEMS <<< ");
    addDefaultItems();
  }

  DEBUG(" N = ", N);
  // DEBUG("PRINTING ITEMS (START).");
  // printItems();
  // DEBUG("PRINTING ITEMS (END).");
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
    if (s > N / 4) { // This is not a theoretically justified estimate
      std::vector<unsigned int> allIndices = std::vector<unsigned int>(N);
      std::iota(allIndices.begin(), allIndices.end(), 0);
      std::shuffle(allIndices.begin(), allIndices.end(), g);

      auto itemCount = items.size();
      for (auto i = 0; i < itemCount; i++) {
        auto item = this->items[allIndices[i]];
        if (!item.selected) {
          // DEBUG("ITEM:(", item.item.u, " ,",  item.item.v, ") value = ",
          // item.value, " lastUpdated = ", item.lastUpdated, " index = ",
          // item.index, " selected = ", item.selected);
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
        unsigned int v = std::rand() % N;
        if (indicesSet.count(v) == 0) {
          indicesSet.insert(v);
          auto item = this->items[v];
          if (!item.selected) {
            // DEBUG("ITEM:(", item.item.u, " ,",  item.item.v, ") value = ",
            // item.value, " lastUpdated = ", item.lastUpdated, " index = ",
            // item.index, " selected = ", item.selected);
            R.push(item);
          }
        }
      }
    }
    // ==========================================================================

    DEBUG("AFTER POPULATING PRIORITY QUEUE.");

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
          // DEBUG(" TOP :(", c.item.u, " ,",  c.item.v, ") value = ", c.value,
          // " lastUpdated = ", c.lastUpdated, " index = ", c.index, " selected
          // = ", c.selected);
          item.value = c.value;
          c.lastUpdated = this->round;
          item.lastUpdated = this->round;
          R.push(c);
        }
      } // Else: dont put it back

      // DEBUG("PRINTING ITEMS (START).");
      // printItems();
      // DEBUG("PRINTING ITEMS (END).");
    }
    if (candidatesLeft) {
      this->results.push_back(c.item);
      this->totalValue += c.value;
      // DEBUG(" >>> TOTALVALUE = ", this->totalValue, " <<< ");
      // DEBUG(" SELECTED value = ", c.value, " of edge = (", c.item.u, ", ",
      // c.item.v, ")");
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

#endif // GREEDY_H
