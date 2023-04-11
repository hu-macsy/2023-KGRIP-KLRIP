// collections of test functions moved from main to this file
#ifndef ROBUSTNESS_TESTS_H
#define ROBUSTNESS_TESTS_H

#include <chrono>
#include <exception>
#include <iostream>
#include <iterator>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cassert>
#include <cmath>
#include <cstdlib>

#include <Eigen/Dense>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/centrality/ApproxElectricalCloseness.hpp>
#include <networkit/components/ConnectedComponents.hpp>
#include <networkit/generators/BarabasiAlbertGenerator.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/generators/HyperbolicGenerator.hpp>
#include <networkit/generators/WattsStrogatzGenerator.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphTools.hpp>
#include <networkit/io/EdgeListReader.hpp>
#include <networkit/io/GMLGraphReader.hpp>
#include <networkit/io/NetworkitBinaryReader.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/numerics/Preconditioner/DiagonalPreconditioner.hpp>

#include <networkit/auxiliary/Log.hpp>
#include <slepceps.h>

#include <greedy.hpp>
#include <laplacian.hpp>
#include <robustnessGreedy.hpp>
#include <robustnessUSTGreedy.hpp>
#include <slepc_adapter.hpp>

void testLaplacian(int seed, bool verbose = false) {
  // Create Example Graphs
  Aux::Random::setSeed(seed, false);

  // Test Laplacian pinv approximation
  Graph G = Graph();
  G.addNodes(4);
  G.addEdge(0, 1);
  G.addEdge(0, 2);
  G.addEdge(1, 2);
  G.addEdge(1, 3);
  G.addEdge(2, 3);

  auto diagonal = approxLaplacianPseudoinverseDiagonal(G);

  assert(std::abs(diagonal(0) - 0.3125) < 0.05);
  assert(std::abs(diagonal(1) - 0.1875) < 0.05);
  assert(std::abs(diagonal(2) - 0.1876) < 0.05);
  assert(std::abs(diagonal(3) - 0.3125) < 0.05);

  // Test Laplacian pinv update formulas
  auto Lmat = laplacianMatrix<Eigen::MatrixXd>(G);
  auto lpinv = laplacianPseudoinverse(G);
  auto diffvec = laplacianPseudoinverseColumnDifference(lpinv, 0, 3, 0);
  if (verbose) {
    std::cout << "Laplacian: \n" << Lmat << "\n";
    std::cout << "Laplacian pinv: \n" << lpinv << "\n";
  }
  std::vector<double> expected{-0.125, 0.0, 0.0, 0.125};
  for (int i = 0; i < 4; i++) {
    assert(std::abs(diffvec(i) - expected[i]) < 0.001);
  }

  if (verbose) {
    std::cout << "Laplacian trace difference upon adding (0,3): "
              << laplacianPseudoinverseTraceDifference(lpinv, 0, 3) << "\n";
    std::cout
        << "Laplacian trace difference upon removing (0,3) and adding (0,3): "
        << laplacianPseudoinverseTraceDifference(
               lpinv, std::vector<Edge>{Edge(0, 3), Edge(0, 3)}, {1.0, -1.0})
        << "\n";
    std::cout
        << "Laplacian trace difference upon removing (0,3) and adding (0,1): "
        << laplacianPseudoinverseTraceDifference(
               lpinv, std::vector<Edge>{Edge(0, 3), Edge(0, 1)}, {1.0, -1.0})
        << "\n";
  }
  assert(std::abs(laplacianPseudoinverseTraceDifference(lpinv, 0, 3) + 0.25) <
         0.001);
  assert(std::abs(laplacianPseudoinverseTraceDifference(
             lpinv, std::vector<Edge>{Edge(0, 3), Edge(0, 3)}, {1.0, -1.0})) <
         0.001);
  assert(std::abs(laplacianPseudoinverseTraceDifference(
             lpinv, std::vector<Edge>{Edge(0, 3), Edge(0, 1)}, {1.0, -1.0})) <
         0.001);
  // assert(std::abs(laplacianPseudoinverseTraceDifference(col0, 0, col3, 3) +
  // 0.25) < 0.001);

  /*
  Graph largeG = NetworKit::BarabasiAlbertGenerator(10, 1000, 2).generate();
  int pivot;
  diagonal = approxLaplacianPseudoinverseDiagonal(largeG, 0.5);
  std::cout << diagonal(5);
  lpinv = laplacianPseudoinverse(largeG);
  std::cout << " " << lpinv(5, 5);
  */
}

int testEigensolver(NetworKit::Graph const &g, int count, bool verbose = false,
                    bool expe = false) {
  Time beforeInit;
  auto n = g.numberOfNodes();
  if (verbose) {
    std::cout << " Graph =  (" << n << "," << g.numberOfEdges() << ")\n";
    std::cout << " count =  " << count << "\n";
  }

  beforeInit = std::chrono::high_resolution_clock::now();
  SlepcAdapter solver(g, 0, count);
  // if(verbose) { solver.info_eigensolver(); }
  auto t = std::chrono::high_resolution_clock::now();
  auto duration = t - beforeInit;
  using scnds = std::chrono::duration<float, std::ratio<1, 1>>;
  double *vectors = solver.get_eigenpairs();
  double *values = solver.get_eigenvalues();
  PetscInt nconv = solver.get_nconv();
  std::cout << "  Time:    "
            << std::chrono::duration_cast<scnds>(duration).count() << "\n";

  auto maxDegree = [&]() -> NetworKit::count {
    NetworKit::count maxDeg = 0;
    g.forNodes([&](const node u) {
      const auto degU = g.degree(u);
      if (degU > maxDeg) {
        maxDeg = degU;
      }
    });
    return maxDeg;
  };

  auto minDegree = [&]() -> NetworKit::count {
    NetworKit::count minDeg = std::numeric_limits<NetworKit::count>::max();
    g.forNodes([&](const node u) {
      const auto degU = g.degree(u);
      if (degU < minDeg) {
        minDeg = degU;
      }
    });
    return minDeg;
  };

  if (verbose) {
    assert(values);
    std::cout << " eigenvalues are:\n [ ";
    for (int i = 0; i < count + 1; i++)
      std::cout << values[i] << " ";
    std::cout << "]\n";

    // for (int i = 0 ; i < count; i++) {
    //   std::cout << "Vector " << i << " " ;
    //   for(int j = 0; j < n; j++) {
    // 	std::cout << *(vectors + i + j*nconv ) << " ";
    //   }
    //   std::cout << "\n";
    // }
    std::cout << " max degree = " << maxDegree() << "\n";
    std::cout << " min degree = " << minDegree() << "\n";
  }

  if (expe) {

    auto lpinv = laplacianPseudoinverse(g);
    NetworKit::count seeds = 1;
    // int sizes[] =
    // {1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800};
    // int s = 26;
    // if (nconv < 800) {
    //   std::cout << "Warning: Eigensolver did not converged (" << nconv <<")
    //   \n"; s = 18 + (nconv / 100);
    // }

    int sizes[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                   31, 32, 33, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43};
    int s = 43;
    std::cout << "converged =" << nconv << "\n";
    // if (nconv < 800) {
    //   std::cout << "Warning: Eigensolver did not converged (" << nconv <<")
    //   \n"; s = 18 + (nconv / 100);
    // }

    double *gain1 = (double *)calloc(1, s * sizeof(double));
    double *gain2 = (double *)calloc(1, s * sizeof(double));
    double *gain3 = (double *)calloc(1, s * sizeof(double));
    double reference = 0.0;

    for (NetworKit::count i = 0; i < seeds; i++) {
      NetworKit::node u, v;
      Aux::Random::setSeed(42, false);
      do {
        u = GraphTools::randomNode(g);
        v = GraphTools::randomNode(g);
      } while (g.hasEdge(u, v));
      std::cout << "Edge (" << u << "," << v << ") \n";
      for (NetworKit::count e = 0; e < s; e++) {
        if (sizes[e] <= nconv) {
          std::cout << "Eigencount = " << sizes[e] << " \n";
          gain2[e] = n * solver.SpectralApproximationGainDifference2(
                             u, v, vectors, values, sizes[e]);
          gain1[e] = n * solver.SpectralApproximationGainDifference1(
                             u, v, vectors, values, sizes[e]);
          gain3[e] = n * solver.SpectralApproximationGainDifference3(
                             u, v, vectors, values, sizes[e]);
          reference =
              n * (-1.0) * laplacianPseudoinverseTraceDifference(lpinv, u, v);
        }
      }
    }
    // take mean
    for (NetworKit::count e = 0; e < s; e++) {
      gain1[e] = gain1[e] / seeds;
      gain2[e] = gain2[e] / seeds;
      gain3[e] = gain3[e] / seeds;
    }
    reference = reference / seeds;
    std::cout << "  Reference Gain:  " << reference << "\n";
    std::cout << "  Spectral Gain1:  ";
    for (NetworKit::count e = 0; e < s; e++)
      std::cout << gain1[e] << " ";
    std::cout << "\n";
    std::cout << "  Spectral Gain2:  ";
    for (NetworKit::count e = 0; e < s; e++)
      std::cout << gain2[e] << " ";
    std::cout << "\n";
    std::cout << "  Spectral Gain3:  ";
    for (NetworKit::count e = 0; e < s; e++)
      std::cout << gain3[e] << " ";
    std::cout << "\n";
    std::cout << "  Eigenpairs:  ";
    for (NetworKit::count e = 0; e < s; e++)
      std::cout << sizes[e] << " ";
    std::cout << "\n";

    // std::cout << "  Reference KI:  " << lpinv.trace() * n << "\n";
    // std::cout << "  Spectral KI:  ";
    // for (NetworKit::count e = 0; e < s; e++)
    //   std::cout << solver.SpectralToTalEffectiveResistance(sizes[e]) << " ";
    // std::cout << "\n";
  }

  // std::cout << "  Reference KI:  " << lpinv.trace() * n << "\n";
  // std::cout << "  Spectral KI:  ";
  // std::cout << solver.SpectralToTalEffectiveResistance() << " ";
  // std::cout << "\n";

  return 0;
}

/*
void testSampleUSTWithEdge() {
        omp_set_num_threads(1);
        Graph G = Graph();
        G.addNodes(4);
        G.addEdge(0, 1);
        G.addEdge(0, 2);
        G.addEdge(1, 2);
        G.addEdge(1, 3);
        G.addEdge(2, 3);
        NetworKit::ApproxElectricalCloseness apx(G);
        apx.run();
        node a = 1;
        node b = 2;
        apx.testSampleUSTWithEdge(a, b);
}
*/

void testDynamicColumnApprox() {
  const double eps = 0.1;

  // for (int seed : {1, 2, 3}) {
  int seed = 1;
  count n = 30;
  Aux::Random::setSeed(seed, true);

  auto G = HyperbolicGenerator(n, 4, 3).generate();
  G = ConnectedComponents::extractLargestConnectedComponent(G, true);
  n = G.numberOfNodes();

  // Create a biconnected component with size 2.
  G.addNodes(2);
  G.addEdge(n - 1, n);
  G.addEdge(n, n + 1);

  /*
  Graph G = Graph();
  G.addNodes(4);
  G.addEdge(0, 1);
  G.addEdge(1, 2);
  G.addEdge(2, 3);
  */
  ApproxElectricalCloseness apx(G);
  apx.run();
  auto diag = apx.getDiagonal();
  auto gt = apx.computeExactDiagonal(1e-12);
  G.forNodes([&](node u) { assert(std::abs(diag[u] - gt[u]) < eps); });
  assert(apx.scores().size() == G.numberOfNodes());
  n = G.numberOfNodes();

  G.forNodes([&](node u) {
    auto col = apx.approxColumn(u);
    auto exactCol = apx.computeExactColumn(u);
    G.forNodes([&](node v) {
      if (std::abs(col[v] - exactCol[v]) > 2 * eps)
        std::cout << "C " << u << ", " << v << ": " << col[v] - exactCol[v]
                  << std::endl;
    });
  });

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> distN(0, n - 1);

  node a;
  node b;

  for (int i = 0; i < 20; i++) {
    do {
      a = distN(rng);
      b = distN(rng);
    } while (G.hasEdge(a, b) || a == b);

    G.addEdge(a, b);
    apx.edgeAdded(a, b);

    std::cout << "Edge added " << a << ", " << b << std::endl;

    diag = apx.getDiagonal();
    gt = apx.computeExactDiagonal(1e-12);
    G.forNodes([&](node u) {
      if (std::abs(diag[u] - gt[u]) > eps) {
        std::cout << "Diagonal off: " << u << "err: " << diag[u] - gt[u]
                  << " rnd: " << i << " seed: " << seed << std::endl;
      }
    });
    assert(apx.scores().size() == G.numberOfNodes());

    G.forNodes([&](node u) {
      auto col = apx.approxColumn(u);
      auto exactCol = apx.computeExactColumn(u);
      G.forNodes([&](node v) {
        if (std::abs(col[v] - exactCol[v]) > eps) {
          std::cout << "Column off: " << u << ", " << v << ": "
                    << col[v] - exactCol[v] << " rnd: " << i << " seed:" << seed
                    << std::endl;
        }
      });
    });
  }
  //}
}

void testRobustnessSubmodularGreedy() {

  Graph G1;
  G1.addNodes(4);
  G1.addEdge(0, 1);
  G1.addEdge(0, 2);
  G1.addEdge(1, 2);
  G1.addEdge(1, 3);
  G1.addEdge(2, 3);

  Graph G2;
  G2.addNodes(6);
  std::vector<std::pair<unsigned long, unsigned long>> edges = {
      {0, 1}, {0, 2}, {1, 3}, {2, 3}, {1, 2}, {1, 4}, {3, 5}, {4, 5}};
  for (auto p : edges) {
    G2.addEdge(p.first, p.second);
  }

  GreedyParams args(G2, 2);
  StGreedy rg2(args);
  rg2.run();
  assert(std::abs(rg2.getTotalValue() - 4.351) < 0.01);

  // rg2.summarize();

  Graph G3;
  G3.addNodes(14);
  for (size_t i = 0; i < 14; i++) {
    G3.addEdge(i, (i + 1) % 14);
  }
  G3.addEdge(4, 13);
  G3.addEdge(5, 10);

  GreedyParams args2(G3, 4);
  StGreedy rg3(args2);
  rg3.run();
  assert(std::abs(rg3.getTotalValue() - 76.789) < 0.01);
  // rg3.summarize();

  // rgd3.summarize();
}

void testRobustnessStochasticGreedySpectral() {

  Graph G1;
  G1.addNodes(4);
  G1.addEdge(0, 1);
  G1.addEdge(0, 2);
  G1.addEdge(1, 2);
  G1.addEdge(1, 3);
  G1.addEdge(2, 3);

  Graph G2;
  G2.addNodes(6);
  std::vector<std::pair<unsigned long, unsigned long>> edges = {
      {0, 1}, {0, 2}, {1, 3}, {2, 3}, {1, 2}, {1, 4}, {3, 5}, {4, 5}};
  for (auto p : edges) {
    G2.addEdge(p.first, p.second);
  }

  GreedyParams args(G2, 2);
  SpecStoch<SparseLUSolver> rg2(args);
  // createSpecific<RobustnessRandomAveraged<SparseLUSolver>>()
  rg2.run();
  assert(std::abs(rg2.getTotalValue() - 4.351) < 0.01);

  // rg2.summarize();

  Graph G3;
  G3.addNodes(14);
  for (size_t i = 0; i < 14; i++) {
    G3.addEdge(i, (i + 1) % 14);
  }
  G3.addEdge(4, 13);
  G3.addEdge(5, 10);

  GreedyParams args2(G3, 4);
  SpecStoch<SparseLUSolver> rg3(args2);
  rg3.run();
  assert(std::abs(rg3.getTotalValue() - 76.789) < 0.01);
  // rg3.summarize();

  // rgd3.summarize();
}

void testJLT(NetworKit::Graph g, std::string instanceFile, int k) {
  int n = g.numberOfNodes();

  JLTLUSolver jlt_solver;
  SparseLUSolver reference_solver;

  double epsilon = 0.5;
  jlt_solver.setup(g, epsilon, n);
  reference_solver.setup(g, 0.1);

  std::mt19937 gen(1);
  std::uniform_int_distribution<> distrib_n(0, n - 1);

  /*
  for (int i = 0 ; i < 5; i++) {
          node u = distrib(gen);
          node v = distrib(gen);
          if (!g.hasEdge(u, v)) {
                  g.addEdge(u, v);
                  jlt_solver.addEdge(u, v);
                  reference_solver.addEdge(u, v);
          }
  }
  */

  std::cout << "Runs: \n";
  std::cout << "- Instance: '" << instanceFile << "'\n";
  std::cout << "  Nodes: " << n << "\n";
  std::cout << "  Edges: " << g.numberOfEdges() << "\n";

  std::cout << "  JLT-Test: true \n";
  std::cout << "  Rel-Errors: [";

  for (int i = 0; i < 1000; i++) {
    node u = distrib_n(gen);
    node v = distrib_n(gen);
    if (!g.hasEdge(u, v) && u != v) {
      auto approx_gain = jlt_solver.totalResistanceDifferenceApprox(u, v);
      auto reference_gain =
          reference_solver.totalResistanceDifferenceExact(u, v);
      auto rel_error = std::abs(reference_gain - approx_gain) / reference_gain;
      std::cout << rel_error << ", ";
    }
  }
  std::cout << "] \n";

  std::cout << "  Rel-Errors-2: [";

  JLTLUSolver jlt_solver2;
  SparseLUSolver reference_solver2;

  epsilon = 0.75;
  int c = std::max(n / std::sqrt(2) / k * std::log2(1. / 0.9), 3.);
  jlt_solver2.setup(g, epsilon, c);
  std::uniform_int_distribution<> distrib_c(0, c - 1);
  std::set<int> node_subset;
  while (node_subset.size() < c) {
    node_subset.insert(distrib_n(gen));
  }
  std::vector<NetworKit::node> node_subset_vec(node_subset.begin(),
                                               node_subset.end());
  assert(node_subset_vec.size() == c);

  for (int i = 0; i < 1000; i++) {
    node u = node_subset_vec[distrib_c(gen)];
    node v = node_subset_vec[distrib_c(gen)];
    if (!g.hasEdge(u, v) && u != v) {
      auto approx_gain = jlt_solver.totalResistanceDifferenceApprox(u, v);
      auto reference_gain =
          reference_solver.totalResistanceDifferenceExact(u, v);
      auto rel_error = std::abs(reference_gain - approx_gain) / reference_gain;
      std::cout << rel_error << ", ";
    }
  }

  std::cout << "] \n";
}

void testSolverSetupTime(const Graph& g) {
    using scnds = std::chrono::duration<float, std::ratio<1, 1>>;

    auto beforeSlepcAdapter = std::chrono::high_resolution_clock::now();
    SlepcAdapter slepc_adapter(g, 100, 100);
    auto afterSlepcAdapter = std::chrono::high_resolution_clock::now();
    
    std::cout << "SlepcAdapter init with k=100, ne=100: " << std::chrono::duration_cast<scnds>(afterSlepcAdapter - beforeSlepcAdapter).count() << "\n";


    auto beforeLamg = std::chrono::high_resolution_clock::now();
    LamgDynamicLaplacianSolver lamgsolver;
    lamgsolver.setup(g, 0.9, std::ceil(g.numberOfNodes() * std::sqrt(1. / (double)(100)*std::log(1.0 / 0.9))));
    auto afterLamg = std::chrono::high_resolution_clock::now();
    
    std::cout << "Lamg init with k=100, eps=0.9: " << std::chrono::duration_cast<scnds>(afterLamg - beforeLamg).count() << "\n";

    auto beforeLU = std::chrono::high_resolution_clock::now();
    SparseLUSolver lusolver;
    lusolver.setup(g, 0.9, std::ceil(g.numberOfNodes() * std::sqrt(1. / (double)(100)*std::log(1.0 / 0.9))));
    auto afterLU = std::chrono::high_resolution_clock::now();
    
    std::cout << "LU init with k=100, eps=0.9: " << std::chrono::duration_cast<scnds>(afterLU - beforeLU).count() << "\n";

    auto beforeJLTLamg = std::chrono::high_resolution_clock::now();
    JLTLamgSolver jltlamgsolver;
    jltlamgsolver.setup(g, 0.9, std::ceil(g.numberOfNodes() * std::sqrt(1. / (double)(100)*std::log(1.0 / 0.9))));
    auto afterJLTLamg = std::chrono::high_resolution_clock::now();
    
    std::cout << "Lamg JLT init with k=100, eps=0.9: " << std::chrono::duration_cast<scnds>(afterJLTLamg - beforeJLTLamg).count() << "\n";

    auto beforeJLTLU = std::chrono::high_resolution_clock::now();
    JLTLUSolver jltlusolver;
    jltlusolver.setup(g, 0.9, std::ceil(g.numberOfNodes() * std::sqrt(1. / (double)(100)*std::log(1.0 / 0.9))));
    auto afterJLTLU = std::chrono::high_resolution_clock::now();
    
    std::cout << "LU JLT init with k=100, eps=0.9: " << std::chrono::duration_cast<scnds>(afterJLTLU - beforeJLTLU).count() << "\n";

    auto beforeAPX = std::chrono::high_resolution_clock::now();
    auto apx = std::make_unique<ApproxElectricalCloseness>(g, 10);
    apx->run();
    auto diag = apx->getDiagonal();
    auto afterAPX = std::chrono::high_resolution_clock::now();
    
    std::cout << "ApproxElectricalCloseness init with k=100, eps=0.9: " << std::chrono::duration_cast<scnds>(afterAPX - beforeAPX).count() << "\n";

    // copy timing
     SlepcAdapter slepc_adapter_copy(g, 1, 1);
    auto beforeSlepcAdapterCopy = std::chrono::high_resolution_clock::now();
    slepc_adapter_copy = slepc_adapter;
    auto afterSlepcAdapterCopy = std::chrono::high_resolution_clock::now();
    std::cout << "SlepcAdapter copy assign: " << std::chrono::duration_cast<scnds>(afterSlepcAdapterCopy - beforeSlepcAdapterCopy).count() << "\n";

    LamgDynamicLaplacianSolver lamgcopy;
    auto beforeLamgCopy = std::chrono::high_resolution_clock::now();
    lamgcopy = lamgsolver;
    auto afterLamgCopy = std::chrono::high_resolution_clock::now();
    std::cout << "Lamg copy assign: " << std::chrono::duration_cast<scnds>(afterLamgCopy - beforeLamgCopy).count() << "\n";

    JLTLamgSolver jltlamgcopy;
    auto beforeJLTLamgCopy = std::chrono::high_resolution_clock::now();
    jltlamgcopy = jltlamgsolver;
    auto afterJLTLamgCopy = std::chrono::high_resolution_clock::now();
    
    std::cout << "Lamg JLT copy assign: " << std::chrono::duration_cast<scnds>(afterJLTLamgCopy - beforeJLTLamgCopy).count() << "\n";

    SparseLUSolver lucopy;
    auto beforeLUCopy = std::chrono::high_resolution_clock::now();
    lucopy = lusolver;
    auto afterLUCopy = std::chrono::high_resolution_clock::now();
    
    std::cout << "LU copy assign: " << std::chrono::duration_cast<scnds>(afterLUCopy - beforeLUCopy).count() << "\n";

    JLTLUSolver jltlucopy;
    auto beforeJLTLUCopy = std::chrono::high_resolution_clock::now();
    jltlucopy = jltlusolver;
    auto afterJLTLUCopy = std::chrono::high_resolution_clock::now();
    
    std::cout << "LU JLT copy assign: " << std::chrono::duration_cast<scnds>(afterJLTLUCopy - beforeJLTLUCopy).count() << "\n";

    auto beforeAPXCopy = std::chrono::high_resolution_clock::now();
    auto apxcopy = std::make_unique<ApproxElectricalCloseness>(*apx);
    auto afterAPXCopy = std::chrono::high_resolution_clock::now();
    
    std::cout << "ApproxElectricalCloseness copy assign: " << std::chrono::duration_cast<scnds>(afterAPXCopy - beforeAPXCopy).count() << "\n";

}

#endif