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

#include <dynamicLaplacianSolver/LamgDynamicLaplacianSolver.hpp>
#include <dynamicLaplacianSolver/JLTLamgSolver.hpp>
#include <dynamicLaplacianSolver/JLTSolver.hpp>
#include <dynamicLaplacianSolver/SparseLUSolver.hpp>
#include <greedy.hpp>
#include <laplacian.hpp>
#include <robustnessGreedy.hpp>
#include <robustnessUSTGreedy.hpp>
#include <slepc_adapter.hpp>
#include <tests.hpp>

//#include <robustnessSimulatedAnnealing.hpp>

#include <algorithm>

using namespace NetworKit;

typedef decltype(std::chrono::high_resolution_clock::now()) Time;

enum class LinAlgType { none, lu, lamg, jlt_lu_sparse, jlt_lamg };

enum class AlgorithmType {
  none,
  stGreedy,
  simplStoch,
  specStoch,
  exhaustive,
  simplStochDyn,
  colStoch,
};

class RobustnessExperiment {
public:
  NetworKit::count k;
  NetworKit::count n;
  double epsilon;
  double epsilon2;
  NetworKit::Graph g;
  NetworKit::Graph g_cpy;
  LinAlgType linalg;
  AlgorithmType alg;
  bool verify_result = false;
  bool verbose = false;
  bool vv = false;
  Time beforeInit;
  int seed;
  std::string name;
  unsigned int threads = 1;
  unsigned int updatePerRound = 1;
  unsigned int diff = 1; // TODO: change to default value
  unsigned int ne = 1;
  std::unique_ptr<GreedyParams> params;
  HeuristicType heuristic;
  CandidateSetSize candidatesize;
  bool always_use_known_columns_as_candidates = false;
  unsigned int num_focus_nodes = 1;
  int focus_node_seed;

  std::vector<NetworKit::Edge> edges;
  double resultResistance;
  double originalResistance;

  std::string algorithmName;
  std::string call;
  std::string instanceFile;

  std::unique_ptr<AbstractOptimizer<NetworKit::Edge>> greedy;

  void createGreedy() {
    if (alg == AlgorithmType::stGreedy) {
      algorithmName = "stGreedy";
      createSpecific<StGreedy>();
    } else if (alg == AlgorithmType::simplStoch) {
      algorithmName = "simplStoch";
      createSpecific<SimplStoch>();
    } else if (alg == AlgorithmType::colStoch) {
      algorithmName = "colStoch";
      createLinAlgGreedy<ColStoch>();
    } else if (alg == AlgorithmType::simplStochDyn) {
      algorithmName = "simplStochDyn";
      createLinAlgGreedy<simplStochDyn>();
    } else if (alg == AlgorithmType::specStoch) {
      algorithmName = "specStoch";
      createLinAlgGreedy<SpecStoch>();
    } else if (alg == AlgorithmType::exhaustive) {
      algorithmName = "Exhaustive";
      createSpecific<RobustnessExhaustiveSearch>();
    } else {
      throw std::logic_error("Algorithm not implemented!");
    }
  }

  template <template <typename __Solver> class Greedy>
  void createLinAlgGreedy() {
    if (linalg == LinAlgType::lu) {
      createSpecific<Greedy<SparseLUSolver>>();
    } else if (linalg == LinAlgType::lamg) {
      createSpecific<Greedy<LamgDynamicLaplacianSolver>>();
    } else if (linalg == LinAlgType::jlt_lu_sparse) {
      createSpecific<Greedy<JLTLUSolver>>();
    } else if (linalg == LinAlgType::jlt_lamg) {
      createSpecific<Greedy<JLTLamgSolver>>();
    } else {
      throw std::logic_error("Solver not implemented!");
    }
  }

  template <class Greedy> void createSpecific() {
    auto grdy = new Greedy(*params);
    greedy = std::unique_ptr<Greedy>(grdy);
  }

  void run() {
    // Initialize
    n = g.numberOfNodes();
    omp_set_num_threads(threads);
    Aux::Random::setSeed(seed, true);
    g_cpy = g;

    if (alg == AlgorithmType::none) {
      throw std::logic_error("Not implemented!");
    }

    params = std::make_unique<GreedyParams>(g_cpy, k);
    params->epsilon = epsilon;
    params->epsilon2 = epsilon2;
    params->ne = ne;
    if (updatePerRound < k)
      params->updatePerRound = updatePerRound;
    else
      params->updatePerRound = 1;

    params->diff = diff;
    params->heuristic = heuristic;
    params->candidatesize = candidatesize;
    if (linalg == LinAlgType::jlt_lu_sparse || linalg == LinAlgType::jlt_lamg) {
      // params->solverEpsilon = 0.75;
      params->solverEpsilon = 0.55;
    }
    params->always_use_known_columns_as_candidates =
        this->always_use_known_columns_as_candidates;

    std::cout << "Runs: \n";
    std::cout << "- Instance: '" << instanceFile << "'\n";
    std::cout << "  Nodes: " << n << "\n";
    std::cout << "  Edges: " << g.numberOfEdges() << "\n";
    std::cout << "  k: " << k << "\n";
    std::cout << "  Call: " << call << "\n";
    std::cout << "  Threads:  " << threads << "\n";
    std::cout << "  Candidate size:  ";
    if (candidatesize == CandidateSetSize::small) {
      std::cout << "small\n";
    } else if (candidatesize == CandidateSetSize::large) {
      std::cout << "large\n";
    } else {
      std::cout << "unknown\n";
    }
    std::cout << "  SolverEpsilon:  " << params->solverEpsilon << "\n";
    std::cout << "  All-Columns: ";
    if (params->always_use_known_columns_as_candidates) {
      std::cout << "True\n";
    } else {
      std::cout << "False\n";
    }

    if (alg == AlgorithmType::colStoch || alg == AlgorithmType::simplStochDyn ||
        alg == AlgorithmType::specStoch) {
      std::string linalgName = "";
      if (linalg == LinAlgType::lamg) {
        linalgName = "LAMG";
      } else if (linalg == LinAlgType::lu) {
        linalgName = "LU";
      } else if (linalg == LinAlgType::none) {
        linalgName = "Dense CG";
      } else if (linalg == LinAlgType::jlt_lu_sparse) {
        linalgName = "JLT via Sparse LU";
      } else if (linalg == LinAlgType::jlt_lamg) {
        linalgName = "JLT via LAMG";
      }

      if (linalgName != "") {
        std::cout << "  Linalg: " << linalgName << "\n";
      }
    }

    if (alg == AlgorithmType::colStoch) {
      std::string heuristicName;
      if (heuristic == HeuristicType::lpinvDiag) {
        heuristicName = "Lpinv Diagonal";
      }
      if (heuristic == HeuristicType::similarity) {
        heuristicName = "Similarity";
      }
      if (heuristic == HeuristicType::random) {
        heuristicName = "Random";
      }
      std::cout << "  Heuristic: " << heuristicName << "\n";
      std::cout << "  similarityIterations: " << params->similarityIterations << "\n";
      std::cout << "  Epsilon2: " << epsilon2 << "\n";
    }

    if (alg == AlgorithmType::simplStoch || alg == AlgorithmType::simplStochDyn ||
        alg == AlgorithmType::colStoch || alg == AlgorithmType::specStoch) {
      std::cout << "  Epsilon: " << epsilon << "\n";
    }

    // run for num_focus_nodes. pick at random.
    std::mt19937 gen(focus_node_seed);
    
    std::vector<node> all_nodes, focus_nodes;
    all_nodes.reserve(g.numberOfNodes());
    focus_nodes.reserve(num_focus_nodes);
    g.forNodes([&](node u) { all_nodes.push_back(u); });
    std::sample(all_nodes.begin(), all_nodes.end(), std::back_inserter(focus_nodes), num_focus_nodes, gen);

    // initialize once, then run for each focus node
    beforeInit = std::chrono::high_resolution_clock::now();
    createGreedy();
    auto afterInit = std::chrono::high_resolution_clock::now();
    std::cout << "  Algorithm:  "
              << "'" << algorithmName << "'"
              << "\n";

    using scnds = std::chrono::duration<float, std::ratio<1, 1>>;
    std::cout << "  InitTime:    "
              << std::chrono::duration_cast<scnds>(afterInit - beforeInit).count() << "\n";

    std::cout << "  Results:\n";

    for (node focus_node : focus_nodes){
      std::cout << "  - Focus Node: " << focus_node << "\n";
      
      // Run greedy
      
      auto beforeReset = std::chrono::high_resolution_clock::now();
      greedy->reset_focus(focus_node);
      auto beforeRun = std::chrono::high_resolution_clock::now();
      greedy->run();
      auto t = std::chrono::high_resolution_clock::now();
      auto duration = t - beforeReset;
      auto duration_reset = beforeRun - beforeReset;

      // Verify Results
      if (!greedy->isValidSolution()) {
        std::cout << algorithmName << " failed!\n";
        throw std::logic_error(std::string("Algorithm") + algorithmName +
                              "failed!");
      }

      edges = greedy->getResultItems();
      resultResistance = greedy->getResultValue();
      originalResistance = greedy->getOriginalValue();

      // Output Results
      
      if (vv) {
        std::cout << "    EdgeList: [";
        g.forEdges([](NetworKit::node u, NetworKit::node v) {
          std::cout << "(" << u << ", " << v << "), ";
        });
        std::cout << "]\n" << std::endl;
      }
      std::cout << "    Value:  " << resultResistance << "\n";
      std::cout << "    Original Value:  " << originalResistance << "\n";
      std::cout << "    Gain:  " << originalResistance - resultResistance << "\n";

      using scnds = std::chrono::duration<float, std::ratio<1, 1>>;
      std::cout << "    Time:    "
                << std::chrono::duration_cast<scnds>(duration).count() << "\n";
      std::cout << "    Reset Time:    "
                << std::chrono::duration_cast<scnds>(duration_reset).count() << "\n";


      if (alg == AlgorithmType::specStoch) {
        double spectralResultResistance = greedy->getSpectralResultValue();
        double spectralOriginalResistance =
            greedy->getSpectralOriginalResistance();
        std::cout << "    Spectral Value:  " << spectralResultResistance << "\n";
        std::cout << "    Spectral Original Value:  " << spectralOriginalResistance
                  << "\n";
        std::cout << "    Spectral Gain:  "
                  << spectralOriginalResistance - spectralResultResistance
                  << "\n";
        std::cout << "    Eigenpairs:  " << (double)(100 * ne) / n << "\n";
        std::cout << "    ne:  " << ne << "\n";
        std::cout << "    Max Eigenvalue:  " << greedy->getMaxEigenvalue() << "\n";
        std::cout << "    Cth Eigenvalues:  [";
        for (auto lambda : greedy->getCthEigenvalues()) {
          std::cout << lambda << ", ";
        }
        std::cout << "]\n" << std::endl;
        std::cout.flush();
        std::cout << "    Diff2:  " << diff << "\n";
        std::cout << "    UpdatePerRound:  " << updatePerRound << "\n";
      }

      std::cout << "    AddedEdgeList:  [";
      for (auto e : edges) {
        std::cout << "(" << e.u << ", " << e.v << "), ";
      }
      std::cout << "]\n" << std::endl;
      std::cout.flush();

      if (verify_result) {
        if (edges.size() != k) {
          std::ostringstream stringStream;
          stringStream << "Result Error: Output does contains " << edges.size()
                      << " edges, not k = " << k << " edges!";
          std::string error_msg = stringStream.str();
          std::cout << error_msg << "\n";
          throw std::logic_error(error_msg);
        }

        LamgDynamicLaplacianSolver solver;
        double gain = originalResistance - resultResistance;
        double gain_exact = 0.;
        auto g_ = g;
        solver.setup(g_, 0.0001, 2);
        for (auto e : edges) {
          if (g.hasEdge(e.u, e.v)) {
            std::cout << "Error: Edge in result is already in original graph!\n";
            throw std::logic_error(
                "Error: Edge from result already in original graph!");
          }
          if (e.u != focus_node && e.v != focus_node) {
            std::cout << "Error: Edge is not adjacent to the focus node!\n";
            throw std::logic_error("Error: Edge is not adjacent to the focus node!");
          }

          gain_exact += solver.totalResistanceDifferenceExact(e.u, e.v);
          solver.addEdge(e.u, e.v);
          g_.addEdge(e.u, e.v);
        }

        if (std::abs((gain - gain_exact) / gain_exact / k) > 0.01) {
          std::ostringstream stringStream;
          stringStream << "Error: Gain Test failed. Output: " << gain
                      << ", exact: " << gain_exact << ".";
          std::string error_msg = stringStream.str();
          std::cout << error_msg << "\n";
          throw std::logic_error(error_msg);
        }
      }
    };
  }
};

int main(int argc, char *argv[]) {
  // // FOLLOWING :
  // https://petsc.org/release/faq/#in-c-i-get-a-crash-on-vecdestroy-or-some-other-petsc-object-at-the-end-of-the-program
  int ierr;
  ierr = SlepcInitialize(&argc, &argv, (char *)0, "");
  if (ierr) {
    throw std::runtime_error("SlepcInitialize() not working!");
  }
  {
    omp_set_num_threads(1);

    RobustnessExperiment experiment;

    for (int i = 0; i < argc; i++) {
      experiment.call += argv[i];
      experiment.call += " ";
    }

    bool run_tests = false;
    bool run_experiments = true;

    bool verbose = false;

    double k_factor = 1.0;
    bool km_sqrt = false;
    bool km_linear = false;
    bool km_crt = false;

    bool test_jlt = false;
    bool test_esolve = false;
    int count = 0;

    LinAlgType linalg;

    Graph g;
    std::string instance_filename = "";
    int instance_first_node = 0;
    char instance_sep_char = ' ';
    bool instance_directed = false;
    // std::string instance_comment_prefix = "%";
    std::string instance_comment_prefix = "#";

    std::string helpstring =
        "EXAMPLE CALL\n"
        "\trobustness -a1 -i graph.gml\n"
        "OPTIONS:\n"
        "\t-i <path to instance>\n"
        "\t\t\tAccepted Formats: .gml, Networkit Binary via .nkb, edge list.\n"
        "\t-v\t\tAlso output result edges \n"
        "\t-km\t\tk as a function of n, possible values: linear, sqrt, crt, "
        "const (default: const).\n"
        "\t-k 20\t\tmultiplicative factor to influence k (double).\n"
        "\t-a[0-6]\tAlgorithm Selection. 0: Random Edges, 1: Submodular "
        "Greedy, 2: Stochastic Submodular Greedy, via entire pseudoinverse, 3: "
        "Stochastic Submodular Greedy, via on demand column computation, (4: "
        "Unassigned), 5: Main Algorithm Prototype, via entire pseudoinverse, "
        "6: Main Algorithm.\n"
        "\t-tr\t\tTest Results for correctness (correct number of edges, "
        "correct total resistance value, no edges from original graph). \n"
        "\t-vv\t\tAlso output edges of input graph. \n"
        "\t-eps\t\tEpsilon value for approximative algorithms. Default: 0.1\n"
        "\t-eps2\t\tAbsolute Error bound for UST based approximation. \n"
        "\t-j 12\t\tNumber of threads\n"
        "\t--lamg\t\tUse NetworKit LAMG solver (currently only with a6 and "
        "a3)\n"
        "\t--lu\t\tUse Eigen LU solver (currently only with a3 and a6)\n"
        "\t--jlt-lamg\tUse NetworKit LAMG solver in combination with JLT (only "
        "with a6)\n"
        "\t--jlt-lu\tUse Eigen LU solver in combination with JLT (only with "
        "a6)\n"
        "\t-h[0-2]\t\tHeuristics for a6. 0: random, 1: resistance, 2: "
        "similarity\n"
        "\t-s[0-1]\t\tSize of candidate set for stochastic approaches 0: "
        "larger space (n^2-m)/k, 1: nlogn/k\n"
        "\t--seed\t\tSeed for NetworKit random generators.\n"
        "\t-in\t\tFirst node of edge list instances\n"
        "\t-isep\t\tSeparating character of edge list instance file\n"
        "\t-ic\t\tComment character of edge list instance file\n"
        "\t-ne\t\tNumber of eigenpairs to be calculated. \n"
        "\t-ur\t\tNumber of rounds to update eigenpairs. \n"
        "\t-diff2\t\tSelect one of diff evaluation for spectral approach "
        "{0,1,2}. \n"
        "\t-nfocus<n>\tNumber of focus nodes to calculate"
        "\t--loglevel\t\tActivate loging. Levels: TRACE, DEBUG, INFO, WARN, "
        "ERROR, FATAL.\n"
        "\n";

    if (argc < 2) {
      std::cout << "Error: Call without arguments.\n" << helpstring;
      return 1;
    }

    if (argv == 0) {
      std::cout << "Error!";
      return 1;
    }
    if (argc < 2) {
      std::cout << helpstring;
      return 1;
    }

    // Return next arg and increment i
    auto nextArg = [&](int &i) {
      if (i + 1 >= argc || argv[i + 1] == 0) {
        std::cout << "Error!";
        throw std::exception();
      }
      i++;
      return argv[i];
    };

    for (int i = 1; i < argc; i++) {
      if (argv[i] == 0) {
        std::cout << "Error!";
        return 1;
      }
      std::string arg = argv[i];

      if (arg == "-h" || arg == "--help") {
        std::cout << helpstring;
        return 0;
      }
      if (arg == "--loglevel") {
        std::string logLevel = nextArg(i);
        Aux::Log::setLogLevel(logLevel);
      }

      if (arg == "-v" || arg == "--verbose") {
        experiment.verbose = true;
        verbose = true;
        continue;
      }
      if (arg == "-vv" || arg == "--very-verbose") {
        verbose = true;
        experiment.vv = true;
        continue;
      }

      if (arg == "--test-jlt") {
        test_jlt = true;
      }

      if (arg == "--test-esolve") {
        auto arg_esolve = nextArg(i);
        count = atoi(arg_esolve);
        test_esolve = true;
      }

      if (arg == "-tr") {
        experiment.verify_result = true;
      }

      if (arg == "-k" || arg == "--override-k" || arg == "--k-factor") {
        std::string k_string = nextArg(i);
        k_factor = std::stod(k_string);
        continue;
      }
      if (arg == "-km") {
        std::string km_string = nextArg(i);
        if (km_string == "sqrt") {
          km_sqrt = true;
        } else if (km_string == "linear") {
          km_linear = true;
        } else if (km_string == "crt") {
          km_crt = true;
        } else {
          std::cout << "Error: bad argument to -km. Possible Values: 'sqrt', "
                       "'linear', 'crt'. Default is const.\n";
          return 1;
        }
        continue;
      }

      if (arg == "-j") {
        auto arg_str = nextArg(i);
        experiment.threads = atoi(arg_str);
        omp_set_num_threads(atoi(arg_str));
      }

      if (arg == "-eps" || arg == "--eps") {
        experiment.epsilon = std::stod(nextArg(i));
        continue;
      }
      if (arg == "-eps2" || arg == "--eps2") {
        experiment.epsilon2 = std::stod(nextArg(i));
        continue;
      }
      if (arg == "-ne" || arg == "--ne") {
        auto arg_ne = nextArg(i);
        experiment.ne = atoi(arg_ne);
      }
      if (arg == "-ur" || arg == "--ur") {
        auto arg_ur = nextArg(i);
        // if (atoi(arg_ur) < k_factor)
        experiment.updatePerRound = atoi(arg_ur);
      }
      if (arg == "-diff" || arg == "--diff") {
        auto arg_diff = nextArg(i);
        experiment.diff = atoi(arg_diff);
        continue;
      }

      if (arg == "--nfocus" | arg == "-nf") {
        experiment.num_focus_nodes = std::stoi(nextArg(i));
        continue;
      }

      if (arg == "--focus-seed") {
        experiment.focus_node_seed = std::stoi(nextArg(i));
        continue;
      }

      if (arg == "--all-columns") {
        experiment.always_use_known_columns_as_candidates = true;
      }

      if (arg == "--seed") {
        auto seed_string = nextArg(i);
        experiment.seed = atoi(seed_string);
        continue;
      }
      if (arg == "-a1") {
        experiment.alg = AlgorithmType::stGreedy;
        continue;
      }
      if (arg == "-a2") {
        experiment.alg = AlgorithmType::simplStoch;
        continue;
      }
      if (arg == "-a3") {
        experiment.alg = AlgorithmType::simplStochDyn;
        continue;
      }
      if (arg == "-a6") {
        experiment.alg = AlgorithmType::colStoch;
        continue;
      }
      if (arg == "-a7") {
        experiment.alg = AlgorithmType::specStoch;
        continue;
      }
      if (arg == "-a8") {
        experiment.alg = AlgorithmType::exhaustive;
        continue;
      }

      if (arg == "-h1") {
        experiment.heuristic = HeuristicType::lpinvDiag;
      }
      if (arg == "-h2") {
        experiment.heuristic = HeuristicType::similarity;
      }

      if (arg == "-s0") {
        experiment.candidatesize = CandidateSetSize::small;
      }
      if (arg == "-s1") {
        experiment.candidatesize = CandidateSetSize::large;
      }

      if (arg == "-t") {
        run_tests = true;
        run_experiments = false;
        continue;
      }

      if (arg == "-i" || arg == "--instance") {
        instance_filename = nextArg(i);
        experiment.instanceFile = instance_filename;
        continue;
      }
      if (arg == "-in") {
        instance_first_node = atoi(nextArg(i));
      }
      if (arg == "-isep") {
        instance_sep_char = nextArg(i)[0];
      }
      if (arg == "--isepspace") {
        instance_sep_char = ' ';
        std::cout << "OY";
      }
      if (arg == "-id") {
        instance_directed = true;
      }
      if (arg == "-ic") {
        instance_comment_prefix = nextArg(i);
      }

      if (arg == "--lamg") {
        linalg = LinAlgType::lamg;
      }
      if (arg == "--lu") {
        linalg = LinAlgType::lu;
      }
      if (arg == "--jlt-lu") {
        linalg = LinAlgType::jlt_lu_sparse;
      }
      if (arg == "--jlt-lamg") {
        linalg = LinAlgType::jlt_lamg;
      }
      experiment.linalg = linalg;
    }

    // Read graph file

    auto hasEnding = [&](std::string const &fullString,
                         std::string const &ending) {
      if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(),
                                        ending.length(), ending));
      } else {
        return false;
      }
    };
    auto hasStart = [&](std::string const &fullString,
                        std::string const &start) {
      if (fullString.length() >= start.length()) {
        return (0 == fullString.compare(0, start.length(), start));
      } else {
        return false;
      }
    };
    if (instance_filename != "") {
      if (hasEnding(instance_filename, ".gml")) {
        GMLGraphReader reader;
        try {
          g = reader.read(instance_filename);
        } catch (const std::exception &e) {
          std::cout << "Failed to open or parse gml file " + instance_filename
                    << '\n';
          return 1;
        }
      } else if (hasEnding(instance_filename, "nkb") ||
                 hasEnding(instance_filename, "networkit")) {
        NetworkitBinaryReader reader;
        try {
          g = reader.read(instance_filename);
        }

        catch (const std::exception &e) {
          std::cout << "Failed to open or parse networkit binary file "
                    << instance_filename << '\n';
          return 1;
        }
      } else {
        try {
          NetworKit::EdgeListReader reader(
              instance_sep_char, NetworKit::node(instance_first_node),
              instance_comment_prefix, true, instance_directed);
          g = reader.read(instance_filename);
        } catch (const std::exception &e) {
          std::cout << "Failed to open or parse edge list file " +
                           instance_filename
                    << '\n'
                    << e.what() << "\n";
          return 1;
        }
      }
      g = NetworKit::ConnectedComponents::extractLargestConnectedComponent(
          g, true);
      g.removeSelfLoops();
      assert(g.checkConsistency());
    }
    int k = 1;
    int n = g.numberOfNodes();
    if (km_sqrt) {
      k = std::sqrt(n) * k_factor;
    } else if (km_linear) {
      k = n * k_factor;
    } else if (km_crt) {
      k = std::pow(n, 0.33333) * k_factor;
    } else {
      k = k_factor;
    }
    k = std::max(k, 1);
    experiment.k = k;

    if (test_jlt) {
      run_experiments = false;
      testJLT(g, instance_filename, k);
    }

    if (test_esolve) {
      run_experiments = false;
      std::cout << "Runs: \n";
      std::cout << "- Instance: '" << instance_filename << "'\n";
      std::cout << "  Nodes: " << g.numberOfNodes() << "\n";
      std::cout << "  Edges: " << g.numberOfEdges() << "\n";
      std::string algorithmName = "Eigensolver";
      std::cout << "  Algorithm:  "
                << "'" << algorithmName << "'"
                << "\n";
      testEigensolver(g, count, true, true);
    }

    if (run_experiments) {
      NetworKit::ConnectedComponents comp{g};
      comp.run();
      if (comp.numberOfComponents() != 1) {
        std::cout << "Error: Instance " << instance_filename
                  << " is not connected!\n";
        return 1;
      }
      experiment.g = g;
      experiment.run();
    }

    if (run_tests) {
      // testRobustnessStochasticGreedySpectral();
      testSolverSetupTime(g);
      // testDynamicColumnApprox();
      // testRobustnessSubmodularGreedy();
    }
  }

  ierr = SlepcFinalize();
  if (ierr) {
    return ierr;
  }
  return 0;
}
