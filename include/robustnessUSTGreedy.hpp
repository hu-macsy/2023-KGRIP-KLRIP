#ifndef ROBUSTNESS_UST_GREEDY_H
#define ROBUSTNESS_UST_GREEDY_H


#include <utility>
#include <random>
#include <exception>
#include <cmath>
#include <set>
#include <vector>
#include <map>

#include <laplacian.hpp>
#include <greedy.hpp>
#include <dynamicLaplacianSolver.hpp>
#include <greedy_params.hpp>

#include <Eigen/Dense>
#include <networkit/graph/Graph.hpp>
#include <networkit/centrality/ApproxElectricalCloseness.hpp>
#include <networkit/numerics/Preconditioner/DiagonalPreconditioner.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/components/ConnectedComponents.hpp>



using namespace NetworKit;




template <class DynamicLaplacianSolver=LamgDynamicLaplacianSolver>
//using DynamicLaplacianSolver = LeastSquaresDynamicLaplacianSolver;
class RobustnessTreeGreedy final : public AbstractOptimizer<NetworKit::Edge> {
public:
    RobustnessTreeGreedy(GreedyParams params) : G(params.g) {
        this->n = G.numberOfNodes();
        this->k = params.k;
        this->epsilon = params.epsilon;
        this->heuristic = params.heuristic;
        this->round = 0;

        solver.setup(G, 0.1, numberOfNodeCandidates());
        this->totalValue = 0.;
        this->originalResistance = 0.;

        gen = std::mt19937(Aux::Random::getSeed());

        if (heuristic == HeuristicType::lpinvDiag) {
            apx = std::make_unique<ApproxElectricalCloseness>(params.g, params.epsilon2);
            apx->run();
            this->diag = apx->getDiagonal();
            G.forNodes([&](node u) { this->totalValue -= static_cast<double>(this->n) * this->diag[u]; });
            originalResistance = -1. * totalValue;
            INFO("Usts: ", apx->getUstCount());
        } else {
            similarityPhi = params.similarityPhi;
            similarityIterations = params.similarityIterations;
            // pass
        }
        //std::cout << "exact: " << exact << ", approx: " << this->totalValue << ", diff: " << exact-totalValue << "\n";
    }


    double getResultValue() {
        return this->totalValue * (-1.0);
    }

    double getOriginalValue() {
        return this->originalResistance;
    }

    std::vector<NetworKit::Edge> getResultItems() {
        return this->results;
    }

    bool isValidSolution() {
        return this->validSolution;
    }

    count numberOfNodeCandidates() {
        unsigned int s = (unsigned int)std::sqrt(1.0 * (this->n * (this->n-1) /2 - this->G.numberOfEdges() - this->round) / k * std::log(1.0/epsilon));
        if (s < 2) { s = 2; }
        if (s > n/2) { s = n/2; }
        return s;
    }

    void run() {
        this->round = 0;
        this->validSolution = false;
        this->results.clear();
        struct edge_cmp {
            bool operator() (const Edge& lhs, const Edge& rhs) const {
                return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
            }
        };
        std::set<Edge, edge_cmp> resultSet;

        if (k + G.numberOfEdges() > (n * (n-1) / 8*3)) { // 3/4 of the complete graph.
            this->hasRun = true;
            std::cout << "Bad call to TreeGreedy, adding this many edges is not supported! Attempting to have " << k + G.numberOfEdges() << " edges, limit is " << n*(n-1) / 8 *3;
            return;
        }

        for (int r = 0; r < k; r++)
        {
            double bestGain = -std::numeric_limits<double>::infinity();
            Edge bestEdge;

            int it = 0;

            do {
                // Collect nodes set for current round
                std::set<NetworKit::node> nodes;
                unsigned int s = numberOfNodeCandidates();
                if (r == 0) { INFO("Columns in round 1: ", s); }

                double min = std::numeric_limits<double>::infinity();

                std::vector<double> nodeWeights(n);

                // the random choice following this may fail if all the vertex pairs are already present as edges, we use heuristic information the first time, uniform distribution if it fails
                if (heuristic == HeuristicType::lpinvDiag && it++ < 2) {
                    double tr = -1. * this->totalValue;
                    G.forNodes([&](node u) {
                        double val = static_cast<double>(n) * diag[u] + tr;
                        if (val < min) { min = val; }
                        nodeWeights[u] = val;
                    });
                    for (auto &v : nodeWeights) {
                        auto u = v - min;
                        v = u*u;
                    }
                } else if (heuristic == HeuristicType::similarity && it++ < 2) {
                    // TODO test this
                    // TODO export the parameters ...
                    double phi = similarityPhi;
                    count iterations = similarityIterations;
                    Eigen::SparseMatrix<double> A = adjacencyMatrix(G);
                    Eigen::VectorXd d (n);
                    G.forNodes([&](node i) { d(i) = 1. / G.degree(i); });
                    Eigen::VectorXd u = d;
                    for (int i = 0; i < iterations; i++) {
                        u = phi * A * u + d;
                    }
                    Eigen::VectorXd summedSimilarity = u;

                    // We are interested in nodes that are dissimilar to other nodes i.e. the summed similarity score should be small.
                    double max = -std::numeric_limits<double>::infinity();
                    G.forNodes([&](node i) { 
                        summedSimilarity(i) /= G.degree(i); 
                        double val = summedSimilarity(i);
                        if (val > max) { max = val; }
                        nodeWeights[i] = val;
                    });
                    for (auto &v : nodeWeights) {
                        auto w = max - v;
                        v = w * w;
                    }
                } else {
                    G.forNodes([&](node u) {
                        nodeWeights[u] = 1.;
                    });
                }

                std::discrete_distribution<> distribution_nodes_heuristic (nodeWeights.begin(), nodeWeights.end());

                nodeWeights.clear();
                for (int i = 0; i < n; i++) {
                    nodeWeights.push_back(1.0);
                }
                std::discrete_distribution<> distribution_nodes_uniform (nodeWeights.begin(), nodeWeights.end());

                while (nodes.size() < s / 2) {
                    nodes.insert(distribution_nodes_heuristic(gen));
                }
                
                while (nodes.size() < s) {
                    nodes.insert(distribution_nodes_uniform(gen));
                }
                std::vector<node> nodesVec {nodes.begin(), nodes.end()};

                solver.computeColumns(nodesVec);

                // Determine best edge between nodes from node set
                auto edgeValid = [&](node u, node v){
                    if (this->G.hasEdge(u, v) || this->G.hasEdge(v, u)) return false;
                    if (resultSet.count(Edge(u, v)) > 0 || resultSet.count(Edge(v, u)) > 0) { return false; }
                    return true;
                };
                for (int i = 0; i < s; i++) {
                    auto u = nodesVec[i];
                    for (int j = 0; j < i; j++) {
                        auto v = nodesVec[j];
                        if (edgeValid(u, v)) {
                            double gain = objectiveDifference(Edge(u, v));
                            if (gain > bestGain) {
                                bestEdge = Edge(u, v);
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

            this->totalValue += bestGain;

            if (this->round < k-1) {
                solver.addEdge(u, v);

                if (heuristic == HeuristicType::lpinvDiag) {
                    apx->edgeAdded(u, v);
                    this->diag = apx->getDiagonal();
                }
            } else {
                this->validSolution = true;
            }
            this->round++;
        }
        this->hasRun = true;
        this->results = {resultSet.begin(), resultSet.end()};
        INFO("Computed columns: ", solver.getComputedColumnCount());        
    }

    
private:
    virtual double objectiveDifference(Edge e) {
        auto u = e.u;
        auto v = e.v;
        return solver.totalResistanceDifference(u, v);
    }


    Graph& G;
	std::vector<Edge> results;

    int n;
    bool validSolution = false;
	int round=0;
    int k;

	double totalValue = 0.0;
    double epsilon = 0.1;
    double originalResistance = 0.;

    count similarityIterations = 100;
    double similarityPhi = 0.5;
    std::mt19937 gen;
    std::vector<double> diag;

    std::unique_ptr<ApproxElectricalCloseness> apx;
    DynamicLaplacianSolver solver;

    HeuristicType heuristic;
};


#endif