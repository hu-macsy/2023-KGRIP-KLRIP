#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <vector>
#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/algebraic/Vector.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/algebraic/Vector.hpp>
#include <networkit/graph/Graph.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>



Eigen::MatrixXd laplacianMatrix(NetworKit::Graph const & g);
Eigen::SparseMatrix<double> laplacianMatrixSparse(NetworKit::Graph const & g);

Eigen::MatrixXd laplacianPseudoinverse(NetworKit::Graph const & g);
Eigen::MatrixXd laplacianPseudoinverse(Eigen::MatrixXd laplacian);

Eigen::VectorXd laplacianPseudoinverseColumn(const Eigen::SparseMatrix<double> & L, int k);
std::vector<Eigen::VectorXd> laplacianPseudoinverseColumns(const Eigen::SparseMatrix<double> &L, std::vector<NetworKit::node> indices);
std::vector<NetworKit::Vector> laplacianPseudoinverseColumns(const NetworKit::CSRMatrix &laplacian, std::vector<NetworKit::node> indices, double tol=0.1);

void updateLaplacian(Eigen::SparseMatrix<double> &laplacian, NetworKit::node a, NetworKit::node b);


// Update formula for the pseudoinverse of the Laplacian as an edge is added to the graph.
// Add edge (i, j) and compute difference between the k-th column of the pseudoinverse.
// Takes the columns of the pseudoinverse that correspond to the vertices i and j.
// Add this to the old to get the new.
Eigen::VectorXd laplacianPseudoinverseColumnDifference(Eigen::MatrixXd const & lpinv, int i, int j, int k, double conductance = 1.0);

// Update formula for the trace of the lap pinv as an edge is added to the graph.
// Add edge (i, j) and compute the difference of the traces of the pseudoinverses.
// Add this to the old to get the new.
double laplacianPseudoinverseTraceDifference(Eigen::VectorXd const & column_i, int i, Eigen::VectorXd const &column_j, int j, double conductance = 1.0);
double laplacianPseudoinverseTraceDifference(Eigen::MatrixXd const & lpinv, int i, int j, double conductance = 1.0);
double laplacianPseudoinverseTraceDifference(Eigen::MatrixXd const & lpinv, std::vector<NetworKit::Edge> edges, std::vector<double> conductances=std::vector<double>());

// Update formula for the trace of the lpinv as two edges are added to the graph.
//double laplacianPseudoinverseTraceDifference2(Eigen::MatrixXd const & lpinv, Edge e1, Edge e2, double conductance1=1.0, double conductance2=1.0);
double laplacianPseudoinverseTraceDifference2(Eigen::MatrixXd const & lpinv, NetworKit::Edge e1, NetworKit::Edge e2, double conductance1=1.0, double conductance2=1.0);


void updateLaplacianPseudoinverse(Eigen::MatrixXd & lpinv, NetworKit::Edge e, double conductance = 1.0);
Eigen::MatrixXd updateLaplacianPseudoinverseCopy(Eigen::MatrixXd const & lpinv, NetworKit::Edge e, double conductance = 1.0);

// Compute a stochastic approximation of the diagonal of the Moore-Penrose pseudoinverse of the laplacian matrix of a graph.
Eigen::VectorXd approxLaplacianPseudoinverseDiagonal(NetworKit::Graph const &G, double epsilon = 0.1);
Eigen::VectorXd approxEffectiveResistances(NetworKit::Graph const &G, int &pivot, double epsilon = 0.1);


#endif // LAPLACIAN_H