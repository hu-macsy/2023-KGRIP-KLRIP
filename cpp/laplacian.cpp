#include <laplacian.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <iostream>

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/centrality/ApproxElectricalCloseness.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <networkit/numerics/Preconditioner/DiagonalPreconditioner.hpp>

using namespace Eigen;

// Does not support multiple parallel edges
SparseMatrix<double> laplacianMatrixSparse(NetworKit::Graph const &g) {
  auto n = g.numberOfNodes();
  auto m = g.numberOfEdges();

  typedef Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(3 * m);
  VectorXd diagonal = VectorXd::Constant(n, 0.0);
  g.forEdges([&](NetworKit::node u, NetworKit::node v, double weight) {
    if (u == v) {
      std::cout << "Warning: Graph has edge with equal target and destination!";
    }
    tripletList.push_back(T(u, v, -1.));
    tripletList.push_back(T(v, u, -1.));
    diagonal(u) += 1.;
    diagonal(v) += 1.;
  });
  for (int u = 0; u < n; u++) {
    tripletList.push_back(T(u, u, diagonal(u)));
  }
  SparseMatrix<double> laplacian(n, n);
  laplacian.setFromTriplets(tripletList.begin(), tripletList.end());

  return laplacian;
}

void updateLaplacian(SparseMatrix<double> &laplacian, NetworKit::node a,
                     NetworKit::node b) {
  laplacian.coeffRef(a, a) += 1.;
  laplacian.coeffRef(b, b) += 1.;
  laplacian.coeffRef(a, b) -= 1.;
  laplacian.coeffRef(b, a) -= 1.;
}

std::vector<NetworKit::Vector>
laplacianPseudoinverseColumns(const NetworKit::CSRMatrix &laplacian,
                              std::vector<NetworKit::node> indices,
                              double tol) {

  NetworKit::ConjugateGradient<NetworKit::CSRMatrix,
                               NetworKit::DiagonalPreconditioner>
      cg(tol);
  cg.setupConnected(laplacian);

  NetworKit::count n = laplacian.numberOfRows();
  NetworKit::Vector rhs(n);

  std::vector<NetworKit::Vector> results;
  for (unsigned int i = 0; i < n; i++) {
    rhs[i] = 1. / static_cast<double>(n);
  }
  for (auto ind : indices) {
    rhs[ind] += 1.;

    NetworKit::Vector result(n);
    cg.solve(rhs, result);
    double sum = 0.;
    for (unsigned int i = 0; i < n; i++) {
      sum += result[i];
    }
    result -= sum / static_cast<double>(n);
    results.push_back(result);

    rhs[ind] -= 1.;
  }
  return results;
}

std::vector<VectorXd>
laplacianPseudoinverseColumns(const SparseMatrix<double> &L,
                              std::vector<NetworKit::node> indices) {
  std::vector<Eigen::VectorXd> result;
  SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
  solver.analyzePattern(L);
  solver.factorize(L);

  // ConjugateGradient<SparseMatrix<double>, Lower|Upper, IncompleteCholesky>
  // solver; solver.compute(L);
  if (solver.info() != Success) {
    // solver failed
    throw std::logic_error("Solver failed.");
  }
  int n = L.cols();

  for (auto ind : indices) {
    Eigen::VectorXd b = VectorXd::Constant(n, -1.0 / n);
    b(ind) += 1.0;

    auto x = solver.solve(b);
    if (solver.info() != Success) {
      // solving failed
      throw std::logic_error(std::string("Solving failed! Error code: ") +
                             std::to_string(solver.info()) +
                             std::string(", node: ") + std::to_string(ind));
    }
    double avg = x.sum() / static_cast<double>(n);
    auto vec = x - VectorXd::Constant(n, avg);
    result.push_back(vec);
  }
  return result;
}

MatrixXd laplacianPseudoinverse(MatrixXd lp) {
  int n = lp.cols();
  MatrixXd J = MatrixXd::Constant(n, n, 1.0 / n);
  MatrixXd I = MatrixXd::Identity(n, n);

  return (lp + J).llt().solve(I) - J;
}
MatrixXd laplacianPseudoinverse(NetworKit::Graph const &g) {
  auto mat = laplacianMatrix<MatrixXd>(g);
  return laplacianPseudoinverse(mat);
}

VectorXd laplacianPseudoinverseColumnDifference(MatrixXd const &lpinv, int i,
                                                int j, int k,
                                                double conductance) {
  double R_ij = lpinv(i, i) + lpinv(j, j) - 2 * lpinv(i, j);
  double w = 1.0 / (1.0 / conductance + R_ij);
  VectorXd v = lpinv.col(i) - lpinv.col(j);
  return v * (v(k) * w * (-1.0));
}

// Update formula for the trace of the lap pinv as an edge is added to the
// graph. Add edge (i, j) and compute the difference of the traces of the
// pseudoinverses. Depends only on i-th and j-th column of lpinv. Add this to
// the old to get the new.
double laplacianPseudoinverseTraceDifference(VectorXd const &column_i, int i,
                                             VectorXd const &column_j, int j,
                                             double conductance) {
  double R_ij = column_i(i) + column_j(j) - 2 * column_i(j);
  double w = 1.0 / (1.0 / conductance + R_ij);
  return (column_i - column_j).squaredNorm() * w * (-1.0);
}

double laplacianPseudoinverseTraceDifference(MatrixXd const &lpinv, int i,
                                             int j, double conductance) {
  auto &col_i = lpinv.col(i);
  auto &col_j = lpinv.col(j);
  double R_ij = col_i(i) + col_j(j) - 2 * col_i(j);
  double w = 1.0 / (1.0 / conductance + R_ij);

  return (col_i - col_j).squaredNorm() * w * (-1.0);
}

// Update formula for the trace of the lap pinv as two edges are added to the
// graph. Based on Sherman-Morrison-Woodbury formula.
/*
double laplacianPseudoinverseTraceDifference2(std::vector<Vector> const &
columns, NetworKit::Edge e1, NetworKit::Edge e2, double conductance1, double
conductance2)
{
        auto e1_vec = columns[e1.u] - columns[e1.v];
        auto e2_vec = columns[e2.u] - columns[e2.v];
        double a = conductance1 + e1_vec[e1.u] - e1_vec[e1.v];
        double b = e1_vec[e2.u] - e1_vec[e2.v];
        double c = conductance2 + e2_vec[e2.u] - e2_vec[e2.v];
        double det = a*c-b*b;
        return (-1.0) / det * (Vector::innerProduct(e1_vec, e1_vec) * c
+ 2.0*Vector::innerProduct(e1_vec, e2_vec) * b + Vector::innerProduct(e2_vec,
e2_vec) * a);
}
*/

// Trace difference between lpinv when two edges are added, based on woodbury
// matrix identity. O(n^2). Does not copy and update the pseudoinverse
// explicitly. If the lpinv is available and can be written then this should be
// done in place by updating the matrix instead.
double laplacianPseudoinverseTraceDifference2(MatrixXd const &lpinv,
                                              NetworKit::Edge e1,
                                              NetworKit::Edge e2,
                                              double conductance1,
                                              double conductance2) {
  auto u1 = e1.u;
  auto u2 = e2.u;
  auto v1 = e1.v;
  auto v2 = e2.v;

  auto e1_L = lpinv.col(u1) - lpinv.col(v1);
  auto e2_L = lpinv.col(u2) - lpinv.col(v2);

  double a = 1.0 / conductance1 + e1_L(u1) - e1_L(v1);
  double b = e1_L(u2) - e1_L(v2);
  double c = 1.0 / conductance2 + e2_L(u2) - e2_L(v2);
  double det = a * c - b * b;

  return (-1.0) / det *
         (c * e1_L.squaredNorm() - (2.0 * b) * e1_L.dot(e2_L) +
          a * e2_L.squaredNorm());
}

// Based on Woodbury matrix identity. O(k^3 + k n^2 + k^2 n) where k is the
// number of edges. Does not compute the pseudoinverse explicitly. Don't use for
// now, not performance optimized. There's sparse vectors here which is not
// exploited at all.
double laplacianPseudoinverseTraceDifference(MatrixXd const &lpinv,
                                             std::vector<NetworKit::Edge> edges,
                                             std::vector<double> conductances) {
  int k = edges.size();
  int n = lpinv.rows();
  MatrixXd u = MatrixXd::Zero(n, k);
  MatrixXd c_inv = MatrixXd::Zero(k, k);
  for (int i = 0; i < k; i++) {
    double resistance = 1.0;
    if (i < conductances.size()) {
      resistance = 1.0 / conductances[i];
    }
    c_inv(i, i) = resistance;
    auto e = edges[i];
    u(e.u, i) = 1.0;
    u(e.v, i) = -1.0;
  }
  // auto a = lpinv * u;
  // return (-1.0) * (a * (c_inv + u.transpose() * lpinv * u).inverse() *
  // a.transpose()).trace();
  return (-1.0) * (lpinv * u * (c_inv + u.transpose() * lpinv * u).inverse() *
                   u.transpose() * lpinv)
                      .trace();
}

void updateLaplacianPseudoinverse(MatrixXd &lpinv, NetworKit::Edge e,
                                  double conductance) {
  auto i = e.u;
  auto j = e.v;
  double R_ij = lpinv(i, i) + lpinv(j, j) - 2 * lpinv(i, j);
  double w_negative = -1.0 / (1.0 / conductance + R_ij);
  VectorXd v = lpinv.col(i) - lpinv.col(j);
  // VectorXd col_i = lpinv.col(i);
  // VectorXd col_j = lpinv.col(j);

  int n = lpinv.cols();
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    lpinv.col(i) += v(i) * w_negative * v;
  }

  // This version causes pagefaults on my system which slows down the entire
  // thing by a factor of 10 ...
  // lpinv += (w_negative * v) * v.transpose();
}

MatrixXd updateLaplacianPseudoinverseCopy(MatrixXd const &lpinv,
                                          NetworKit::Edge e,
                                          double conductance) {
  auto i = e.u;
  auto j = e.v;
  double R_ij = lpinv(i, i) + lpinv(j, j) - 2 * lpinv(i, j);
  double w = 1.0 / (1.0 / conductance + R_ij);
  VectorXd v = lpinv.col(i) - lpinv.col(j);

  return lpinv - w * v * v.transpose();
}

// Compute a stochastic approximation of the diagonal of the Moore-Penrose
// pseudoinverse of the laplacian matrix of a graph.
VectorXd approxLaplacianPseudoinverseDiagonal(NetworKit::Graph const &G,
                                              double epsilon) {
  auto n = G.numberOfNodes();

  NetworKit::ApproxElectricalCloseness diagAlg(G, epsilon);
  diagAlg.run();
  auto diag_std_vec = diagAlg.getDiagonal();

  VectorXd lpinvDiag(n);
  for (auto i = 0; i < n; i++) {
    lpinvDiag(i) = diag_std_vec[i];
  }

  return lpinvDiag;
}
