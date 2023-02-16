#ifndef LAPLACIAN_H
#define LAPLACIAN_H

#include <networkit/algebraic/CSRMatrix.hpp>
#include <networkit/algebraic/DenseMatrix.hpp>
#include <networkit/algebraic/Vector.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/numerics/ConjugateGradient.hpp>
#include <networkit/numerics/LAMG/Lamg.hpp>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <functional>

inline NetworKit::Vector eigen_to_nk(Eigen::VectorXd v) {
  auto n = v.size();
  NetworKit::Vector result(n);
  for (int i = 0; i < n; i++) {
    result[i] = v(i);
  }
  return result;
}

inline Eigen::VectorXd nk_to_eigen(NetworKit::Vector v) {
  auto n = v.getDimension();
  Eigen::VectorXd result(n);
  for (int i = 0; i < n; i++) {
    result(i) = v[i];
  }
  return result;
}

template <class NK_Matrix>
inline NK_Matrix nk_matrix_from_expr(
    NetworKit::count n_rows, NetworKit::count n_cols,
    std::function<double(NetworKit::count, NetworKit::count)> expr) {
  std::vector<NetworKit::Triplet> triplets;
  triplets.reserve(n_rows * n_cols);
  for (NetworKit::count i = 0; i < n_rows; i++) {
    for (NetworKit::count j = 0; j < n_cols; j++) {
      double v = expr(i, j);
      if (v != 0.) {
        triplets.push_back({i, j, v});
      }
    }
  }
  return NK_Matrix{n_rows, n_cols, triplets};
}

inline NetworKit::CSRMatrix nk_dense_to_csr(NetworKit::DenseMatrix dense) {
  auto entry = [&](NetworKit::count i, NetworKit::count j) -> double {
    return dense(i, j);
  };
  return nk_matrix_from_expr<NetworKit::CSRMatrix>(
      dense.numberOfRows(), dense.numberOfColumns(), entry);
}
inline NetworKit::DenseMatrix nk_csr_to_dense(NetworKit::CSRMatrix sparse) {
  auto entry = [&](NetworKit::count i, NetworKit::count j) -> double {
    return sparse(i, j);
  };
  return nk_matrix_from_expr<NetworKit::DenseMatrix>(
      sparse.numberOfRows(), sparse.numberOfColumns(), entry);
}

template <class MatrixType>
MatrixType laplacianMatrix(NetworKit::Graph const &g);

template <>
inline Eigen::MatrixXd
laplacianMatrix<Eigen::MatrixXd>(NetworKit::Graph const &g) {
  auto n = g.numberOfNodes();
  Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(n, n);
  g.forEdges([&](NetworKit::node u, NetworKit::node v, double weight) {
    if (u == v) {
      std::cout << "Warning: Graph has edge with equal target and destination!";
    }
    laplacian(u, u) += weight;
    laplacian(v, v) += weight;
    laplacian(u, v) -= weight;
    laplacian(v, u) -= weight;
  });
  return laplacian;
}

inline Eigen::SparseMatrix<double> sparseFromDense(Eigen::MatrixXd dense) {
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;

  auto m = dense.rows();
  auto n = dense.cols();

  tripletList.reserve(3 * m * n);

  for (unsigned int x = 0; x < n; x++) {
    for (unsigned int y = 0; y < m; y++) {
      tripletList.push_back(T(x, y, dense(x, y)));
    }
  }

  Eigen::SparseMatrix<double> sparse(m, n);
  sparse.setFromTriplets(tripletList.begin(), tripletList.end());

  return sparse;
}

template <>
inline Eigen::SparseMatrix<double>
laplacianMatrix<Eigen::SparseMatrix<double>>(NetworKit::Graph const &g) {
  auto n = g.numberOfNodes();
  auto m = g.numberOfEdges();

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(3 * m);
  Eigen::VectorXd diagonal = Eigen::VectorXd::Constant(n, 0.0);
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
  Eigen::SparseMatrix<double> laplacian(n, n);
  laplacian.setFromTriplets(tripletList.begin(), tripletList.end());

  return laplacian;
}

inline Eigen::SparseMatrix<double> incidenceMatrix(const NetworKit::Graph &g) {
  auto m = g.numberOfEdges();
  auto n = g.numberOfNodes();
  Eigen::SparseMatrix<double> incidence(n, m);

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(3 * 2 * m);

  int eid = 0;
  g.forEdges([&](NetworKit::node u, NetworKit::node v) {
    if (u > v) {
      std::swap(u, v);
    }
    tripletList.push_back(T(u, eid, 1.));
    tripletList.push_back(T(v, eid, -1.));
    eid++;
  });

  incidence.setFromTriplets(tripletList.begin(), tripletList.end());
  return incidence;
}

inline Eigen::SparseMatrix<double> adjacencyMatrix(NetworKit::Graph const &g) {
  auto n = g.numberOfNodes();
  auto m = g.numberOfEdges();

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(3 * m);
  g.forEdges([&](NetworKit::node u, NetworKit::node v, double weight) {
    if (u == v) {
      std::cout << "Warning: Graph has edge with equal target and destination!";
    }
    tripletList.push_back(T(u, v, 1.));
    tripletList.push_back(T(v, u, 1.));
  });
  Eigen::SparseMatrix<double> adjacency(n, n);
  adjacency.setFromTriplets(tripletList.begin(), tripletList.end());

  return adjacency;
}

inline Eigen::SparseMatrix<double> degreeMatrix(NetworKit::Graph const &g) {
  auto n = g.numberOfNodes();
  auto m = g.numberOfEdges();

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(3 * n);
  Eigen::VectorXd diagonal = Eigen::VectorXd::Constant(n, 0.0);
  g.forEdges([&](NetworKit::node u, NetworKit::node v, double weight) {
    if (u == v) {
      std::cout << "Warning: Graph has edge with equal target and destination!";
    }
    diagonal(u) += 1.;
    diagonal(v) += 1.;
  });
  for (int u = 0; u < n; u++) {
    tripletList.push_back(T(u, u, diagonal(u)));
  }
  Eigen::SparseMatrix<double> dmat(n, n);
  dmat.setFromTriplets(tripletList.begin(), tripletList.end());

  return dmat;
}

Eigen::SparseMatrix<double> laplacianMatrixSparse(NetworKit::Graph const &g);

Eigen::MatrixXd laplacianPseudoinverse(NetworKit::Graph const &g);
Eigen::MatrixXd laplacianPseudoinverse(Eigen::MatrixXd laplacian);

template <class MatrixType = Eigen::SparseMatrix<double>>
Eigen::VectorXd laplacianPseudoinverseColumn(const MatrixType &L, int k);

// solver is expected to be already set up
// rhs is expected to be Eigen::VectorXd::Constant(n, -1. /
// static_cast<double>(n))
template <class Solver>
Eigen::VectorXd solveLaplacianPseudoinverseColumn(Solver &solver,
                                                  Eigen::VectorXd &rhs, int n,
                                                  int k) {
  if (rhs.size() != n) {
    rhs = Eigen::VectorXd::Constant(n, -1.0 / static_cast<double>(n));
  }

  rhs(k) = 1. - 1.0 / static_cast<double>(n);
  Eigen::VectorXd x = solver.solve(rhs);
  rhs(k) = -1. / static_cast<double>(n);

  if (solver.info() != Eigen::Success) {
    // solving failed
    throw std::logic_error("Solving failed.!");
  }
  double avg = x.sum() / n;
  return x - Eigen::VectorXd::Constant(n, avg);
}

template <>
inline Eigen::VectorXd
laplacianPseudoinverseColumn<Eigen::MatrixXd>(const Eigen::MatrixXd &L, int k) {
  Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower | Eigen::Upper> solver;
  solver.compute(L);

  int n = L.cols();
  Eigen::VectorXd b = Eigen::VectorXd::Constant(n, -1.0 / n);
  b(k) += 1.;

  solver.compute(L);
  if (solver.info() != Eigen::Success) {
    // solver failed
    throw std::logic_error("Solver failed.");
  }

  Eigen::VectorXd x = solver.solve(b);
  if (solver.info() != Eigen::Success) {
    // solving failed
    throw std::logic_error("Solving failed.!");
  }
  double avg = x.sum() / n;
  Eigen::VectorXd vec = x - Eigen::VectorXd::Constant(n, avg);

  return vec;
}

template <class MatrixType = Eigen::SparseMatrix<double>>
inline Eigen::VectorXd laplacianPseudoinverseColumn(const MatrixType &L,
                                                    int k) {
  Eigen::SparseLU<MatrixType, Eigen::COLAMDOrdering<int>> solver;
  solver.analyzePattern(L);
  solver.factorize(L);

  int n = L.cols();
  Eigen::VectorXd b = Eigen::VectorXd::Constant(n, -1.0 / n);
  b(k) += 1.;

  solver.compute(L);
  if (solver.info() != Eigen::Success) {
    // solver failed
    throw std::logic_error("Solver failed.");
  }

  Eigen::VectorXd x = solver.solve(b);
  if (solver.info() != Eigen::Success) {
    // solving failed
    throw std::logic_error("Solving failed.!");
  }
  double avg = x.sum() / n;
  Eigen::VectorXd vec = x - Eigen::VectorXd::Constant(n, avg);

  return vec;
}

std::vector<Eigen::VectorXd>
laplacianPseudoinverseColumns(const Eigen::SparseMatrix<double> &L,
                              std::vector<NetworKit::node> indices);
std::vector<NetworKit::Vector>
laplacianPseudoinverseColumns(const NetworKit::CSRMatrix &laplacian,
                              std::vector<NetworKit::node> indices,
                              double tol = 0.1);

void updateLaplacian(Eigen::SparseMatrix<double> &laplacian, NetworKit::node a,
                     NetworKit::node b);

// Update formula for the pseudoinverse of the Laplacian as an edge is added to
// the graph. Add edge (i, j) and compute difference between the k-th column of
// the pseudoinverse. Takes the columns of the pseudoinverse that correspond to
// the vertices i and j. Add this to the old to get the new.
Eigen::VectorXd
laplacianPseudoinverseColumnDifference(Eigen::MatrixXd const &lpinv, int i,
                                       int j, int k, double conductance = 1.0);

// Update formula for the trace of the lap pinv as an edge is added to the
// graph. Add edge (i, j) and compute the difference of the traces of the
// pseudoinverses. Add this to the old to get the new.
double laplacianPseudoinverseTraceDifference(Eigen::VectorXd const &column_i,
                                             int i,
                                             Eigen::VectorXd const &column_j,
                                             int j, double conductance = 1.0);
double laplacianPseudoinverseTraceDifference(Eigen::MatrixXd const &lpinv,
                                             int i, int j,
                                             double conductance = 1.0);
double laplacianPseudoinverseTraceDifference(
    Eigen::MatrixXd const &lpinv, std::vector<NetworKit::Edge> edges,
    std::vector<double> conductances = std::vector<double>());

// Update formula for the trace of the lpinv as two edges are added to the
// graph.
// double laplacianPseudoinverseTraceDifference2(Eigen::MatrixXd const & lpinv,
// Edge e1, Edge e2, double conductance1=1.0, double conductance2=1.0);
double laplacianPseudoinverseTraceDifference2(Eigen::MatrixXd const &lpinv,
                                              NetworKit::Edge e1,
                                              NetworKit::Edge e2,
                                              double conductance1 = 1.0,
                                              double conductance2 = 1.0);

void updateLaplacianPseudoinverse(Eigen::MatrixXd &lpinv, NetworKit::Edge e,
                                  double conductance = 1.0);
Eigen::MatrixXd updateLaplacianPseudoinverseCopy(Eigen::MatrixXd const &lpinv,
                                                 NetworKit::Edge e,
                                                 double conductance = 1.0);

// Compute a stochastic approximation of the diagonal of the Moore-Penrose
// pseudoinverse of the laplacian matrix of a graph.
Eigen::VectorXd approxLaplacianPseudoinverseDiagonal(NetworKit::Graph const &G,
                                                     double epsilon = 0.1);
Eigen::VectorXd approxEffectiveResistances(NetworKit::Graph const &G,
                                           int &pivot, double epsilon = 0.1);

#endif // LAPLACIAN_H