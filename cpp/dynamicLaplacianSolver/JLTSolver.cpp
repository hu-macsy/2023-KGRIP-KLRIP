#include <dynamicLaplacianSolver/JLTSolver.hpp>

#include <laplacian.hpp>

template <class MatrixType, class DynamicSolver>
void JLTSolver<MatrixType, DynamicSolver>::setup(const Graph &g,
                                                 double tolerance,
                                                 count eqnsPerRound) {
  n = g.numberOfNodes();
  m = g.numberOfEdges();

  epsilon = tolerance;

  l = jltDimension(eqnsPerRound, epsilon);

  G = g;
  solver.setup(g, 0.01, 2 * l + 2);

  G.indexEdges();
  incidence = incidenceMatrix(G);

  computeIntermediateMatrices();
}

template <class MatrixType, class DynamicSolver>
void JLTSolver<MatrixType, DynamicSolver>::computeColumns(
    std::vector<node> nodes) {
  // pass
}

template <class MatrixType, class DynamicSolver>
void JLTSolver<MatrixType, DynamicSolver>::addEdge(node u, node v) {
  G.addEdge(u, v);
  solver.addEdge(u, v);
  m++;

  G.indexEdges();
  incidence = incidenceMatrix(G);
  computeIntermediateMatrices();
}

template <class MatrixType, class DynamicSolver>
count JLTSolver<MatrixType, DynamicSolver>::getComputedColumnCount() {
  return solver.getComputedColumnCount();
}

template <class MatrixType, class DynamicSolver>
double
JLTSolver<MatrixType, DynamicSolver>::totalResistanceDifferenceApprox(node u,
                                                                      node v) {
  double R = effR(u, v);
  double phiNormSq = phiNormSquared(u, v);
  return n / (1. + R) * phiNormSq;
}

template <class MatrixType, class DynamicSolver>
double
JLTSolver<MatrixType, DynamicSolver>::totalResistanceDifferenceExact(node u,
                                                                     node v) {
  return solver.totalResistanceDifferenceExact(u, v);
}

template <class MatrixType, class DynamicSolver>
double JLTSolver<MatrixType, DynamicSolver>::effR(node u, node v) {
  return (PBL.col(u) - PBL.col(v)).squaredNorm();
}

template <class MatrixType, class DynamicSolver>
double JLTSolver<MatrixType, DynamicSolver>::phiNormSquared(node u, node v) {
  return (PL.col(u) - PL.col(v)).squaredNorm();
}

template <class MatrixType, class DynamicSolver>
int JLTSolver<MatrixType, DynamicSolver>::jltDimension(int n, double epsilon) {
  return std::max(std::log(n) / (epsilon * epsilon), 1.);
}

template <class MatrixType, class DynamicSolver>
void JLTSolver<MatrixType, DynamicSolver>::computeIntermediateMatrices() {
  // Generate projection matrices

  auto random_projection = [&](count n_rows, count n_cols) {
    auto normal = [&](int) { return d(gen) / std::sqrt(n_rows); };
    Eigen::MatrixXd P = Eigen::MatrixXd::NullaryExpr(n_rows, n_cols, normal);
    Eigen::VectorXd avg = P * Eigen::VectorXd::Constant(n_cols, 1. / n_cols);
    P -= avg * Eigen::MatrixXd::Constant(1, n_cols, 1.);
    return P;
  };

  P_n = random_projection(l, n);
  P_m = random_projection(l, m);

  // Compute columns of P L^\dagger and P' B^T L^\dagger where B is the
  // incidence matrix of G We first compute the transposes of the targets. For
  // the first, solve LX = P^T, for the second solve LX = B P^T

  Eigen::MatrixXd rhs1 = P_n.transpose();
  Eigen::MatrixXd rhs2 = incidence * P_m.transpose();

  PL = solver.solve(rhs1);
  PBL = solver.solve(rhs2);

  PL.transposeInPlace();
  PBL.transposeInPlace();
}

template class JLTSolver<Eigen::SparseMatrix<double>, SparseLUSolver>;
