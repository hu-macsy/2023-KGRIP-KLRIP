#include <dynamicLaplacianSolver/JLTLamgSolver.hpp>

#include <laplacian.hpp>

void JLTLamgSolver::setup(const Graph &g, double tolerance,
                          count eqnsPerRound) {
  n = g.numberOfNodes();
  m = g.numberOfEdges();

  epsilon = tolerance;

  l = jltDimension(eqnsPerRound, epsilon);
  G = g;
  solver.setup(g, 0.0001, 2 * l + 2);

  G.indexEdges();
  incidence = CSRMatrix::incidenceMatrix(G);

  computeIntermediateMatrices();
}

void JLTLamgSolver::computeColumns(std::vector<node> nodes) {
  // pass
}

void JLTLamgSolver::addEdge(node u, node v) {
  G.addEdge(u, v);
  solver.addEdge(u, v);
  m++;

  G.indexEdges();
  incidence = CSRMatrix::incidenceMatrix(G);
  computeIntermediateMatrices();
}

count JLTLamgSolver::getComputedColumnCount() {
  return solver.getComputedColumnCount();
}

double JLTLamgSolver::totalResistanceDifferenceApprox(node u, node v) {
  double R = effR(u, v);
  double phiNormSq = phiNormSquared(u, v);

  return n / (1. + R) * phiNormSq;
}

double JLTLamgSolver::totalResistanceDifferenceExact(node u, node v) {
  return solver.totalResistanceDifferenceExact(u, v);
}

double JLTLamgSolver::effR(node u, node v) {
  auto r = PBL.column(u) - PBL.column(v);
  return NetworKit::Vector::innerProduct(r, r);
}

double JLTLamgSolver::phiNormSquared(node u, node v) {
  auto r = PL.column(u) - PL.column(v);
  return NetworKit::Vector::innerProduct(r, r);
}

int JLTLamgSolver::jltDimension(int perRound, double epsilon) {
  return std::max(std::log(perRound) / (epsilon * epsilon), 1.);
}

void JLTLamgSolver::computeIntermediateMatrices() {

  // Generate projection matrices

  auto random_projection = [&](count n_rows,
                               count n_cols) -> NetworKit::DenseMatrix {
    auto normal = [&]() { return d(gen) / std::sqrt(n_rows); };
    auto normal_expr = [&](NetworKit::count, NetworKit::count) {
      return normal();
    };
    DenseMatrix P =
        nk_matrix_from_expr<DenseMatrix>(n_rows, n_cols, normal_expr);
    NetworKit::Vector avg = P * NetworKit::Vector(n_cols, 1. / n_cols);
    P -= NetworKit::Vector::outerProduct<DenseMatrix>(
        avg, NetworKit::Vector(n_cols, 1.));
    return P;
  };

  auto P_n = random_projection(l, n);
  auto P_m = random_projection(l, m);

  // Compute columns of P L^\dagger and P' B^T L^\dagger where B is the
  // incidence matrix of G We first compute the transposes of the targets. For
  // the first, solve LX = P^T, for the second solve LX = B P^T

  // CSRMatrix rhs1 = P_n.transpose();
  CSRMatrix rhs2_mat = incidence * nk_dense_to_csr(P_m.transpose());
  std::vector<NetworKit::Vector> rhss1;
  std::vector<NetworKit::Vector> rhss2;

  for (int i = 0; i < l; i++) {
    rhss1.push_back(P_n.row(i).transpose());
    rhss2.push_back(rhs2_mat.column(i));
  }

  auto xs1 = solver.parallelSolve(rhss1);
  auto xs2 = solver.parallelSolve(rhss2);

  PL = DenseMatrix(l, n);
  PBL = DenseMatrix(l, n);

  for (int i = 0; i < l; i++) {
    for (int j = 0; j < n; j++) {
      PL.setValue(i, j, xs1[i][j]);
      PBL.setValue(i, j, xs2[i][j]);
    }
  }
}