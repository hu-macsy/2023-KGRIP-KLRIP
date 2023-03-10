#include <dynamicLaplacianSolver/LamgDynamicLaplacianSolver.hpp>

#include <laplacian.hpp>

void LamgDynamicLaplacianSolver::setup(const Graph &g, double tol,
                                       count eqnsPerRound) {
  this->laplacian = NetworKit::CSRMatrix::laplacianMatrix(g);
  n = g.numberOfNodes();

  this->tolerance = tol;

  colAge.resize(n, -1);
  cols.resize(n);
  solverAge = 0;

  auto t0 = std::chrono::high_resolution_clock::now();

  lamg.~Lamg<CSRMatrix>();
  new (&lamg) Lamg<CSRMatrix>(tolerance);
  lamg.setup(laplacian);

  auto t1 = std::chrono::high_resolution_clock::now();

  getColumn(0);

  auto t2 = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd u = Eigen::VectorXd::Constant(n, 1.);
  for (int i = 0; i < 1000; i++) {
    u += u * 0.1;
  }
  volatile double a = u(0);
  auto t3 = std::chrono::high_resolution_clock::now();
  double duration0 = (t1 - t0).count();
  double duration1 = (t2 - t1).count();
  double duration2 = (t3 - t2).count() / 1000;

  ageToRecompute = std::max(static_cast<int>(duration1 / duration2), 1);
  roundsPerSolverUpdate =
      std::max(static_cast<int>(duration0 / duration2 / eqnsPerRound), 1);
}

void LamgDynamicLaplacianSolver::solveSingleColumn(node u) {
  Vector rhs(n, -1.0 / static_cast<double>(n));
  Vector x(n);
  rhs[u] += 1.;
  auto status = lamg.solve(rhs, x);
  assert(status.converged);

  double avg = x.transpose() * Vector(n, 1.0 / static_cast<double>(n));
  x -= avg;

  cols[u] = nk_to_eigen(x);

  colAge[u] = solverAge;
  computedColumns++;
}

std::vector<NetworKit::Vector> LamgDynamicLaplacianSolver::parallelSolve(
    std::vector<NetworKit::Vector> &rhss) {
  auto size = rhss.size();
  std::vector<NetworKit::Vector> xs(size, NetworKit::Vector(n));
  // lamg.parallelSolve(rhss, xs);
  for (count i = 0; i < size; ++i) {
    lamg.solve(rhss[i], xs[i]);
  }

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    auto &x = xs[i];
    auto &rhs = rhss[i];
    double avg = x.transpose() * Vector(n, 1.0 / static_cast<double>(n));
    x -= avg;

    for (count r = solverAge; r < round; r++) {
      auto upv = eigen_to_nk(updateVec[r]);
      x -= upv * (NetworKit::Vector::innerProduct(upv, rhs) * updateW[r]);
    }
  }

  computedColumns += size;

  return xs;
}

Eigen::VectorXd &LamgDynamicLaplacianSolver::getColumn(node u) {
  computeColumns({u});
  return cols[u];
}

void LamgDynamicLaplacianSolver::computeColumns(std::vector<node> nodes) {
  // Determine which nodes need the solver
  std::vector<node> nodes_to_solve;
  for (auto u : nodes) {
    if (round - colAge[u] >= ageToRecompute || colAge[u] == -1) {
      nodes_to_solve.push_back(u);
    }
  }

  // Solve
  auto nodes_to_solve_count = nodes_to_solve.size();
  std::vector<NetworKit::Vector> rhss(
      nodes_to_solve_count, NetworKit::Vector(n, -1. / static_cast<double>(n)));
  for (int i = 0; i < nodes_to_solve_count; i++) {
    auto u = nodes_to_solve[i];
    rhss[i][u] += 1.;
  }
  std::vector<NetworKit::Vector> xs(nodes_to_solve_count, NetworKit::Vector(n));
  lamg.parallelSolve(rhss, xs);
  computedColumns += nodes_to_solve_count;

  // Ensure slns average 0
  for (int i = 0; i < nodes_to_solve_count; i++) {
    auto &x = xs[i];
    auto u = nodes_to_solve[i];
    double avg = NetworKit::Vector::innerProduct(
        x, NetworKit::Vector(n, 1.0 / static_cast<double>(n)));
    x -= avg;

    cols[u] = nk_to_eigen(x);
    colAge[u] = solverAge;
  }

  // Update
  for (int i = 0; i < nodes.size(); i++) {
    auto u = nodes[i];
    auto &col = cols[u];
    for (auto &r = colAge[u]; r < round; r++) {
      auto &upv = updateVec[r];
      col -= upv * (upv[u] * updateW[r]);
    }
    colAge[u] = round;
  }
}

void LamgDynamicLaplacianSolver::addEdge(node u, node v) {
  computeColumns({u, v});
  auto colU = getColumn(u);
  auto colV = getColumn(v);

  double R = colU(u) + colV(v) - 2 * colU(v);
  double w = (1. / (1. + R));
  assert(w < 1.);
  auto upv = colU - colV;

  rChange = upv.squaredNorm() * w * static_cast<double>(n);

  laplacian.setValue(u, u, laplacian(u, u) + 1.);
  laplacian.setValue(v, v, laplacian(v, v) + 1.);
  laplacian.setValue(u, v, laplacian(u, v) - 1.);
  laplacian.setValue(v, u, laplacian(v, u) - 1.);

  updateVec.push_back(upv);
  updateW.push_back(w);
  round++;

  if (round % roundsPerSolverUpdate == 0) {
    lamg.~Lamg<CSRMatrix>();
    new (&lamg) Lamg<CSRMatrix>(tolerance);
    lamg.setup(laplacian);

    solverAge = round;
  }
}

count LamgDynamicLaplacianSolver::getComputedColumnCount() {
  return computedColumns;
}

double LamgDynamicLaplacianSolver::totalResistanceDifferenceApprox(node u,
                                                                   node v) {
  return static_cast<double>(this->n) * (-1.0) *
         laplacianPseudoinverseTraceDifference(getColumn(u), u, getColumn(v),
                                               v);
}

double LamgDynamicLaplacianSolver::totalResistanceDifferenceExact(node u,
                                                                  node v) {
  return totalResistanceDifferenceApprox(u, v);
}