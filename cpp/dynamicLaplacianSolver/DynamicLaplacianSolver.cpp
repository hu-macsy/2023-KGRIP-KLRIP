#include <dynamicLaplacianSolver/DynamicLaplacianSolver.hpp>

#include <laplacian.hpp>

template <class MatrixType, class Solver>
void DynamicLaplacianSolver<MatrixType, Solver>::setup(const Graph &g,
                                                       double tolerance,
                                                       count eqnsPerRound) {
  this->laplacian = laplacianMatrix<MatrixType>(g);
  n = laplacian.rows();

  colAge.clear();
  colAge.resize(n, -1);
  cols.clear();
  cols.resize(n);
  updateVec.clear();
  updateW.clear();
  round = 0;

  rhs.clear();
  rhs.resize(omp_get_max_threads(), Eigen::VectorXd::Constant(n, -1.0 / n));

  auto t0 = std::chrono::high_resolution_clock::now();

  setup_solver();

  auto t1 = std::chrono::high_resolution_clock::now();
  computeColumnSingle(0);

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

template <class MatrixType, class Solver>
Eigen::VectorXd DynamicLaplacianSolver<MatrixType, Solver>::getColumn(node i) {
  computeColumnSingle(i);
  return cols[i];
}

template <class MatrixType, class Solver>
Eigen::MatrixXd
DynamicLaplacianSolver<MatrixType, Solver>::solve(const Eigen::MatrixXd &rhs) {
  Eigen::MatrixXd sol = solver.solve(rhs);
  if (solver.info() != Eigen::Success) {
    // solving failed
    throw std::logic_error("Solving failed.!");
  }

  Eigen::MatrixXd avgs = Eigen::VectorXd::Constant(n, 1. / n).transpose() * sol;
  sol -= Eigen::VectorXd::Constant(n, 1.) * avgs;
  computedColumns += rhs.cols();

  for (count age = solverAge; age < round; age++) {
    auto upv = updateVec[age];
    Eigen::MatrixXd intermediate1 = upv.transpose() * rhs;
    Eigen::VectorXd intermediate2 = updateW[age] * upv;
    sol -= intermediate2 * intermediate1;
  }
  return sol;
}

template <class MatrixType, class Solver>
void DynamicLaplacianSolver<MatrixType, Solver>::computeColumnSingle(node i) {
  if (colAge[i] == -1 || round - colAge[i] > ageToRecompute) {
    cols[i] = solveLaplacianPseudoinverseColumn(
        solver, rhs[omp_get_thread_num()], n, i);
    computedColumns++;
    colAge[i] = solverAge;
  }
  for (auto &age = colAge[i]; age < round; age++) {
    cols[i] -= updateVec[age] * (updateVec[age](i) * updateW[age]);
  }
}

template <class MatrixType, class Solver>
void DynamicLaplacianSolver<MatrixType, Solver>::computeColumns(
    std::vector<node> nodes) {
#pragma omp parallel for
  for (auto it = nodes.begin(); it < nodes.end(); it++) {
    computeColumnSingle(*it);
  }
}

template <class MatrixType, class Solver>
void DynamicLaplacianSolver<MatrixType, Solver>::addEdge(node u, node v) {
  computeColumns({u, v});
  const Eigen::VectorXd &colU = cols[u];
  const Eigen::VectorXd &colV = cols[v];

  volatile double R = colU(u) + colV(v) - 2. * colU(v);
  double w = 1. / (1. + R);
  assert(w < 1);

  updateVec.push_back(colU - colV);
  updateW.push_back(w);

  rChange = (colU - colV).squaredNorm() * w * static_cast<double>(n);

  laplacian.coeffRef(u, u) += 1.;
  laplacian.coeffRef(v, v) += 1.;
  laplacian.coeffRef(u, v) -= 1.;
  laplacian.coeffRef(v, u) -= 1.;

  round++;

  if (round % roundsPerSolverUpdate == 0) {
    solver.~Solver();
    new (&solver) Solver();
    setup_solver();
  }
}

template <class MatrixType, class Solver>
count DynamicLaplacianSolver<MatrixType, Solver>::getComputedColumnCount() {
  return computedColumns;
}

template <class MatrixType, class Solver>
double
DynamicLaplacianSolver<MatrixType, Solver>::totalResistanceDifferenceApprox(
    node u, node v) {
  return static_cast<double>(this->n) * (-1.0) *
         laplacianPseudoinverseTraceDifference(getColumn(u), u, getColumn(v),
                                               v);
}

template <class MatrixType, class Solver>
double
DynamicLaplacianSolver<MatrixType, Solver>::totalResistanceDifferenceExact(
    node u, node v) {
  return totalResistanceDifferenceApprox(u, v);
}

template class DynamicLaplacianSolver<
    Eigen::SparseMatrix<double>,
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>>;