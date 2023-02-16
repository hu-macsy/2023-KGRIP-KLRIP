#ifndef SLEPC_ADAPTER_H
#define SLEPC_ADAPTER_H

#include <iostream>
#include <networkit/graph/Graph.hpp>
#include <slepceps.h>
#include <vector>

static char help[] = "INTERFACE TO EIGENSOLVER \n";
/**
 * Wrapper class for computing the largest eigenvalue of a laplacian matrix
 * (symmetric). It operates on Petsc/Slepc objects.
 */
class SlepcAdapter {
public:
  /** Sets up the eigendecomposition of an input graph @a g. It creates the
   * necessary matrix format sets the corresponding options and creates the
   * eigendecomposition environment. Given the number of requested eigenparis @a
   * numberOfEigenpairs, it performs a truncated eigendecompositing of the
   * Laplacian matrix of @a g as: L = U\LambdaU^T. Since our matrices are
   * considered to be of Laplacian type by default, we deflate the eigenvector
   * that corresponds to the zero eigenvalue to boost the convergence of the
   * involved methods.
   *
   * @param	G			the graph
   * @param	offset			the number of expected edge additions so that we
   * include it in the preallocation of the matrix.
   * @param	numberOfEigenpairs	number of requested eigenpairs on the lower
   * end of the spectrum.
   */

  void setup(NetworKit::Graph const &g, NetworKit::count offset,
             unsigned int numberOfEigenpairs) {

    int arg_c = 0;
    PetscErrorCode ierr = SlepcInitialize(&arg_c, NULL, (char *)0, help);
    if (ierr) {
      throw std::runtime_error("SlepcInitialize() not working!");
    }
    if (!numberOfEigenpairs) {
      throw std::runtime_error("Requesting no eigenpairs!");
    }

    k = offset;
    n = (PetscInt)g.numberOfNodes();
    NetworKit::count m = (PetscInt)g.numberOfEdges();
    DEBUG("GRAPH INPUT: (n = ", n, " m = ", m, ")\n");

    // TODO: ADJUST FOR ALLOCATING MORE SPACE BASED ON k!
    // INSTEAD OF DEGREE(V) + 1, ALLOCATE DEGREE(V) + 1 + k
    // TO AVOID ANOTHER MALLOC (k IS SMALL COMPARED TO AVG DEGREE).

    PetscInt *nnz = (PetscInt *)malloc(n * sizeof(PetscInt));
    g.forNodes([&](NetworKit::node v) {
      nnz[v] = (PetscInt)g.degree(v) + 1; //+ offset;
    });

    // includes preallocation
    MatCreateSeqAIJ(PETSC_COMM_WORLD, n, n, 0, nnz, &A);
    MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetType(A, MATSEQAIJ);
    MatSetUp(A);

    MatSetValuesROW(g, nnz, &A);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    free(nnz);

    c = (PetscInt)numberOfEigenpairs;
    // storage for eigenpairs
    e_vectors = (double *)calloc(1, n * c * sizeof(double));
    e_values = (double *)calloc(1, (c + 1) * sizeof(double));
    // Vec x for deflation
    MatCreateVecs(A, &x, NULL);
    VecSet(x, 1.0);

    // create eps environment
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSSetOperators(eps, A, NULL);
    EPSSetProblemType(eps, EPS_HEP);
    EPSSetFromOptions(eps);
    // set deflation space
    EPSSetDeflationSpace(eps, 1, &x);
  }

  ~SlepcAdapter() {
    free(e_vectors);
    free(e_values);
    EPSDestroy(&eps);
    MatDestroy(&A);
    VecDestroy(&x);
    VecDestroyVecs(nconv + 1, &Q);
    SlepcFinalize();
  }

  /** Updates the eigensolver environment. To do so it resets the (updated)
   matrix, the initial space with the current solution (to speed up convergence)
   and the deflation space.
   * Finally reruns the eigensolver with the new setting.

   * @return Petsc error code in case of a failure of the EPSSolve().
   */

  PetscErrorCode update_eigensolver() {

    EPSSetOperators(eps, A, NULL);
    EPSSetDeflationSpace(eps, 1, &x);
    EPSSetInitialSpace(eps, nconv + 1, Q);
    PetscErrorCode ierr = run_eigensolver();
    return ierr;
  }

  /**
   * Runs the eigensolver.
   * Computes c eigenpairs in the lower side of the spectrum and one on the
   right side.

   * @return Petsc error code in case of a failure of the EPSSolve().
   */

  PetscErrorCode run_eigensolver() {

    // PetscReal error;
    // PetscReal norm;

    PetscScalar val;
    Vec vec;
    PetscInt i;
    // allocate for eigenvector
    MatCreateVecs(A, NULL, &vec);

    EPSSetDimensions(eps, c, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);
    PetscErrorCode ierr = EPSSolve(eps);
    EPSGetConverged(eps, &nconv);
    if (nconv > c)
      nconv = c;
    // allocation eigenvectors (one more for largest eigenvector)
    VecDuplicateVecs(vec, nconv + 1, &Q);

    for (i = 0; i < nconv; i++) {
      EPSGetEigenpair(eps, i, &val, NULL, vec, NULL);
      // compute relative error associated to each eigenpair
      // EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error);
      // PetscPrintf(PETSC_COMM_WORLD,"   %12f      %12g \n", (double)val,
      // (double)error); PetscPrintf(PETSC_COMM_WORLD,"\n"); VecNorm(vec,
      // NORM_2, &norm); DEBUG("Norm of evector %d : %g \n", i, norm);
      // VecView(vec, PETSC_VIEWER_STDOUT_WORLD);
      e_values[i] = (double)val;
      VecCopy(vec, Q[i]);
      for (PetscInt j = 0; j < n; j++) {
        PetscScalar w;
        VecGetValues(vec, 1, &j, &w);
        *(e_vectors + i + j * c) = (double)w;
      }
    }

    // reset for largest eigenpair
    EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
    ierr = EPSSolve(eps);
    EPSGetConverged(eps, &nconv_l);
    if (!nconv_l) {
      throw std::runtime_error("Largest eigenvalue has not converged!");
    }
    EPSGetEigenpair(eps, 0, &val, NULL, vec, NULL);
    // DEBUG("           k          ||Ax-kx||/||kx||\n"
    //"   ----------------- ------------------\n");
    // EPSComputeError(eps, 0, EPS_ERROR_RELATIVE, &error);
    // DEBUG("   %12f       %12g\n",(double)val,(double)error, "\n");
    //  store top eigenvector at the end of Q.
    VecCopy(vec, Q[i]);
    e_values[i] = val;
    VecDestroy(&vec);
    return ierr;
  }

  /**
   * Prints information regarding the eigensolver.
   */

  void info_eigensolver() {

    EPSType type;         /* type of solver */
    PetscReal error, tol; /* error and tolerance of the solver */
    PetscInt maxit, its;  /* max iterations, actual iterations */
    PetscInt nev, nc;     /* # of computed values, # of converged values */
    DEBUG(" -------------- INFO ----------------- ");
    EPSGetType(eps, &type);
    DEBUG(" SOLUTION METHOD: ", type);
    EPSGetIterationNumber(eps, &its);
    DEBUG(" ITERATION COUNT: ", its);
    EPSGetTolerances(eps, &tol, &maxit);
    DEBUG(" STOP COND: tol= ", (double)tol, " maxit= ", maxit);
    EPSGetDimensions(eps, &nev, NULL, NULL);
    DEBUG(" COMPUT EVALUES: ", nev);
    EPSGetConverged(eps, &nc);
    DEBUG(" CONVRG EVALUES: ", nc);
    PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,
                          PETSC_VIEWER_ASCII_INFO_DETAIL);
    EPSConvergedReasonView(eps, PETSC_VIEWER_STDOUT_WORLD);
    EPSErrorView(eps, EPS_ERROR_RELATIVE, PETSC_VIEWER_STDOUT_WORLD);
    PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
    EPSView(eps, PETSC_VIEWER_STDOUT_WORLD);
  }

  /**
   * @return all converged eigenvectors.
   */

  double *get_eigenpairs() const { return e_vectors; }

  /**
   * @return all converged eigenvalues.
   */

  double *get_eigenvalues() const { return e_values; }

  /**
   * @return top eigenvalue.
   */
  double get_max_eigenvalue() const { return e_values[nconv]; }

  /**
   * @return number of converged eigenvalues.
   */
  PetscInt get_nconv() const { return nconv; }

  /**
   * Computes gain(a,b) for two vertices @a a, @a b using the available
   * eigenvalue information. The bound uses middle points for approximating
   * lambdas. For the effective resistance expression, we use l_m = (l_c +
   * l_n)/2, where l_c is the eigenvalue on the cutoff value c and l_n is the
   * largest eigenvalue. For the biharmonic expression, we have l_m^2 = (l_c^2 +
   * l_n^2)/2.
   * @param      a      vertex
   * @param      b      vertex
   * @return     approximation of gain difference.
   */

  double SpectralApproximationGainDifference3(NetworKit::node a,
                                              NetworKit::node b) {
    // assert(a < n && b < n);
    double g = 0.0;
    double d = 0.0, r = 0.0;

    double constant_avg = 2.0 / (e_values[nconv] + e_values[nconv - 1]);
    double constant_avg2 = 2.0 / (e_values[nconv - 1] * e_values[nconv - 1] +
                                  e_values[nconv] * e_values[nconv]);
    double sq_diff;

    for (int i = 0; i < nconv; i++) {

      sq_diff = *(e_vectors + a * c + i) - *(e_vectors + b * c + i);
      sq_diff *= sq_diff;

      d += (1.0 / (e_values[i] * e_values[i]) - constant_avg2) * sq_diff;
      r += (1.0 / e_values[i] - constant_avg) * sq_diff;
    }
    g = (2 * constant_avg2 + d) / (1 + 2 * constant_avg + r);
    return g;
  }

  /**
   * Computes gain(a,b) for two vertices @a a, @a b using upper and lower
   * bounds. First computes the bounds and then takes average value: gain(a,b) =
   * (U_bound + L_bound)/2.
   * @param      a      vertex
   * @param      b      vertex
   * @return approximation of gain difference.
   */
  double SpectralApproximationGainDifference1(NetworKit::node a,
                                              NetworKit::node b) {

    double g = 0.0;
    double dlow = 0.0, dup = 0.0, rlow = 0.0, rup = 0.0;

    double constant_n = 1.0 / (e_values[nconv] * e_values[nconv]);
    double constant_c = 1.0 / (e_values[nconv - 1] * e_values[nconv - 1]);
    double sq_diff;

    for (int i = 0; i < nconv; i++) {

      sq_diff = *(e_vectors + a * c + i) - *(e_vectors + b * c + i);
      sq_diff *= sq_diff;
      dup += (1.0 / (e_values[i] * e_values[i]) - constant_n) * sq_diff;
      rup += (1.0 / e_values[i] - 1.0 / e_values[nconv - 1]) * sq_diff;

      dlow += (1.0 / (e_values[i] * e_values[i]) - constant_c) * sq_diff;
      rlow += (1.0 / e_values[i] - 1.0 / e_values[nconv]) * sq_diff;
    }
    g = ((constant_c + dlow) / (1.0 + 1.0 / e_values[nconv] + rlow)) +
        ((constant_n + dup) / (1.0 + 1.0 / e_values[nconv - 1] + rup));

    return (g / 2.0);
  }

  /**
   * Computes gain(a,b) for two vertices @a a, @a b using
   * separate averages for the effective resistance and the biharmonic
   * distances. First computes the averages: D_appx = (upD + lowD)/2, R_appx =
   * (upR + lowR)/2. Then computes the gain approximation as gain(a,b) = D_appx
   * / (1.0 + R_approx).
   * @param      a      vertex
   * @param      b      vertex
   * @return approximation of gain difference.
   */

  double SpectralApproximationGainDifference2(NetworKit::node a,
                                              NetworKit::node b) {

    double upD = 0.0, upR = 0.0, lowD = 0.0, lowR = 0.0;
    double g = 0.0;
    double lambda_n = 1.0 / (e_values[nconv] * e_values[nconv]);
    double lambda_c = 1.0 / (e_values[nconv - 1] * e_values[nconv - 1]);
    double sq_diff;

    for (int i = 0; i < nconv; i++) {
      sq_diff = *(e_vectors + a * c + i) - *(e_vectors + b * c + i);
      sq_diff *= sq_diff;

      lowD += (1.0 / (e_values[i] * e_values[i]) - lambda_n) * sq_diff;
      upR += (1.0 / e_values[i] - 1.0 / e_values[nconv - 1]) * sq_diff;

      upD += (1.0 / (e_values[i] * e_values[i]) - lambda_c) * sq_diff;
      lowR += (1.0 / e_values[i] - 1.0 / e_values[nconv]) * sq_diff;
    }

    return (upD + 2.0 * lambda_c + lowD + 2.0 * lambda_n) /
           (2.0 + 2.0 / e_values[nconv] + 2.0 / e_values[nconv - 1] + upR +
            lowR);
  }

  /**
   * Computes gain(a,b) for two vertices @a a, @a b similarly to
   * SpectralApproximationGainDifference2(NetworKit::node a, NetworKit::node b)
   * but the eigenvectors @a vectors and eigenvalues @a values are given as
   * input. Also the values of the cutoff @a s for the approximations is given
   * as input.
   * @param      a       a valid vertex value
   * @param      b       a valid vertex value
   * @param      vectors contains the eigenvectors
   * @param      values  contains the eigenvectors
   * @param      s       cutoff value for the approximation
   * @return approximation of gain difference.
   */

  double SpectralApproximationGainDifference2(NetworKit::node a,
                                              NetworKit::node b,
                                              double *vectors, double *values,
                                              int s) {
    double upD = 0.0, upR = 0.0, lowD = 0.0, lowR = 0.0;
    double g = 0.0;
    double lambda_n = 1.0 / (values[nconv] * values[nconv]);
    double lambda_c = 1.0 / (values[s - 1] * values[s - 1]);
    double sq_diff;

    for (int i = 0; i < s; i++) {
      sq_diff = *(vectors + a * c + i) - *(vectors + b * c + i);
      sq_diff *= sq_diff;

      lowD += (1.0 / (values[i] * values[i]) - lambda_n) * sq_diff;
      upR += (1.0 / values[i] - 1.0 / values[s - 1]) * sq_diff;

      upD += (1.0 / (values[i] * values[i]) - lambda_c) * sq_diff;
      lowR += (1.0 / values[i] - 1.0 / values[nconv]) * sq_diff;
    }

    return (upD + 2.0 * lambda_c + lowD + 2.0 * lambda_n) /
           (2.0 + 2.0 / values[nconv] + 2.0 / values[s - 1] + upR + lowR);
  }

  /**
   * Computes gain(a,b) for two vertices @a a, @a b similarly to
   * SpectralApproximationGainDifference1(NetworKit::node a, NetworKit::node b)
   * but the eigenvectors @a vectors and eigenvalues @a values are given as
   * input. Also the values of the cutoff @a s for the approximations is given
   * as input.
   * @param      a       a valid vertex value
   * @param      b       a valid vertex value
   * @param      vectors contains the eigenvectors
   * @param      values  contains the eigenvectors
   * @param      s       cutoff value for the approximation
   * @return approximation of gain difference.
   */

  double SpectralApproximationGainDifference1(NetworKit::node a,
                                              NetworKit::node b,
                                              double *vectors, double *values,
                                              int s) {
    double g = 0.0;
    double dlow = 0.0, dup = 0.0, rlow = 0.0, rup = 0.0;

    double lambda_n = 1.0 / (values[nconv] * values[nconv]);
    double lambda_c = 1.0 / (values[s - 1] * values[s - 1]);
    double sq_diff;

    for (int i = 0; i < s; i++) {
      sq_diff = *(vectors + a * c + i) - *(vectors + b * c + i);
      sq_diff *= sq_diff;
      dup += (1.0 / (values[i] * values[i]) - lambda_n) * sq_diff;
      rup += (1.0 / values[i] - 1.0 / values[s - 1]) * sq_diff;

      dlow += (1.0 / (values[i] * values[i]) - lambda_c) * sq_diff;
      rlow += (1.0 / values[i] - 1.0 / values[nconv]) * sq_diff;
    }

    g = ((2.0 * lambda_c + dlow) / (1.0 + 2.0 / values[nconv] + rlow)) +
        ((2.0 * lambda_n + dup) / (1.0 + 2.0 / values[s - 1] + rup));

    return (g / 2.0);
  }

  /**
   * Computes gain(a,b) for two vertices @a a, @a b using
   * middle points for approximating lambdas.
   * Similar to SpectralApproximationGainDifference3(NetworKit::node a,
   * NetworKit::node b) but the eigenvectors @a vectors and eigenvalues @a
   * values are given as input. Also the values of the cutoff @a s for the
   * approximations is given as input.
   * @param      a       a valid vertex value
   * @param      b       a valid vertex value
   * @param      vectors contains the eigenvectors
   * @param      values  contains the eigenvectors
   * @param      s       cutoff value for the approximation
   * @return approximation of gain difference.
   */

  double SpectralApproximationGainDifference3(NetworKit::node a,
                                              NetworKit::node b,
                                              double *vectors, double *values,
                                              int s) {
    // assert(a < n && b < n);

    double g = 0.0;
    double d = 0.0, r = 0.0;

    double constant_avg = 2.0 / (values[nconv] + values[s - 1]);
    double constant_avg2 =
        2.0 / (values[s - 1] * values[s - 1] + values[nconv] * values[nconv]);
    double sq_diff;

    for (int i = 0; i < nconv; i++) {

      sq_diff = *(vectors + a * c + i) - *(vectors + b * c + i);
      sq_diff *= sq_diff;

      d += (1.0 / (values[i] * values[i]) - constant_avg2) * sq_diff;
      r += (1.0 / values[i] - constant_avg) * sq_diff;
    }
    g = (2 * constant_avg2 + d) / (1 + 2 * constant_avg + r);
    return g;
  }

  /** Computes a spectral approximation of the total effective resistance using
   * the available eigenvalues.
   * @return approximation of total effective resistance
   */

  double SpectralToTalEffectiveResistance() {
    double Sum = 1.0 / e_values[nconv];
    for (int i = 0; i < nconv; i++)
      Sum += 1.0 / e_values[i];
    return n * Sum;
  }

  /** Computes a spectral approximation of the total effective resistance using
   * @a s eigenvalues.
   * @return approximation of total effective resistance
   */
  double SpectralToTalEffectiveResistance(int s) {
    assert(s <= nconv);
    double Sum = 1.0 / e_values[nconv];
    for (int i = 0; i < s; i++)
      Sum += 1.0 / e_values[i];
    return n * Sum;
  }

  /** Adds the elements that correspond to edge @a u, @a v into the operator
    matrix.
    * Currently implemented for unweighted entries.
    * @param      u       a valid vertex value
    * @param      v       a valid vertex value

   */
  void addEdge(NetworKit::node u, NetworKit::node v) {

    if (u == v) {
      std::cout << "Warning: Graph has edge with equal target and destination!";
      return;
    }

    PetscInt a = (PetscInt)u;
    PetscInt b = (PetscInt)v;
    PetscScalar w = 1.0;
    PetscScalar nw = -1.0;

    MatSetValues(A, 1, &a, 1, &a, &w, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &b, &w, ADD_VALUES);
    MatSetValues(A, 1, &a, 1, &b, &nw, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &a, &nw, ADD_VALUES);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  }

private:
  /**
   * Creates/preallocates a Petsc matrix object @a A based on the relevant graph
   * information.
   *
   * @param       comm            MPI_Comm communicator for Petsc.
   * @param	m		number of rows of matrix @a A / number of nodes of
   * graph.
   * @param	n		number of columns of matrix @a A / number of nodes of
   * graph.
   * @param       nz              If @a nnz[] is NULL @a nz is the number of
   * nnz. Else is 0.
   * @param       nnz[]           the nnz/degree information for each row/node.
   */
  void MatCreateSeqAIJMP(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt nz,
                         const PetscInt nnz[], Mat *A) {
    MatCreate(comm, A);
    MatSetSizes(*A, m, n, m, n);
    MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetFromOptions(*A);
    MatSetType(*A, MATSEQAIJ);
    MatSeqAIJSetPreallocation(*A, nz, (PetscInt *)nnz);
  }

  /**
   * Sets values of a Petsc matrix @a A based on an input graph information.
   * The values are inserted element by element.
   *
   * @param       g            Graph from witch we set the values of @a A
   * @param	A	     Petsc type matrix.
   */
  void MatSetValuesELM(NetworKit::Graph const &g, Mat *A) {
    g.forEdges([&](NetworKit::node u, NetworKit::node v, double w) {
      if (u == v) {
        std::cout
            << "Warning: Graph has edge with equal target and destination!";
      }
      PetscInt a = (PetscInt)u;
      PetscInt b = (PetscInt)v;
      PetscScalar vv = (PetscScalar)w;
      PetscScalar nv = (PetscScalar)-w;
      // DONT MIX ADD_VALUES AND INSERT_VALUES
      MatSetValues(*A, 1, &a, 1, &a, &vv, ADD_VALUES);
      MatSetValues(*A, 1, &b, 1, &b, &vv, ADD_VALUES);
      MatSetValues(*A, 1, &a, 1, &b, &nv, ADD_VALUES);
      MatSetValues(*A, 1, &b, 1, &a, &nv, ADD_VALUES);
    });
  }

  /**
   * Sets values of a Petsc matrix @a A based on an input graph information.
   * The values are inserted row by row.
   *
   * @param       g            Graph from witch we set the values of @a A
   * @param       nnz           the nnz/degree information for each row/node.
   * @param	A	     Petsc type matrix.
   */
  void MatSetValuesROW(NetworKit::Graph const &g, PetscInt *nnz, Mat *A) {
    // g.balancedParallelForNodes([&](NetworKit::node v) {
    g.forNodes([&](const NetworKit::node v) {
      double weightedDegree = 0.0;
      PetscInt *col = (PetscInt *)malloc(nnz[v] * sizeof(PetscInt));
      PetscScalar *val = (PetscScalar *)malloc(nnz[v] * sizeof(PetscScalar));
      unsigned int idx = 0;
      g.forNeighborsOf(v, [&](const NetworKit::node u, double w) { // - adj  mat
        // exclude diag. (would be subtracted by adj weight
        if (u != v) {
          weightedDegree += w;
        }
        col[idx] = (PetscInt)u;
        val[idx] = -(PetscScalar)w;
        idx++;
      });
      col[idx] = v;
      val[idx] = weightedDegree;
      PetscInt a = (PetscInt)v;
      MatSetValues(*A, 1, &a, nnz[v], col, val, INSERT_VALUES);
    });
  }

  EPS eps;                        /* eigenproblem solver context*/
  Mat A;                          /* operator matrix */
  PetscInt n;                     /* size of matrix */
  Vec x;                          /* vector representing the nullspace */
  PetscErrorCode ierr;            /* diagnostic: petscerror */
  PetscInt c, nconv, nconv_l = 0; /* # requested values, # converged low values,
                                     # converged top value */
  double *e_vectors; /* stores the eigenvectors (of size n*nconv) */
  double *e_values;  /* stores eigenvalues (of size nconv + 1) */
  Vec *Q;
  NetworKit::count k;
};

#endif // SLEPC_ADAPTER_H
