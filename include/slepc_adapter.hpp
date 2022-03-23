#ifndef SLEPC_ADAPTER_H
#define SLEPC_ADAPTER_H

// TODO: add path to petsc that is required by slec
// following ex11.c from slepc/src/eps/tutorials/

#include <vector>
#include <iostream>
#include <networkit/graph/Graph.hpp>
#include <slepceps.h>


#define EIGENVALUE_MULTIPLIER 4000

static char help[] = "INTERFACE TO EIGENSOLVER \n";

class SlepcAdapter {
public:
    void setup(NetworKit::Graph const & g, NetworKit::count offset)  {

        int arg_c = 0;
	char ** v = NULL;
	ierr = SlepcInitialize(&arg_c,NULL,(char*)0,help);
	if (ierr) {
	  throw std::runtime_error("SlepcInitialize not working!");
	}
	
	n = (PetscInt)g.numberOfNodes();
	NetworKit::count m = (PetscInt)g.numberOfEdges();

	// TODO: ADJUST FOR ALLOCATING MORE SPACE BASED ON k!
	// INSTEAD OF DEGREE(V) + 1, ALLOCATE DEGREE(V) + 1 + k
	// TO AVOID ANOTHER MALLOC (k IS SMALL COMPARED TO AVG DEGREE).	

	PetscInt * nnz = (PetscInt *) malloc( n * sizeof( PetscInt ) );	
	g.forNodes([&](NetworKit::node v) {
		     assert(v < n);
		     nnz[v] = (PetscInt) g.degree(v) + 1; //+ offset;
		   });
	
	// =================================================================
	// SEQUENTIAL SPARSE MATRIX CREATION (#rows, #columns)	
	MatCreateSeqAIJ(PETSC_COMM_WORLD, n, n, 0, nnz, &A); // includes preallocation
	MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
	// TODO: TEMP. PLEASE REMOVE FOLLOWING LINE!!
	MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); // ignores new malloc error
	MatSetType(A, MATSEQAIJ); 
	MatSetFromOptions(A);
	MatSetUp(A);
	std::cout << "INFO: MATRIX IS CREATED SUCCESSFULLY!\n";
	// =================================================================
       	
	// SETTING MATRIX ELEMENTS
	MatSetValues_Row(g, nnz, &A);
	// ALWAYS ASSEMBLY AFTER MATSETVALUES().
	// TODO: MAT_FINAL_ASSEMBLY OR MAT_FLUSH_ASSEMBLY

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	std::cout << "INFO: MATRIX PRINTING SUCCESSFULLY AFTER INSERTION!\n";	
	free(nnz);
    }

  
    ~SlepcAdapter() {
      
      free(e_vectors);
      free(e_values);
      EPSDestroy(&eps);
      EPSDestroy(&eps_l);
      MatDestroy(&A);
      VecDestroy(&x);
      SlepcFinalize();

    }


  PetscErrorCode update_eigensolver() {

    // RESET 'NEW' MATRIX (AFTER ADDED EDGE)
    ierr = EPSSetOperators(eps, A, NULL); CHKERRQ(ierr);

    // RESET DEFLATION SPACE
    ierr = EPSSetDeflationSpace(eps, 1, &x); CHKERRQ(ierr);


    // RESET 'NEW' MATRIX (AFTER ADDED EDGE)
    ierr = EPSSetOperators(eps_l, A, NULL); CHKERRQ(ierr);

    // RESET DEFLATION SPACE
    ierr = EPSSetDeflationSpace(eps_l, 1, &x); CHKERRQ(ierr);


    
    // SET EPSSetInitialSpace() TO EXPLOIT INITIAL SOLUTION
    
    run_eigensolver();
    info_eigensolver(); 
    set_eigenpairs();
	
    std::cout << "INFO: UPDATE EIGENSOLVER SUCCESSFULLY! \n";
    return ierr;
    }

  

  // ROUTINE TO SET THE EIGENSOLVER PRIOR TO EXECUTION
  
  PetscErrorCode set_eigensolver(NetworKit::count numberOfEigenpairs) {
    if ( !numberOfEigenpairs ) {
      std::cout << "WARN: NO EIGENPAIRS ARE TO BE COMPUTED.\n";
      return 0;
    }
    
    std::cout << " INFO: SETTING NUMBER OF EIGENPAIRS = " << numberOfEigenpairs << "\n";
    c = (PetscInt) numberOfEigenpairs;
    // storage for eigenpairs
    e_vectors = (double *) calloc(1, n * c * sizeof(double));
    e_values = (double *) calloc(1, (c + 1) * sizeof(double));
    // Vec x;
    ierr = MatCreateVecs(A, &x, NULL); CHKERRQ(ierr); // CLASS VARIABLE
    ierr = VecSet(x, 1.0); CHKERRQ(ierr);

    // solver for c smallest eigenvalues
    ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
    ierr = EPSSetOperators(eps, A, NULL); CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(ierr);
    // # of eigenpairs
    ierr = EPSSetDimensions(eps, c, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE); CHKERRQ(ierr);
    //ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE); // EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);
    ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);

    // solver for largest eigenvalue
    ierr = EPSCreate(PETSC_COMM_WORLD, &eps_l); CHKERRQ(ierr);
    ierr = EPSSetOperators(eps_l, A, NULL); CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps_l, EPS_HEP); CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(eps_l, EPS_LARGEST_MAGNITUDE);
    //EPSSetWhichEigenpairs(eps_l,EPS_LARGEST_REAL);
    ierr = EPSSetFromOptions(eps_l);
    

    ierr = EPSSetDeflationSpace(eps, 1, &x); CHKERRQ(ierr);
    ierr = EPSSetDeflationSpace(eps_l, 1, &x); CHKERRQ(ierr);

    std::cout << "INFO: SET EIGENSOLVER SUCCESSFULLY! \n";
    return ierr;
  }
    /* ========================================================================================== */ 


  // ROUTINE TO RUN THE EIGENSOLVER
  PetscErrorCode run_eigensolver() {
    
    ierr = EPSSolve(eps); CHKERRQ(ierr);
    ierr = EPSSolve(eps_l); CHKERRQ(ierr);
    std::cout << "INFO: RUN EIGENSOLVER SUCCESSFULLY! \n";
    return ierr;
  }

  // ROUTINE TO RUN DIAGNOSTICS ON THE EIGENSOLVER
  // TODO: NOT ONLY DIAGNOSTICS! SETTING NCONV IS IMPORTANT AND IS DONE HERE!!
  PetscErrorCode info_eigensolver() {

      ierr = EPSGetType(eps, &type); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type); CHKERRQ(ierr);
      EPSGetIterationNumber(eps, &its);
      PetscPrintf(PETSC_COMM_WORLD," Iteration count: %D\n", its);
      EPSGetTolerances(eps, &tol, &maxit);
      PetscPrintf(PETSC_COMM_WORLD," Stopping cond: tol=%.4g, maxit=%D\n", (double)tol, maxit);
      ierr = EPSGetDimensions(eps, &nev, NULL, NULL); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD," Requested eigenvalue count: %D\n", c);
      ierr = PetscPrintf(PETSC_COMM_WORLD," Computed eigenvalue count: %D\n", nev);
      ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD," Converged eigenvalue count: %D\n", nconv);
      if (nconv > c) nconv = c;
      ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);
      CHKERRQ(ierr);
      ierr = EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

      
      std::cout << "INFO: INFO EIGENSOLVER SUCCESSFULLY! \n";
      return ierr;
    }
  /* ========================================================================================== */
    

  // TODO: IMPORTANT I HAVENT COMPUTED THE LARGEST EIGENVALUE YET,
  // ONLY APPROXIMATE IT TO BE c TIMES LARGER THAN THE CURRENTLY LARGEST ONE
  // (FROM THE SET OF COMPUTED EVALUES).
  
  void set_eigenpairs() {
    PetscScalar val;
    Vec vec;
    // create once and overwrite in loop
    MatCreateVecs(A, NULL, &vec);
    PetscInt i;
    for (i = 0 ; i < nconv; i++) {
      EPSGetEigenpair(eps, i, &val, NULL, vec, NULL);
      //Compute relative error associated to each eigenpair
      EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error);
      PetscPrintf(PETSC_COMM_WORLD,"   %12f      %12g\n", (double)val, (double)error);
      PetscPrintf(PETSC_COMM_WORLD,"\n");
      	e_values[i] = (double) val;
      	for(PetscInt j = 0; j < n; j++) {
      	  PetscScalar w;
      	  //VecGetValues(Vec x,PetscInt ni,const PetscInt ix[],PetscScalar y[])
      	  VecGetValues(vec, 1, &j, &w);
      	  *(e_vectors + i*c + j ) = (double) w; 
      	}
    }

    EPSType type_l;
    ierr = EPSGetType(eps_l, &type_l); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type_l); CHKERRQ(ierr);
    EPSGetConverged(eps_l,&nconv_l);
    PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs FOR LARGE EIGENVALUE: %D\n\n",nconv_l);
    if ( !nconv_l ) {
      std::cout << "WARN: LARGEST EIGENVALUE IS NOT COMPUTED.\n";
    }
    assert(nconv_l == 1);
    PetscPrintf(PETSC_COMM_WORLD,
		"           k          ||Ax-kx||/||kx||\n"
		"   ----------------- ------------------\n");
    
    EPSGetEigenvalue(eps_l, 0, &val, NULL);
    EPSComputeError(eps_l, 0, EPS_ERROR_RELATIVE, &error);
    PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12g\n",(double)val,(double)error);
    PetscPrintf(PETSC_COMM_WORLD,"\n");

    std::cout << "INFO: i = " << i <<" \n";
    e_values[i] = val; //EIGENVALUE_MULTIPLIER * e_values[i-1];
    std::cout << "INFO: e_values[i+1] = " << e_values[i] <<" \n";

    
    VecDestroy(&vec);
    std::cout << "INFO: RUN SETTING_EIGENPAIRS SUCCESSFULLY! \n";
  }


  double * get_eigenpairs() const {return e_vectors;}

  double * get_eigenvalues() const {return e_values;}
  
  // TODO: rename to totalResistanceDifferenceExact
  // input solver, a , b --- require c, e_vectors, e_values 
  double SpectralApproximationGainDifference(NetworKit::node a, NetworKit::node b) {
    double * vectors = get_eigenpairs();
    double * values = get_eigenvalues();

    // std::cout << " eigenvalues are:\n [ ";
    // for (int i = 0 ; i < c + 1; i++)
    //   std::cout << values[i] << " ";
    //  std::cout << "]\n";

    double g = 0.0;
    double dlow = 0.0, dup = 0.0, rlow = 0.0, rup = 0.0;

    assert(values[nconv] > 0);
    double constant_n = 1.0/(values[nconv] * values[nconv]);
    double constant_c = 1.0/(values[nconv-1] * values[nconv-1]);

    double sq_diff;
    
    for (int i = 0 ; i < nconv; i++) {
      assert(values[i] > 0);
      sq_diff = *(vectors+a*c+i) - *(vectors+b*c+i);
      sq_diff *= sq_diff;
      dlow += (1.0/(values[i] * values[i]) - constant_n) * sq_diff;
      dup += (1.0/(values[i] * values[i]) - constant_c) * sq_diff;
      rlow += (1.0/values[i] - 1.0/values[nconv]) * sq_diff;
      rup += (1.0/values[i] - 1.0/values[nconv-1]) * sq_diff;      
    }

    
    g = ( (constant_c + dlow)/ (1.0 + 2.0/values[nconv] + rlow)  ) +
        ( (constant_n + dup) / (1.0 + 2.0/values[nconv-1] + rup) );
    //std::cout << "INFO: COMPUTING METRIC SUCCESSFULLY! \n";
    return (g / 2.0);
  }
  
  //TODO: supposing unweighted here!
  void addEdge(NetworKit::node u, NetworKit::node v) {
    if (u == v) {
      std::cout << "Warning: Graph has edge with equal target and destination!";
      return;
    }
    
    PetscInt a = (PetscInt) u;
    PetscInt b = (PetscInt) v; 
    PetscScalar w = 1.0;
    PetscScalar nw = -1.0;
		 
    MatSetValues(A, 1, &a, 1, &a, &w, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &b, &w, ADD_VALUES);
    MatSetValues(A, 1, &a, 1, &b, &nw, ADD_VALUES);
    MatSetValues(A, 1, &b, 1, &a, &nw, ADD_VALUES);
    // ALWAYS ASSEMBLY AFTER MATSETVALUES().
    // TODO: MAT_FINAL_ASSEMBLY OR MAT_FLUSH_ASSEMBLY


    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    
    std::cout << "INFO: ADD EDGE FOR k SUCCESSFULLY! \n";
  }
    
  

private:
  // MatCreateSeqAIJMP(PETSC_COMM_WORLD, n, n, 0, nnz, &A);
  PetscErrorCode  MatCreateSeqAIJMP(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
  {
    PetscErrorCode ierr; 
    PetscFunctionBegin;
    ierr = MatCreate(comm, A); 
    ierr = MatSetSizes(*A, m, n, m, n);
    // TODO: TEMP. PLEASE REMOVE FOLLOWING LINE!!
    ierr = MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    ierr = MatSetFromOptions(*A);
    ierr = MatSetType(*A, MATSEQAIJ); // MATAIJ
    ierr = MatSeqAIJSetPreallocation(*A, nz, (PetscInt*)nnz);
    PetscFunctionReturn(0);
  }

  // FIRST APPROACH: INSERT ELEMENTS TO PETSc MATRIX (ONE BY ONE).
  // MatSetValues_Elm(g, &A);
  void  MatSetValues_Elm(NetworKit::Graph const & g, Mat *A) {
    g.forEdges([&](NetworKit::node u, NetworKit::node v, double w) {
		 if (u == v) {
		   std::cout << "Warning: Graph has edge with equal target and destination!";
		 }        	     
		 PetscInt a = (PetscInt) u;
		 PetscInt b = (PetscInt) v; 
		 PetscScalar vv = (PetscScalar) w;
		 PetscScalar nv = (PetscScalar) -w;
		 
		 //std::cout<< " edge (" << a << ", " << b << ") w = " << w << "or " << vv << "\n";
		 
		 MatSetValues(*A, 1, &a, 1, &a, &vv, ADD_VALUES);
		 MatSetValues(*A, 1, &b, 1, &b, &vv, ADD_VALUES);
		 MatSetValues(*A, 1, &a, 1, &b, &nv, ADD_VALUES); // DONT MIX ADD AND INSERT_VALUES
		 MatSetValues(*A, 1, &b, 1, &a, &nv, ADD_VALUES); // DONT MIX ADD AND INSERT_VALUES
	       });
    

  }

  // SECOND APPROACH: INSERT ROW BY ROW.
  // MatSetValues_Row(g, nnz, &A);
  void  MatSetValues_Row(NetworKit::Graph const & g, PetscInt * nnz, Mat *A)
  {
    // g.balancedParallelForNodes([&](NetworKit::node v) {    
    g.forNodes([&](const NetworKit::node v){
  		 double weightedDegree = 0.0;
  		 PetscInt * col  = (PetscInt *) malloc(nnz[v] * sizeof(PetscInt));
  		 PetscScalar * val  = (PetscScalar *) malloc(nnz[v] * sizeof(PetscScalar));
  		 unsigned int idx = 0;
  		 g.forNeighborsOf(v, [&](const NetworKit::node u, double w) { // - adj  mat
  				       if (u != v) { // exclude diagonal since this would be subtracted by the adjacency weight
  					 weightedDegree += w;
  				       }
  				       col[idx] = (PetscInt)u;
  				       val[idx] = -(PetscScalar)w;
  				       idx++;
  				     });
  		 col[idx] = v;
  		 val[idx] = weightedDegree;
  		 PetscInt a = (PetscInt) v;
		 // std::cout<< " node " << a << " : [";
		 // for(int i = 0; i < nnz[v]; i++)
		 //   std::cout << col[i] << " (" << val[i] << ") ";
		 // std::cout << "] \n";
		 // std::cout << "idx =  " << idx << "\n";
		 // std::cout << "nnz[v] =  " << nnz[v] << "\n";
		 
		 MatSetValues(*A, 1, &a, nnz[v] , col, val, INSERT_VALUES);
  	       });	
  }
  
  


  
  EPS            eps;             /* eigenproblem solver context (smallest eigenvalues)*/
  EPS            eps_l;           /* eigenproblem solver context (largest eigenvalue) */
  Mat            A;               /* operator matrix */
  PetscInt       n, Istart, Iend;
  Vec            x;               /* vector representing the nullspace */
  EPSType        type;            /* diagnostic: type of solver */
  PetscReal      error, tol;      /* diagnostic: error and tolerance of the solver */
  PetscInt       maxit, its;      /* diagnostic: max iterations, actual iterations */
  PetscErrorCode ierr;            /* diagnostic: petscerror */
  PetscInt       c, nev, nconv;   /* diagnostic: # of eigenvalues, # of computed eigenvalues, # of converged eigenvalues */
  PetscInt       nconv_l=0;         /* diagnostic: # of largest eigenvale (1 if converged)  */
  double *       e_vectors;       /* stores the eigenvectors (of size n*nconv) */
  double *       e_values;        /* stores eigenvalues (of size nconv + 1) */

  
};





#endif // SLEPC_ADAPTER_H
