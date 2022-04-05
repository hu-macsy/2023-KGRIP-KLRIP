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
	
	k = offset;
	n = (PetscInt)g.numberOfNodes();
	NetworKit::count m = (PetscInt)g.numberOfEdges();
	DEBUG("GRAPH INPUT: (n = ", n, " m = ", m, ")\n");

	// TODO: ADJUST FOR ALLOCATING MORE SPACE BASED ON k!
	// INSTEAD OF DEGREE(V) + 1, ALLOCATE DEGREE(V) + 1 + k
	// TO AVOID ANOTHER MALLOC (k IS SMALL COMPARED TO AVG DEGREE).	

	PetscInt * nnz = (PetscInt *) malloc( n * sizeof( PetscInt ) );	
	g.forNodes([&](NetworKit::node v) {
		     assert(v < n);
		     nnz[v] = (PetscInt) g.degree(v) + 1; //+ offset;
		   });
	
	MatCreateSeqAIJ(PETSC_COMM_WORLD, n, n, 0, nnz, &A); // includes preallocation
	MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
	// TODO: TEMP. PLEASE REMOVE FOLLOWING LINE!! // INGORES NEW MALLOC ERROR!
	MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); 
	MatSetType(A, MATSEQAIJ); 
	// SBAIJ IS SLOWER //MatSetType(A, MATSEQSBAIJ);
	// Q: WHY DO I NEED THE FOLLOWING ?
	//MatSetFromOptions(A);
	// Q: WHY DO I NEED THE FOLLOWING ?
	MatSetUp(A);
	////MatCreateSeqAIJMP(PETSC_COMM_WORLD,n,n,0,nnz,&A);
	
	
	// SETTING MATRIX ELEMENTS
	MatSetValuesROW(g, nnz, &A);
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	DEBUG("MATRIX IS CREATED SUCCESSFULLY.");
	DEBUG("VIEW MATRIX:");
	//MatView(A,PETSC_VIEWER_STDOUT_WORLD);
	free(nnz);
    }

  
    ~SlepcAdapter() {
      
      free(e_vectors);
      free(e_values);
      EPSDestroy(&eps);
      EPSDestroy(&eps_l);
      MatDestroy(&A);
      VecDestroy(&x);
      VecDestroyVecs(nconv,&Q);
      VecDestroy(&top);

      SlepcFinalize();

    }


  PetscErrorCode update_eigensolver() {

    // RESET 'NEW' MATRIX (AFTER ADDED EDGE)
    ierr = EPSSetOperators(eps, A, NULL); CHKERRQ(ierr);

    // RESET DEFLATION SPACE
    ierr = EPSSetDeflationSpace(eps, 1, &x); CHKERRQ(ierr);
    // DO I NEED FOLLOWING LINE EXPLICITLY?
    //ierr = EPSReset(eps); CHKERRQ(ierr);

    // RESET 'NEW' MATRIX (AFTER ADDED EDGE)
    ierr = EPSSetOperators(eps_l, A, NULL); CHKERRQ(ierr);

    // RESET DEFLATION SPACE
    ierr = EPSSetDeflationSpace(eps_l, 1, &x); CHKERRQ(ierr);
    //ierr = EPSReset(eps_l); CHKERRQ(ierr);
    

    EPSSetInitialSpace(eps,nconv,Q);
    EPSSetInitialSpace(eps_l,1,&top);

    //DEBUG("VIEW MATRIX:");
    //MatView(A,PETSC_VIEWER_STDOUT_WORLD);
    
    run_eigensolver();
    info_eigensolver(); 
    set_eigenpairs();
	
    DEBUG("RERUN EIGENSOLVER SUCCESSFULLY.");
    return ierr;
    }

  

  // ROUTINE TO SET THE EIGENSOLVER PRIOR TO EXECUTION
  PetscErrorCode set_eigensolver(NetworKit::count numberOfEigenpairs) {
    if ( !numberOfEigenpairs ) {
      std::cout << "WARN: NO EIGENPAIRS ARE TO BE COMPUTED.\n";
      return 0;
    }
    

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

    //std::cout << "INFO: SET EIGENSOLVER SUCCESSFULLY! \n";
    return ierr;
  }
    /* ========================================================================================== */ 


  // ROUTINE TO RUN THE EIGENSOLVER
  PetscErrorCode run_eigensolver() {
    
    ierr = EPSSolve(eps); CHKERRQ(ierr);
    ierr = EPSSolve(eps_l); CHKERRQ(ierr);
    //std::cout << "INFO: RUN EIGENSOLVER SUCCESSFULLY! \n";
    return ierr;
  }

  // ROUTINE TO RUN DIAGNOSTICS ON THE EIGENSOLVER
  // TODO: NOT ONLY DIAGNOSTICS! SETTING NCONV IS IMPORTANT AND IS DONE HERE!!
  PetscErrorCode info_eigensolver() {

      ierr = EPSGetType(eps, &type); CHKERRQ(ierr);
      DEBUG(" SOLUTION METHOD: ", type); 
      EPSGetIterationNumber(eps, &its);
      DEBUG(" ITERATION COUNT: ", its);
      EPSGetTolerances(eps, &tol, &maxit);
      DEBUG(" STOP COND: tol= ",(double)tol , "maxit= ", maxit);
      ierr = EPSGetDimensions(eps, &nev, NULL, NULL); CHKERRQ(ierr);
      DEBUG(" REQEST EVALUES: ", c);
      DEBUG(" COMPUT EVALUES: ", nev);
      ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
      DEBUG(" CONVRG EVALUES: ", nconv);
      if (nconv > c) nconv = c;
      // ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);
      // // //printing
      // ierr = EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      // ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      // ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      // EPSView(eps,PETSC_VIEWER_STDOUT_WORLD);
      
     DEBUG("INFO: INFO EIGENSOLVER SUCCESSFULLY!");
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
    MatCreateVecs(A,&top,NULL); 
    VecDuplicateVecs(top,nconv,&Q);
    //EPSGetInvariantSubspace(eps,Q);
    //EPSGetInvariantSubspace(eps_l,&top);
    
    
    PetscInt i;
    for (i = 0 ; i < nconv; i++) {
      EPSGetEigenpair(eps, i, &val, NULL, vec, NULL);
      //Compute relative error associated to each eigenpair
      EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error);
      //PetscPrintf(PETSC_COMM_WORLD,"   %12f      %12g\n", (double)val, (double)error);
      //PetscPrintf(PETSC_COMM_WORLD,"\n");
      e_values[i] = (double) val;
      //PetscReal      norm;
      //ierr = VecNorm(vec, NORM_2, &norm);
      //DEBUG("Norm of evector %d : %g\n", i, norm);
      //VecView(vec,PETSC_VIEWER_STDOUT_WORLD);
      VecCopy(vec,Q[i]);
      //std::cout << " e_vector "<< i << " : [ ";
      for(PetscInt j = 0; j < n; j++) {
	PetscScalar w;
	//VecGetValues(Vec x,PetscInt ni,const PetscInt ix[],PetscScalar y[])
	VecGetValues(vec, 1, &j, &w);
	*(e_vectors + i + j*c ) = (double) w;
	//std::cout << *(e_vectors + i*c + j ) << " ";
      }
      //std::cout << "]\n";
    }



    EPSType type_l;
    ierr = EPSGetType(eps_l, &type_l); 
    //ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type_l); 
    EPSGetConverged(eps_l,&nconv_l);
    //PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs FOR LARGE EIGENVALUE: %D\n\n",nconv_l);
    if ( !nconv_l ) {
      std::cout << "WARN: LARGEST EIGENVALUE IS NOT COMPUTED.\n";
    }
    assert(nconv_l >= 1);
    DEBUG("           k          ||Ax-kx||/||kx||\n"
	  "   ----------------- ------------------\n");
    
    //EPSGetEigenvalue(eps_l, 0, &val, NULL);
    EPSGetEigenpair(eps_l, 0, &val, NULL, vec, NULL);
    EPSComputeError(eps_l, 0, EPS_ERROR_RELATIVE, &error);
    DEBUG("   %12f       %12g\n",(double)val,(double)error, "\n");

    //std::cout << "INFO: i = " << i <<" \n";
    VecCopy(vec,top);

    e_values[i] = val; //EIGENVALUE_MULTIPLIER * e_values[i-1];
    //std::cout << "INFO: e_values[i+1] = " << e_values[i] <<" \n";
    VecDestroy(&vec);
    //std::cout << "INFO: RUN SETTING_EIGENPAIRS SUCCESSFULLY! \n";


    
  }


  double * get_eigenpairs() const {return e_vectors;}

  double * get_eigenvalues() const {return e_values;}
  



  // gain =  D_appx / (1.0 + R_approx).
  // D_appx = (upD + lowD)/2,
  // R_appx = (upR + lowR)/2.
  // DOES NOT WORK AS EXPECTED!
  double SpectralApproximationGainDifference2(NetworKit::node a, NetworKit::node b) {
    //double * vectors = get_eigenpairs();
    //double * values = get_eigenvalues();

    double upD = 0.0, upR = 0.0, lowD = 0.0, lowR = 0.0;

    //assert(e_values[nconv] > 0);
    double lambda_n = 1.0/(e_values[nconv] * e_values[nconv]);
    double lambda_c = 1.0/(e_values[nconv-1] * e_values[nconv-1]);    
    double sq_diff;

    for (int i = 0 ; i < nconv; i++) {
      //assert(e_values[i] > 0);
      sq_diff = *(e_vectors+a*c+i) - *(e_vectors+b*c+i);

      sq_diff *= sq_diff;

      upD += (1.0/(e_values[i] * e_values[i]) - lambda_c) * sq_diff;
      lowD += (1.0/(e_values[i] * e_values[i]) - lambda_n) * sq_diff;
      
      upR += (1.0/e_values[i] - 1.0/e_values[nconv-1]) * sq_diff;      
      lowR += (1.0/e_values[i] - 1.0/e_values[nconv]) * sq_diff;
    }

    upD += 2.0*lambda_c;
    lowD += 2.0*lambda_n;
    
    upR += 2.0/e_values[nconv-1];
    lowR +=  2.0/e_values[nconv];

    return (upD + lowD)/(2.0 + upR + lowR);
  }

  
  // gain = (U_bound + L_bound)/2.

  double SpectralApproximationGainDifference1(NetworKit::node a, NetworKit::node b) {
    //double * vectors = get_eigenpairs();
    //double * values = get_eigenvalues();

    double g = 0.0;
    double dlow = 0.0, dup = 0.0, rlow = 0.0, rup = 0.0;

    //assert(e_values[nconv] > 0);
    double constant_n = 1.0/(e_values[nconv] * e_values[nconv]);
    double constant_c = 1.0/(e_values[nconv-1] * e_values[nconv-1]);
    double sq_diff;
    
    for (int i = 0 ; i < nconv; i++) {
      //assert(e_values[i] > 0);
      sq_diff = *(e_vectors+a*c+i) - *(e_vectors+b*c+i);
      sq_diff *= sq_diff;
      dup += (1.0/(e_values[i] * e_values[i]) - constant_n) * sq_diff;
      rup += (1.0/e_values[i] - 1.0/e_values[nconv-1]) * sq_diff;
      
      dlow += (1.0/(e_values[i] * e_values[i]) - constant_c) * sq_diff;
      rlow += (1.0/e_values[i] - 1.0/e_values[nconv]) * sq_diff;
    }
    g = ( (constant_c + dlow)/ (1.0 + 1.0/e_values[nconv] + rlow)  ) +
        ( (constant_n + dup) / (1.0 + 1.0/e_values[nconv-1] + rup) );

    return (g / 2.0);
  }



  double SpectralToTalEffectiveResistance() {

    // double * values = get_eigenvalues();
    // std::cout << " nconv =  " << nconv << " c = " << c <<  "\n";
    // std::cout << " eigenvalues are:\n [ ";
    // for (int i = 0 ; i < c + 1; i++)
    //   std::cout << values[i] << " ";
    // std::cout << "]\n";
    // std::cout << " node 0: [ ";
    // for (int i = 0 ; i < n*c; i++) {
    //   std::cout << vectors[i] << " ";
    //   if (((i % c) == c-1 && i < (n*c-1) )) std::cout << "]\n node " << (i/c) + 1 << ":[ "; 
    // }
    // std::cout << "]\n";
    // std::cout << "=========================\n";
    // std::cout << " all together: [ ";
    // for(int j = 0; j < n*c; j++) {
    //   std::cout << *(vectors + j) << " ";
    // }
    // std::cout << "]\n";
    double Sum = 0.0;
   
    assert(e_values[nconv] > 0);
    for (int i = 0 ; i < nconv; i++) {
      assert(e_values[i] > 0);
      Sum += 1.0/e_values[i];
    }
    return n*Sum;
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

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    DEBUG("ADDING NEW EDGE: (", u, ", ", v, ") SUCCESSFULLY.");    

  }
    
  

private:
  // MatCreateSeqAIJMP(PETSC_COMM_WORLD, #rows, #cols, 0, nnz, &A);
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
    //ierr = MatSetType(*A, MATSEQSBAIJ); // MATSBAIJ
    ierr = MatSeqAIJSetPreallocation(*A, nz, (PetscInt*)nnz);
    PetscFunctionReturn(0);
  }

  // FIRST APPROACH: INSERT ELEMENTS TO PETSc MATRIX (ONE BY ONE).
  // MatSetValuesELM(g, &A);
  void  MatSetValuesELM(NetworKit::Graph const & g, Mat *A) {
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
  // MatSetValuesROW(g, nnz, &A);
  void  MatSetValuesROW(NetworKit::Graph const & g, PetscInt * nnz, Mat *A)
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
  Vec            *Q,top;
  NetworKit::count k;
  
};





#endif // SLEPC_ADAPTER_H
