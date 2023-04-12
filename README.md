# Graph Robustness

This repository contains code related to the paper "Greedy Optimization of Resistance-based Graph Robustness with Global and Local Edge Insertions".

        
## Installation and Building


    git clone https://github.com/hu-macsy/2023-KGRIP-KLRIP.git
    cd 2023-KGRIP-KLRIP
        
### Requirements
        
Initialize submodules. 

    git submodule update --init --recursive

This will initialize some libraries:
* A [NetworKit](https://networkit.github.io/) fork with some additions which are not integrated into NetworKit yet. The fork is hosted at https://gitlab.informatik.hu-berlin.de/goergmat/networkit-robustness
* [Eigen](https://eigen.tuxfamily.org)

Additionally:
* The [PETSc](https://petsc.org) and [SLEPc](https://slepc.upv.es/) libraries are required (available on most packet managers). Install them and make sure that the environment variables `PETSC_DIR`, `SLEPC_DIR` and `PETSC_ARCH` are set accordingly. 
* MPI is required as well and should be the same version that PETSc is build for.
* If cmake does not find PETSc / SLEPc, set pkg config path: `export PKG_CONFIG_PATH=$SLEPC_DIR/$PETSC_ARCH/lib/pkgconfig:$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig/:$PKG_CONFIG_PATH`

### Building

    mkdir build
    cd build
    cmake .. [-DNETWORKIT_WITH_SANITIZERS=address]
    make

To run python scripts with the NetworKit fork, install the submodule (you should use a venv)

    cd ..
    pip install -e networkit


## Download Instances

    python3 load_instances.py


## Usage

### Basic Example

Run `colStoch` algorithm, k=20, ε=0.9, ε_UST = 10, using LAMG, 6 threads. 

    build/robustness -a6 -i ../instances/facebook_ego_combined -k 20 -eps 0.9 -eps2 10 --lamg -j 6


For more details see help string

    ./robustness --help


### With Simexpal

If necessary, change `instdir` in [experiments.yml](experiments.yml) and make sure that the requirements as listed above are met.

    simex b make
    simex e launch [...]
    
Read the [Simexpal docs](https://simexpal.readthedocs.io) for details.

To create figures for k-GRIP:

    python3 eval.py

Look at [eval_local_addition.ipynb](eval_local_addition.ipynb) for figures for k-LRIP.