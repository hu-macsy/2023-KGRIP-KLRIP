# Graph Robustness


        
## Installation and Building


    git clone https://gitlab.informatik.hu-berlin.de/macsy/code-staff/graph-robustness-k.git

    cd graph-robustness-k
        
### Requirements
        
Initialize submodules. Access to the separate NetworKit fork at https://gitlab.informatik.hu-berlin.de/goergmat/networkit-robustness is required.

    git submodule update --init --recursive

* Eigen (currently assumed to be in _eigen_ folder in root)    
* PETSc and SLEPc. Set environment variables PETSC_DIR, SLEPC_DIR and PETSC_ARCH accordingly

### Building

    mkdir build
    cd build
    cmake .. [-DNETWORKIT_WITH_SANITIZERS=address]
    make

To run python scripts with the NetworKit fork, install the submodule (you should use a venv)

    pip install -e networkit


## Download Instances

    cd ..
    python3 load_instances.py


## Usage

Example. Run main algorithm, k=20, ε=0.9, ε_UST = 10, using LAMG, 6 threads. 

    cd build
    ./robustness -a6 -i ../instances/facebook_ego_combined -k 20 -eps 0.9 -eps2 10 --lamg -j 6


For more details see help string

    ./robustness --help