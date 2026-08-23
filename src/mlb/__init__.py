"""mlb: the Model LB replication package, canonical API.

numba-only: the kernel is `kernel_numba`; there is NO pure-Python fallback,
because one could return a silently different calibration number on the MAIC
k<=0 s^2 path (see README, "Monte Carlo design"). `requirements.txt` pins numba.

Module map (paper symbols in each module's docstring):

    kernel_numba   the production numerical kernel (numba)
    kernel_select  the single documented import-time kernel binding
    paths          the single path resolver (inputs AND outputs)
    design         Z = [1, DU_1..DU_m]; break positions from lambdas
    detrend        GLS quasi-difference detrending at cbar
    statistics     the M class: MZa, MSB, MZt, PT, MPT
    lrv            long-run variance: difference-based ('const') and MAIC
    dgp            the Model LB DGP (known, imposed breaks; NO estimator)
    calibrate      tangency cbar*(m,T[,p]); oracle-PT convention
    resample       the package's three resampling procedures + robustness stage
    aggregate      the lambda-aggregation rule of the paper's s.2.2
    io             surface CSV schema and table rendering
"""
