from pytriqs.gf import *
from pytriqs.operators import *
from pytriqs.archive import HDFArchive
import pytriqs.utility.mpi as mpi

#### here comes the solver
#from triqs_cthyb import Solver
from w2dyn_cthyb import Solver

# Parameters
D, V, U = 1.0, 0.2, 4.0
e_f, beta = -U/2.0, 10

# Construct the impurity solver with the inverse temperature
# and the structure of the Green's functions
S = Solver(beta = beta, gf_struct = [ ['up',[0]], ['down',[0]] ], n_l = 30, n_iw=200,  n_tau=2002)

# Initialize the non-interacting Green's function S.G0_iw
#for name, g0 in S.G0_iw: g0 << inverse(iOmega_n - e_f - V**2 * Wilson(D))

# Run the solver. The results will be in S.G_tau, S.G_iw and S.G_l
S.solve(h_int = U * n('up',0) * n('down',0),     # Local Hamiltonian
        n_cycles  = 10000,                      # Number of QMC cycles
        length_cycle = 100,                      # Length of one cycle
        n_warmup_cycles = 10000,                 # Warmup cycles
        measure_G_l = False)                      # Measure G_l
