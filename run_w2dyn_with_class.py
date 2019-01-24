from pytriqs.gf import *
from pytriqs.operators import *
from pytriqs.archive import HDFArchive
import pytriqs.utility.mpi as mpi

#w2dyn=False
w2dyn=True

#### here comes the solver
if w2dyn:
   from w2dyn_cthyb import Solver
else:
   from triqs_cthyb import Solver

# Parameters
D, V, U = 1.0, 0.2, 4.0
e_f, beta = -U/2.0, 10

# Construct the impurity solver with the inverse temperature
# and the structure of the Green's functions
S = Solver(beta = beta, gf_struct = [ ['up',[0]], ['down',[0]] ], n_l = 30, n_iw=200,  n_tau=402)

if w2dyn:
    ### generate a Delta(tau) for w2dyn
    iw_mesh = MeshImFreq(beta, 'Fermion', S.n_iw)
    Delta_iw = BlockGf(mesh=iw_mesh, gf_struct=S.gf_struct)

    for name, d in Delta_iw: 
        #G0_iw = inverse(iOmega_n - e_f - V**2 * Wilson(D))
        #d << inverse(iOmega_n - e_f - V**2 * Wilson(D))
        d << - V**2 * Wilson(D)

        #d << Fourier(Delta_iw[name])   # this is the same as below, but blockwise

    S.Delta_tau << Fourier(Delta_iw)

else:

   # Initialize the non-interacting Green's function S.G0_iw
   for name, g0 in S.G0_iw: g0 << inverse(iOmega_n - e_f - V**2 * Wilson(D))
    

# Run the solver. The results will be in S.G_tau, S.G_iw and S.G_l
S.solve(h_int = U * n('up',0) * n('down',0) + e_f* ( n('up',0) + n('down',0) ),     # Local Hamiltonian + quadratic terms
        n_cycles  = 50000,                      # Number of QMC cycles
        length_cycle = 100,                      # Length of one cycle
        n_warmup_cycles = 10000,                 # Warmup cycles
        measure_G_l = False)                      # Measure G_l

print "S.G_tau:",  S.G_tau
from pytriqs.plot.mpl_interface import oplot, oploti, oplotr, plt
#subp = [3, 1, 1]
#plt.subplot(*subp); subp[-1] += 1
#oplot(Delta_iw)
#plt.subplot(*subp); subp[-1] += 1
#oplot(S.Delta_tau)
#plt.subplot(*subp); subp[-1] += 1
oplot(S.G_tau)
plt.show()
