
import numpy as np

from pytriqs.operators import c, c_dag, Operator, dagger

from pyed.OperatorUtils import fundamental_operators_from_gf_struct
from pyed.OperatorUtils import quadratic_matrix_from_operator
from pyed.OperatorUtils import quartic_tensor_from_operator

# ----------------------------------------------------------------------    
def triqs_gf_to_w2dyn_ndarray(G_tau):

    g_btoo = np.array([ g_tau.data for block_name, g_tau in G_tau ])
    print g_btoo.shape

# ----------------------------------------------------------------------    
if __name__ == '__main__':

    orb_idxs = [0]
    spin_idxs = ['up', 'do']
    gf_struct = [ [spin_idx, orb_idxs] for spin_idx in spin_idxs ]
    
    fundamental_operators = fundamental_operators_from_gf_struct(gf_struct)

    print 'gf_struct =', gf_struct
    print 'fundamental_operators = ', fundamental_operators

    n0 = c_dag('up', 0) * c('up', 0) + c_dag('do', 0) * c('do', 0)    
    H = 1.0 * n0 * n0 - 0.5 * n0

    t_ab = quadratic_matrix_from_operator(H, fundamental_operators)
    U_abcd = quartic_tensor_from_operator(H, fundamental_operators)
    U_abcd_sym = quartic_tensor_from_operator(H, fundamental_operators, perm_sym=True)

    print 'H =', H
    print 't_ab =\n', t_ab
    print 'U_abcd =\n', U_abcd.real
    print 'U_abcd_sym =\n', U_abcd_sym.real

    # ------------------------------------------------------------------
    
    from pytriqs.gf import MeshImTime, BlockGf

    beta = 10.0
    ntau = 1000
    norb = len(orb_idxs)
    
    mesh = MeshImTime(beta, 'Fermion', ntau)
    G_tau = BlockGf(mesh=mesh, gf_struct=gf_struct)
    triqs_gf_to_w2dyn_ndarray(G_tau)

