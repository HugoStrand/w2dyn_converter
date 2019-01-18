
import itertools
import numpy as np

from pytriqs.operators import c, c_dag, Operator, dagger

from pyed.OperatorUtils import fundamental_operators_from_gf_struct
from pyed.OperatorUtils import quadratic_matrix_from_operator
from pyed.OperatorUtils import quartic_tensor_from_operator

# ----------------------------------------------------------------------    
def NO_to_Nos(A_NO, spin_first=True):

    """ Reshape a rank N tensor with composite spin and orbital
    index to a rank 2N tensor with orbital and then spin index.

    Default is that the composite index has the spin index as the first 
    (slow) index.

    If spin index is the fastest index set: spin_first=False 

    Author: Hugo U. R. Strand (2019)"""
    
    shape = A_NO.shape
    N = len(shape)
    norb = shape[-1] / 2

    np.testing.assert_array_almost_equal(
        np.array(shape) - shape[-1], np.zeros(N))
    
    if spin_first:
        shape_Nso = [2, norb] * N
        A_Nso = A_NO.reshape(shape_Nso)

        # Enumerate axes with every pair of indices permuted
        # i.e. N = 2 givex axes = [1, 0, 3, 2]
        axes = np.arange(2 * N).reshape((N, 2))[:, ::-1].flatten()
        
        A_Nos = np.transpose(A_Nso, axes=axes)
    else:
        shape_Nos = [norb, 2] * N
        A_Nos = A_NO.reshape(shape_Nos)

    return A_Nos

# ----------------------------------------------------------------------    
def triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau(G_tau):

    """ Convert a spin-block Triqs imaginary time response function 
    (BlockGF) to  W2Dynamics ndarray format with indices [tosos]
    where t is time, o is orbital, s is spin index. 

    Returns: 
    g_tosos : ndarray with the response function data
    beta : inverse temperature
    ntau : number of tau points (including tau=0 and beta) 

    Author: Hugo U. R. Strand (2019) """
    
    beta = G_tau.mesh.beta
    tau = np.array([ float(t) for t in G_tau.mesh ])
    ntau = len(tau)
    np.testing.assert_almost_equal(tau[-1], beta)
    
    g_stoo = np.array([ g_tau.data for block_name, g_tau in G_tau ])
    ns, nt, no, nop = g_stoo.shape

    assert( no == nop )
    assert( ns == 2 )
    
    g_tosos = np.zeros((nt, no, ns, no, ns), dtype=g_tau.data.dtype)

    for s in xrange(ns):
        g_tosos[:, :, s, :, s] = g_stoo[s]

    return g_tosos, beta, ntau
    
# ----------------------------------------------------------------------    
def get_test_impurity_model(norb=2, ntau=1000, beta=10.0):

    """ Function that generates a random impurity model for testing """

    from pyed.OperatorUtils import symmetrize_quartic_tensor
    from pyed.OperatorUtils import get_quadratic_operator
    from pyed.OperatorUtils import operator_from_quartic_tensor
    
    orb_idxs = list(np.arange(norb))
    spin_idxs = ['up', 'do']
    gf_struct = [ [spin_idx, orb_idxs] for spin_idx in spin_idxs ]

    # -- Random Hamiltonian
    
    fundamental_operators = fundamental_operators_from_gf_struct(gf_struct)

    N = len(fundamental_operators)
    t_OO = np.random.random((N, N)) + 1.j * np.random.random((N, N))
    t_OO = 0.5 * ( t_OO + np.conj(t_OO.T) )

    #print 't_OO.real =\n', t_OO.real
    #print 't_OO.imag =\n', t_OO.imag
    
    U_OOOO = np.random.random((N, N, N, N)) + 1.j * np.random.random((N, N, N, N))
    U_OOOO = symmetrize_quartic_tensor(U_OOOO, conjugation=True)    
    
    print 'gf_struct =', gf_struct
    #print 'fundamental_operators = ', fundamental_operators

    H_loc = get_quadratic_operator(t_OO, fundamental_operators) + \
        operator_from_quartic_tensor(U_OOOO, fundamental_operators)

    #print 'H_loc =', H_loc

    from pytriqs.gf import MeshImTime, BlockGf

    mesh = MeshImTime(beta, 'Fermion', ntau)
    Delta_tau = BlockGf(mesh=mesh, gf_struct=gf_struct)

    for block_name, delta_tau in Delta_tau:
        delta_tau.data[:] = -0.5

    return gf_struct, Delta_tau, H_loc

# ----------------------------------------------------------------------    
if __name__ == '__main__':

    gf_struct, Delta_tau, H_loc = get_test_impurity_model(norb=3, ntau=1000, beta=10.0)

    # -- Convert the impurity model to ndarrays with W2Dynamics format
    # -- O : composite spin and orbital index
    # -- s, o : pure spin and orbital indices
    
    fundamental_operators = fundamental_operators_from_gf_struct(gf_struct)
    
    t_OO = quadratic_matrix_from_operator(H_loc, fundamental_operators)
    U_OOOO = quartic_tensor_from_operator(H_loc, fundamental_operators, perm_sym=True)

    # -- Reshape tensors by breaking out the spin index separately
    
    t_osos = NO_to_Nos(t_OO, spin_first=True)
    U_osososos = NO_to_Nos(U_OOOO, spin_first=True)

    # -- Extract hybridization as ndarray from Triqs response function (BlockGf)
    
    delta_tosos, beta, ntau = triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau(Delta_tau)

    print 'beta =', beta
    print 'ntau =', ntau
    print 't_osos.shape =', t_osos.shape
    print 'U_osososos.shape =', U_osososos.shape
    print 'delta_tosos.shape =', delta_tosos.shape
