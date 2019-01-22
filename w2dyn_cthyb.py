import numpy as np
import os, sys

### add main directory of w2dyn installation
#home = os.path.expanduser("~")
#auxdir=home+"/w2dynamics_github"
#sys.path.insert(0,auxdir)

import auxiliaries.CTQMC

### here come the necessary imports form w2dyn dmft loop
import dmft.impurity as impurity
import auxiliaries.config as config

class Solver():
    
    def __init__(self, beta, gf_struct, n_iw=1025, n_tau=10001, n_l=30):

        print "init!"

        self.beta = beta
        self.gf_struct= gf_struct
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.n_l = n_l

    def solve(self, **params_kw):

        depr_params = dict(
            measure_g_tau='measure_G_tau',
            measure_g_l='measure_G_l',
            )

        for key in depr_params.keys():
            if key in params_kw.keys():
                print 'WARNING: cthyb.solve parameter %s is deprecated use %s.' % \
                    (key, depr_params[key])
                val = params_kw.pop(key)
                params_kw[depr_params[key]] = val

        #n_cycles = params_kw.pop("n_cycles", True)
        print "params_kw", params_kw
        self.n_cycles = params_kw.pop("n_cycles")  ### what does the True or False mean?
        self.measure_G_l = params_kw.pop("measure_G_l")
        self.n_warmup_cycles = params_kw.pop("n_warmup_cycles")
        self.length_cycle = params_kw.pop("length_cycle")
        self.h_int = params_kw.pop("h_int")
        
        print "self.h_int", self.h_int

        #### load stuff from pyed
        from pyed.OperatorUtils import fundamental_operators_from_gf_struct
        from pyed.OperatorUtils import quadratic_matrix_from_operator
        from pyed.OperatorUtils import quartic_tensor_from_operator

        if isinstance(self.gf_struct,dict):
            print "WARNING: gf_struct should be a list of pairs [ [str,[int,...]], ...], not a dict"
            self.gf_struct = [ [k, v] for k, v in self.gf_struct.iteritems() ]


        ### I now also generate the fundamental operators out of gf_struct and save them
        from pyed.OperatorUtils import fundamental_operators_from_gf_struct
        fundamental_operators = fundamental_operators_from_gf_struct(self.gf_struct)
        print "fundamental_operators ", fundamental_operators 

        ### extract t_ij and U_ijkl from gf_struct
        print "extract t_ij and U_ijkl from gf_struct... "
        t_OO = quadratic_matrix_from_operator(self.h_int, fundamental_operators)
        U_OOOO = quartic_tensor_from_operator(self.h_int, fundamental_operators, perm_sym=False)
        print "done!"

        print "t_OO", t_OO
        print "U_0000", U_OOOO

        ### transform t_ij from (f,f) to (o,s,o,s) format
        from example import NO_to_Nos
        t_osos = NO_to_Nos(t_OO, spin_first=True)

        ### now comes w2dyn!
