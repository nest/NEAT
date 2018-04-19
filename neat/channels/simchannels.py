'''
This file is automatically generated by :func:`write_all_pychannels`
'''
import numpy as np

from ionchannels import SimChannel


class TestChannelSim(SimChannel):
    def __init__(self, inloc_inds, Ninloc, es_eq, g_max, e_rev, flag=flag, mode=mode):
        self.powers = np.array([[3, 3, 1, ], [2, 2, 1, ], ])
        self.factors = np.array([5.0, 1.0, ])
        super(TestChannelSim, self).__init__(self, inloc_inds, Ninloc, es_eq, g_max, e_rev, flag=0, mode=1)

    def svinf(self, V):
        V = V[self.inloc_inds] if self.mode == 1 else V
        sv_inf = np.zeros((2, 3, self.Nelem))
        sv_inf[0,0,:] = 1.0/(np.exp(V - 10.0) + 1.0)
        sv_inf[0,1,:] = 1.0/(np.exp(-V + 10.0) + 1.0)
        sv_inf[0,2,:] = -10.0
        sv_inf[1,0,:] = 1.0/(np.exp(V - 30.0) + 1.0)
        sv_inf[1,1,:] = 1.0/(np.exp(-V + 30.0) + 1.0)
        sv_inf[1,2,:] = -30.0
        return sv_inf 

    def tauinf(self, V):
        V = V[self.inloc_inds] if self.mode == 1 else V
        sv_inf = np.zeros((2, 3, self.Nelem))
        tau_inf[0,0,:] = np.exp(V - 10)
        tau_inf[0,1,:] = np.exp(-V + 10.0)
        tau_inf[0,2,:] = 1.0
        tau_inf[1,0,:] = np.exp(V - 30)
        tau_inf[1,1,:] = np.exp(-V + 30.0)
        tau_inf[1,2,:] = 3.0
        return tau_inf 

