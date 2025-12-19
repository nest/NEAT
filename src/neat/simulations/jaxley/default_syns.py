from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jaxley.channels import Channel
from jaxley.synapses.synapse import Synapse
from jaxley.solver_gate import (
    exponential_euler,
    save_exp,
    solve_gate_exponential,
    solve_inf_gate_exponential,
)



class NEATJaxleySynapse(Channel):
    """
    Compute syanptic current and update synapse state.
    """
    def __init__(self, name = None):
        self.current_is_in_mA_per_cm2=True
        super().__init__(name)
        # self.channel_params = {}
        # self.channel_states = {}

    @property
    def prefix(self):
        return f'{self._name}_'

    def set_spiketrain(self, spktms, dt=0.1, t_calibrate=0.0, t_max=100., weight=1.0):
        maxidx = jnp.round(t_max / dt).astype(int)
        calidx = jnp.round(t_calibrate / dt).astype(int)

        spkidx = jnp.round(jnp.array(spktms) / dt).astype(int)
        spkidx = spkidx[spkidx <= maxidx + 1]
        spkidx += calidx

        # self.cc = 0 # step index counter
        self.spkidx = spkidx[1:]
        print("Setting spiketrain:", self.spkidx)
        self.channel_params["weight"] = weight
        self.channel_states.update({
            f"{self.prefix}cc": 0,
            f"{self.prefix}next_spk": spkidx[0],
        })


class DoubleExpSynapse(NEATJaxleySynapse):
    def __init__(self, name = None, tau_r=.2, tau_d=3., e_r=0.):
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}tau_r": tau_r, 
            f"{self.prefix}tau_d": tau_d, 
            f"{self.prefix}e_r": e_r,
            f"{self.prefix}weight": 0.0, # meant to be reset
        }
        self.channel_states = {
            f"{self.prefix}x_r": 0.0,
            f"{self.prefix}x_d": 0.0,
            f"{self.prefix}cc": 0,
            f"{self.prefix}next_spk": 0,
        }

    def compute_propagators(self, delta_t, params):
        tau_r, tau_d = params[f'{self.prefix}tau_r'], params[f'{self.prefix}tau_d']
        tp = (tau_r * tau_d) / (tau_d - tau_r) * jnp.log( tau_d / tau_r )
        g_norm =  1. / ( -jnp.exp( -tp / tau_r ) + jnp.exp( -tp / tau_d ) )

        p_r = jnp.exp(-delta_t / tau_r)
        p_d = jnp.exp(-delta_t / tau_d)
        return p_r, p_d, g_norm

    def update_states(self, u, dt, voltages, params):
        """Return updated synapse state and current."""
        w = params[f'{self.prefix}weight']
        x_r = u[f'{self.prefix}x_r']
        x_d = u[f'{self.prefix}x_d']
        # spkidx = u[f'{self.prefix}spkidx']
        cc = u[f'{self.prefix}cc']
        next_spk = u[f'{self.prefix}next_spk']

        p_r, p_d, g_norm = self.compute_propagators(dt, params)

        def branch1():
            print("!! in spike branch !!")
            n_x_r = x_r - g_norm * w
            n_x_d = x_d + g_norm * w
            # self.spkidx.delete(0)
            n_cc = cc + 1
            n_spk = next_spk + 1
            # n_spkidx = self.spkidx[n_cc:]
            return n_x_r, n_x_d, n_cc, n_spk
        
        def branch2():
            n_cc = cc + 1
            return x_r, x_d, n_cc, next_spk

        # spike delivery
        new_x_r, new_x_d, new_cc, new_spk = jax.lax.cond(
            (cc == next_spk)[0],
            # True,
            branch1,
            branch2,
        )
        new_x_r *= p_r
        new_x_d *= p_d

        return {
            f'{self.prefix}x_r': new_x_r, 
            f'{self.prefix}x_d': new_x_d,
            # f'{self.prefix}spkidx': new_spkidx,
            f'{self.prefix}cc': new_cc,
            f'{self.prefix}next_spk': new_spk,
        }

    def compute_current(self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]):
        print(self.prefix)#, params)
        e_r = params[f'{self.prefix}e_r']
        v = voltages
        x_r = u[f'{self.prefix}x_r']
        x_d = u[f'{self.prefix}x_d']
        return (x_d - x_r) * (v - e_r)
    

class AMPASynapse(DoubleExpSynapse):
    def __init__(self, name = None, tau_r_AMPA=.2, tau_d_AMPA=3., e_r_AMPA=0.):
        super().__init__(name=name, tau_r=tau_r_AMPA, tau_d=tau_d_AMPA, e_r=e_r_AMPA)
        self.current_name = "i_AMPA"

class GABASynapse(DoubleExpSynapse):
    def __init__(self, name = None, tau_r_GABA=.2, tau_d_GABA=10., e_r_GABA=-80.):
        super().__init__(name=name, tau_r=tau_r_GABA, tau_d=tau_d_GABA, e_r=e_r_GABA)
        self.current_name = "i_GABA"

class AMPA_NMDASynapse(NEATJaxleySynapse):
    def __init__(self, name=None,
            tau_r_AMPA=.2, tau_AMPA=3., e_r_AMPA=0.,     
            tau_r_NMDA=.2, tau_d_NMDA=10., e_r_NMDA=-80.,
            NMDA_ratio=2.0,      
        ):
        self.current_is_in_mA_per_cm2=True
        self.channel_params = {
            "tau_r_AMPA": tau_r_AMPA, 
            "tau_d_AMPA": tau_d_AMPA, 
            "e_r_AMPA": e_r_AMPA,
            "tau_r_NMDA": tau_r_NMDA, 
            "tau_d_NMDA": tau_d_NMDA, 
            "e_r_NMDA": e_r_NMDA,
            "NMDA_ratio": NMDA_ratio,
            "weight": weight, # meant to be reset
        }
        self.channel_states = {
            "x_r_AMPA": 0.0,
            "x_d_NMDA": 0.0
        }