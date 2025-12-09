from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
from jaxley.channels import Channel
from jaxley.synapses.synapse import Synapse
from jaxley.solver_gate import (
    exponential_euler,
    save_exp,
    solve_gate_exponential,
    solve_inf_gate_exponential,
)



class NEATJaxleySynapse(Synapse):
    """
    Compute syanptic current and update synapse state.
    """
    def __init__(self, name = None):
        super().__init__(name)
        self.channel_params = {}
        self.channel_states = {}


    def set_spiketrain(self, spktms, dt=0.1, t_calibrate=0.0, t_max=100., weight=1.0):
        maxidx = jnp.round(t_max / dt).astype(int)
        calidx = jnp.round(t_calibrate / dt).astype(int)

        spkidx = jnp.round(jnp.array(spktms) / dt).astype(int)
        spkidx = spkidx[spkidx < maxidx]
        spkidx += calidx

        self.cc = 0 # step index counter
        self.spkidx = spkidx
        self.channel_params["weight"] = weight


class DoubleExpSynapse(NEATJaxleySynapse):
    def __init__(self, name = None, tau_r=.2, tau_d=3., e_r=0.):
        super().__init__(name)
        self.channel_params = {
            "tau_r": tau_r, 
            "tau_d": tau_d, 
            "e_r": e_r,
            "weight": 0.0, # meant to be reset
        }
        self.channel_states = {
            "x_r": 0.0,
            "x_d": 0.0
        }

    def compute_propagators(self, delta_t, params):
        tau_r, tau_d = params['tau_r'], params['tau_d']
        tp = (tau_r * tau_d) / (tau_d - tau_r) * jnp.log( tau_d / tau_r )
        self.g_norm =  1. / ( -jnp.exp( -tp / tau_r ) + jnp.exp( -tp / tau_d ) )

        p_r = jnp.exp(-delta_t / params["tau_r"])
        p_d = jnp.exp(-delta_t / params["tau_d"])
        return p_r, p_d

    def update_states(self, u, voltages, params):
        """Return updated synapse state and current."""

        # spike delivery
        if self.cc == self.spktms[0]:
            new_x_r = states["x_r"] - self.g_norm * params["weight"]
            new_x_d = states["x_d"] + self.g_norm * params["weight"]
            self.spktms.delete(0)
        self.cc += 1

        p_r, p_d = self.compute_propagators(delta_t, params)
        new_x_r *= p_r
        new_x_d *= p_d

        return {"x_r": new_x_r, "x_d": new_x_d}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        return params["weight"] * (states["x_d"] - states['x_r']) * (post_voltage - params["e_r"])
    

class AMPASynapse(DoubleExpSynapse):
    def __init__(self, name = None, tau_r_AMPA=.2, tau_d_AMPA=3., e_r_AMPA=0.):
        super().__init__(tau_r=tau_r_AMPA, tau_d=tau_d_AMPA, e_r=e_r_AMPA)
        self.current_name = "AMPA"


class GABASynapse(DoubleExpSynapse):
    def __init__(self, name = None, tau_r_GABA=.2, tau_d_GABA=10., e_GABA=-80.):
        super().__init__(tau_r=tau_r_GABA, tau_d=tau_d_GABA, e_r=e_r_GABA)


class AMPA_NMDASynapse(NEATJaxleySynapse):
    def __init__(self, name=None,
            tau_r_AMPA=.2, tau_AMPA=3., e_r_AMPA=0.,     
            tau_r_NMDA=.2, tau_d_NMDA=10., e_r_NMDA=-80.,
            NMDA_ratio=2.0,      
        ):
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