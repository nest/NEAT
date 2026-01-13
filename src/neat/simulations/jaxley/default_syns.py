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



class NEATJaxleySynapse(Synapse):
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


class DoubleExpSynapse(NEATJaxleySynapse):
    def __init__(self, name = None, tau_r=.2, tau_d=3., e_r=0.):
        super().__init__(name)
        self.synapse_params = {
            f"{self.prefix}tau_r": tau_r, 
            f"{self.prefix}tau_d": tau_d, 
            f"{self.prefix}e_r": e_r,
            f"{self.prefix}weight": 0.0, # meant to be reset
        }
        self.synapse_states = {
            f"{self.prefix}x_r": 0.0,
            f"{self.prefix}x_d": 0.0,
        }

    def compute_propagators(self, delta_t, params):
        tau_r, tau_d = params[f'{self.prefix}tau_r'], params[f'{self.prefix}tau_d']
        tp = (tau_r * tau_d) / (tau_d - tau_r) * jnp.log( tau_d / tau_r )
        g_norm =  1. / ( -jnp.exp( -tp / tau_r ) + jnp.exp( -tp / tau_d ) )

        p_r = jnp.exp(-delta_t / tau_r)
        p_d = jnp.exp(-delta_t / tau_d)
        return p_r, p_d, g_norm

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        w = params[f'{self.prefix}weight']
        x_r = states[f'{self.prefix}x_r']
        x_d = states[f'{self.prefix}x_d']

        tau_r, tau_d = params[f'{self.prefix}tau_r'], params[f'{self.prefix}tau_d']

        p_r, p_d, g_norm = self.compute_propagators(delta_t, params)

        # add the spikes to the conductance window, weight is contained in pre_voltage
        pred = pre_voltage >= 1e-15
        new_x_r = jnp.where(
            pred,
            x_r - g_norm * pre_voltage,
            x_r
        )
        new_x_d = jnp.where(
            pred,
            x_d + g_norm * pre_voltage,
            x_d
        )
        # decay the conductance window
        new_x_r *= p_r
        new_x_d *= p_d

        return {
            f'{self.prefix}x_r': new_x_r, 
            f'{self.prefix}x_d': new_x_d,
        }

    def compute_current(self, states, pre_voltage, post_voltage, params):
        e_r = params[f'{self.prefix}e_r']
        v = post_voltage
        x_r = states[f'{self.prefix}x_r']
        x_d = states[f'{self.prefix}x_d']
        return (x_d + x_r) * (v - e_r)
    

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