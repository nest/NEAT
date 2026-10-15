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

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        # self.channel_params = {}
        # self.channel_states = {}

    @property
    def prefix(self):
        return f"{self._name}_"


class DoubleExpSynapse(NEATJaxleySynapse):
    def __init__(self, name=None, tau_r=0.2, tau_d=3.0, e_r=0.0):
        super().__init__(name)
        self.synapse_params = {
            f"{self.prefix}tau_r": tau_r,
            f"{self.prefix}tau_d": tau_d,
            f"{self.prefix}e_r": e_r,
            f"{self.prefix}weight": 0.0,  # meant to be reset
        }
        self.synapse_states = {
            f"{self.prefix}x_r": 0.0,
            f"{self.prefix}x_d": 0.0,
        }

    def compute_propagators(self, delta_t, params):
        tau_r, tau_d = params[f"{self.prefix}tau_r"], params[f"{self.prefix}tau_d"]
        tp = (tau_r * tau_d) / (tau_d - tau_r) * jnp.log(tau_d / tau_r)
        g_norm = 1.0 / (-jnp.exp(-tp / tau_r) + jnp.exp(-tp / tau_d))

        p_r = jnp.exp(-delta_t / tau_r)
        p_d = jnp.exp(-delta_t / tau_d)
        return p_r, p_d, g_norm

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        w = params[f"{self.prefix}weight"]
        x_r = states[f"{self.prefix}x_r"]
        x_d = states[f"{self.prefix}x_d"]

        tau_r, tau_d = params[f"{self.prefix}tau_r"], params[f"{self.prefix}tau_d"]

        p_r, p_d, g_norm = self.compute_propagators(delta_t, params)

        # add the spikes to the conductance window, weight is contained in pre_voltage
        pred = pre_voltage >= 1e-15
        new_x_r = jnp.where(pred, x_r - g_norm * pre_voltage, x_r)
        new_x_d = jnp.where(pred, x_d + g_norm * pre_voltage, x_d)
        # decay the conductance window
        new_x_r *= p_r
        new_x_d *= p_d

        return {
            f"{self.prefix}x_r": new_x_r,
            f"{self.prefix}x_d": new_x_d,
        }

    def compute_current(self, states, pre_voltage, post_voltage, params):
        print(self.prefix)  # , params)
        e_r = params[f"{self.prefix}e_r"]
        v = post_voltage
        x_r = states[f"{self.prefix}x_r"]
        x_d = states[f"{self.prefix}x_d"]
        return (x_d + x_r) * (v - e_r)


class AMPASynapse(DoubleExpSynapse):
    def __init__(self, name=None, tau_r_AMPA=0.2, tau_d_AMPA=3.0, e_r_AMPA=0.0):
        super().__init__(name=name, tau_r=tau_r_AMPA, tau_d=tau_d_AMPA, e_r=e_r_AMPA)
        self.current_name = "i_AMPA"


class GABASynapse(DoubleExpSynapse):
    def __init__(self, name=None, tau_r_GABA=0.2, tau_d_GABA=10.0, e_r_GABA=-80.0):
        super().__init__(name=name, tau_r=tau_r_GABA, tau_d=tau_d_GABA, e_r=e_r_GABA)
        self.current_name = "i_GABA"


class AMPA_NMDASynapse(NEATJaxleySynapse):
    def __init__(
        self,
        name=None,
        tau_r_AMPA=0.2,
        tau_AMPA=3.0,
        e_r_AMPA=0.0,
        tau_r_NMDA=0.2,
        tau_d_NMDA=10.0,
        e_r_NMDA=-80.0,
        NMDA_ratio=2.0,
    ):
        self.current_is_in_mA_per_cm2 = True
        self.channel_params = {
            "tau_r_AMPA": tau_r_AMPA,
            "tau_d_AMPA": tau_d_AMPA,
            "e_r_AMPA": e_r_AMPA,
            "tau_r_NMDA": tau_r_NMDA,
            "tau_d_NMDA": tau_d_NMDA,
            "e_r_NMDA": e_r_NMDA,
            "NMDA_ratio": NMDA_ratio,
            "weight": weight,  # meant to be reset
        }
        self.channel_states = {"x_r_AMPA": 0.0, "x_d_NMDA": 0.0}


class Kv3_1(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gKv3_1": 1e-4,
            f"{self.prefix}eKv3_1": -85.0,
        }
        self.channel_states = {f"{self.prefix}m": 0.0}
        self.current_name = "i_k"

    @property
    def prefix(self):
        return f"{self._name}_"

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        v = voltages
        m = u[f"{self.prefix}m"]

        # advance state variable m
        m_new = solve_inf_gate_exponential(
            m,
            dt,
            1.0
            * save_exp(0.103092783505155 * v)
            / (1.0 * save_exp(0.103092783505155 * v) + 6.874610940966),
            4.0
            * save_exp(0.0226551880380607 * v)
            / (1.0 * save_exp(0.0226551880380607 * v) + 0.348253173014273),
        )

        return {f"{self.prefix}m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gKv3_1"]
        erev = params[f"{self.prefix}eKv3_1"]
        v = voltages
        m = u[f"{self.prefix}m"]
        return gbar * (m) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        m = (
            1.0
            * save_exp(0.103092783505155 * v)
            / (1.0 * save_exp(0.103092783505155 * v) + 6.874610940966)
        )
        return {f"{self.prefix}m": m}


class Na_Ta(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gNa_Ta": 1e-4,
            f"{self.prefix}eNa_Ta": 50.0,
        }
        self.channel_states = {f"{self.prefix}h": 0.0, f"{self.prefix}m": 0.0}
        self.current_name = "i_na"

    @property
    def prefix(self):
        return f"{self._name}_"

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        v = voltages
        h = u[f"{self.prefix}h"]
        m = u[f"{self.prefix}m"]

        # advance state variable h
        h_new = solve_inf_gate_exponential(
            h,
            dt,
            (
                0.015 * v * save_exp(0.166666666666667 * v)
                - 2.50525511853685e-7 * v
                + 0.99 * save_exp(0.166666666666667 * v)
                - 1.65346837823432e-5
            )
            / (
                898.112125727967 * v * save_exp(0.333333333333333 * v)
                - 2.50525511853685e-7 * v
                + 59275.4002980458 * save_exp(0.333333333333333 * v)
                - 1.65346837823432e-5
            ),
            (
                -0.677966101694915 * save_exp(0.166666666666667 * v)
                + 20296.3192254908 * save_exp(0.333333333333333 * v)
                + 5.66159348821887e-6
            )
            / (
                898.112125727967 * v * save_exp(0.333333333333333 * v)
                - 2.50525511853685e-7 * v
                + 59275.4002980458 * save_exp(0.333333333333333 * v)
                - 1.65346837823432e-5
            ),
        )

        # advance state variable m
        m_new = solve_inf_gate_exponential(
            m,
            dt,
            0.182
            * (v + 38.0)
            * (563.030236835951 * save_exp(0.166666666666667 * v) - 1.0)
            * save_exp(0.166666666666667 * v)
            / (
                -(0.00177610354573438 - 1.0 * save_exp(0.166666666666667 * v))
                * (0.124 * v + 4.712)
                + 0.182
                * (v + 38.0)
                * (563.030236835951 * save_exp(0.166666666666667 * v) - 1.0)
                * save_exp(0.166666666666667 * v)
            ),
            (
                0.677966101694915 * save_exp(0.166666666666667 * v)
                - 190.857707402017 * save_exp(0.333333333333333 * v)
                - 0.000602068998554027
            )
            / (
                0.058 * v * save_exp(0.166666666666667 * v)
                - 102.471503104143 * v * save_exp(0.333333333333333 * v)
                + 0.000220236839671063 * v
                + 2.204 * save_exp(0.166666666666667 * v)
                - 3893.91711795744 * save_exp(0.333333333333333 * v)
                + 0.00836899990750039
            ),
        )

        return {f"{self.prefix}h": h_new, f"{self.prefix}m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gNa_Ta"]
        erev = params[f"{self.prefix}eNa_Ta"]
        v = voltages
        h = u[f"{self.prefix}h"]
        m = u[f"{self.prefix}m"]
        return gbar * (h * m**3) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        h = (
            0.015 * v * save_exp(0.166666666666667 * v)
            - 2.50525511853685e-7 * v
            + 0.99 * save_exp(0.166666666666667 * v)
            - 1.65346837823432e-5
        ) / (
            898.112125727967 * v * save_exp(0.333333333333333 * v)
            - 2.50525511853685e-7 * v
            + 59275.4002980458 * save_exp(0.333333333333333 * v)
            - 1.65346837823432e-5
        )
        m = (
            0.182
            * (v + 38.0)
            * (563.030236835951 * save_exp(0.166666666666667 * v) - 1.0)
            * save_exp(0.166666666666667 * v)
            / (
                -(0.00177610354573438 - 1.0 * save_exp(0.166666666666667 * v))
                * (0.124 * v + 4.712)
                + 0.182
                * (v + 38.0)
                * (563.030236835951 * save_exp(0.166666666666667 * v) - 1.0)
                * save_exp(0.166666666666667 * v)
            )
        )
        return {f"{self.prefix}h": h, f"{self.prefix}m": m}
