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


class Ca_HVA(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gCa_HVA": 1e-4,
            f"{self.prefix}eCa_HVA": 50.0,
        }
        self.channel_states = {f"{self.prefix}h": 0.0, f"{self.prefix}m": 0.0}
        self.current_name = "i_ca"

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
            0.000457
            * (save_exp(-1 / 28 * v - 15 / 28) + 1)
            * save_exp((1 / 28) * v + 15 / 28)
            / (
                (0.0065 * save_exp((1 / 50) * v + 13 / 50) + 0.000457)
                * save_exp((1 / 28) * v + 15 / 28)
                + 0.000457
            ),
            1.0
            / (
                0.000457 * save_exp(-1 / 50 * v - 13 / 50)
                + 0.0065 / (save_exp(-1 / 28 * v - 15 / 28) + 1)
            ),
        )

        # advance state variable m
        m_new = solve_inf_gate_exponential(
            m,
            dt,
            (0.055 * v + 1.485)
            * save_exp(0.321981424148607 * v)
            / (
                0.055 * (v + 27) * save_exp(0.321981424148607 * v)
                + 0.0114057221149848 * save_exp(0.263157894736842 * v)
                - 9.36151644263754e-6
            ),
            (1.0 * save_exp(0.263157894736842 * v) - 0.000820773673798209)
            * save_exp((1 / 17) * v)
            / (
                0.055 * (v + 27) * save_exp(0.321981424148607 * v)
                + 0.0114057221149848 * save_exp(0.263157894736842 * v)
                - 9.36151644263754e-6
            ),
        )

        return {f"{self.prefix}h": h_new, f"{self.prefix}m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gCa_HVA"]
        erev = params[f"{self.prefix}eCa_HVA"]
        v = voltages
        h = u[f"{self.prefix}h"]
        m = u[f"{self.prefix}m"]
        return gbar * (h * m**2) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        h = (
            0.000457
            * (save_exp(-1 / 28 * v - 15 / 28) + 1)
            * save_exp((1 / 28) * v + 15 / 28)
            / (
                (0.0065 * save_exp((1 / 50) * v + 13 / 50) + 0.000457)
                * save_exp((1 / 28) * v + 15 / 28)
                + 0.000457
            )
        )
        m = (
            (0.055 * v + 1.485)
            * save_exp(0.321981424148607 * v)
            / (
                0.055 * (v + 27) * save_exp(0.321981424148607 * v)
                + 0.0114057221149848 * save_exp(0.263157894736842 * v)
                - 9.36151644263754e-6
            )
        )
        return {f"{self.prefix}h": h, f"{self.prefix}m": m}


class Ca_LVAst(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gCa_LVAst": 1e-4,
            f"{self.prefix}eCa_LVAst": 50.0,
        }
        self.channel_states = {f"{self.prefix}h": 0.0, f"{self.prefix}m": 0.0}
        self.current_name = "i_ca"

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
            1.0 / (1280165.59676428 * save_exp(0.15625 * v) + 1),
            (8568.15374958056 * save_exp((1 / 7) * v) + 23.7056491911662)
            / (1265.03762380433 * save_exp((1 / 7) * v) + 1),
        )

        # advance state variable m
        m_new = solve_inf_gate_exponential(
            m,
            dt,
            1.0 / (1 + 0.00127263380133981 * save_exp(-1 / 6 * v)),
            (1856.88578179326 * save_exp((1 / 5) * v) + 8.46630328255936)
            / (1096.63315842846 * save_exp((1 / 5) * v) + 1),
        )

        return {f"{self.prefix}h": h_new, f"{self.prefix}m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gCa_LVAst"]
        erev = params[f"{self.prefix}eCa_LVAst"]
        v = voltages
        h = u[f"{self.prefix}h"]
        m = u[f"{self.prefix}m"]
        return gbar * (h * m**2) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        h = 1.0 / (1280165.59676428 * save_exp(0.15625 * v) + 1)
        m = 1.0 / (1 + 0.00127263380133981 * save_exp(-1 / 6 * v))
        return {f"{self.prefix}h": h, f"{self.prefix}m": m}


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


class NaTa_t(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gNaTa_t": 1e-4,
            f"{self.prefix}eNaTa_t": 50.0,
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
            jnp.where(
                jnp.logical_and(v >= -66.000006, v < -65.999994),
                -(0.0416666666666667 * v + 2.25),
                (1.0 * save_exp(0.166666666666667 * v) - 1.67017007902457e-05)
                / (
                    59874.1417151978 * save_exp(0.333333333333333 * v)
                    - 1.67017007902457e-05
                ),
            ),
            jnp.where(
                jnp.logical_and(v >= -66.000006, v < -65.999994),
                1.88140072945764,
                (
                    -45.1536175069833 * save_exp(0.166666666666667 * v)
                    + 1351767.04678348 * save_exp(0.333333333333333 * v)
                    + 0.000377071104599416
                )
                / (
                    59874.1417151978 * v * save_exp(0.333333333333333 * v)
                    - 1.67017007902457e-05 * v
                    + 3951693.35320306 * save_exp(0.333333333333333 * v)
                    - 0.00110231225215621
                ),
            ),
        )

        # advance state variable m
        m_new = solve_inf_gate_exponential(
            m,
            dt,
            jnp.where(
                jnp.logical_and(v > -38.000006, v < -37.999994),
                (0.091 * v + 4.55) / (0.029 * v + 2.938),
                0.182
                * (v + 38.0)
                * (563.030236835951 * save_exp(0.166666666666667 * v) - 1)
                * save_exp(0.166666666666667 * v)
                / (
                    (0.124 * v + 4.712)
                    * (save_exp(0.166666666666667 * v) - 0.00177610354573438)
                    + 0.182
                    * (v + 38.0)
                    * (563.030236835951 * save_exp(0.166666666666667 * v) - 1)
                    * save_exp(0.166666666666667 * v)
                ),
            ),
            jnp.where(
                jnp.logical_and(v > -38.000006, v < -37.999994),
                0.338652131302374 / (0.029 * v + 2.938),
                0.338652131302374
                * (save_exp(0.166666666666667 * v) - 0.00177610354573438)
                * (563.030236835951 * save_exp(0.166666666666667 * v) - 1)
                / (
                    (0.124 * v + 4.712)
                    * (save_exp(0.166666666666667 * v) - 0.00177610354573438)
                    + 0.182
                    * (v + 38.0)
                    * (563.030236835951 * save_exp(0.166666666666667 * v) - 1)
                    * save_exp(0.166666666666667 * v)
                ),
            ),
        )

        return {f"{self.prefix}h": h_new, f"{self.prefix}m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gNaTa_t"]
        erev = params[f"{self.prefix}eNaTa_t"]
        v = voltages
        h = u[f"{self.prefix}h"]
        m = u[f"{self.prefix}m"]
        return gbar * (h * m**3) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        h = jnp.where(
            jnp.logical_and(v >= -66.000006, v < -65.999994),
            -(0.0416666666666667 * v + 2.25),
            (1.0 * save_exp(0.166666666666667 * v) - 1.67017007902457e-05)
            / (
                59874.1417151978 * save_exp(0.333333333333333 * v)
                - 1.67017007902457e-05
            ),
        )
        m = jnp.where(
            jnp.logical_and(v > -38.000006, v < -37.999994),
            (0.091 * v + 4.55) / (0.029 * v + 2.938),
            0.182
            * (v + 38.0)
            * (563.030236835951 * save_exp(0.166666666666667 * v) - 1)
            * save_exp(0.166666666666667 * v)
            / (
                (0.124 * v + 4.712)
                * (save_exp(0.166666666666667 * v) - 0.00177610354573438)
                + 0.182
                * (v + 38.0)
                * (563.030236835951 * save_exp(0.166666666666667 * v) - 1)
                * save_exp(0.166666666666667 * v)
            ),
        )
        return {f"{self.prefix}h": h, f"{self.prefix}m": m}


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


class PiecewiseChannel(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gPiecewiseChannel": 1e-4,
            f"{self.prefix}ePiecewiseChannel": -28.0,
        }
        self.channel_states = {f"{self.prefix}a": 0.0, f"{self.prefix}b": 0.0}
        self.current_name = "i_"

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
        a = u[f"{self.prefix}a"]
        b = u[f"{self.prefix}b"]

        # advance state variable a
        a_new = solve_inf_gate_exponential(
            a,
            dt,
            jnp.where(
                v < -50.0,
                0.1,
                0.9,
            ),
            jnp.where(
                v < -50.0,
                10.0,
                20.0,
            ),
        )

        # advance state variable b
        b_new = solve_inf_gate_exponential(
            b,
            dt,
            jnp.where(
                v < -50.0,
                0.8,
                0.2,
            ),
            jnp.where(
                v < -50.0,
                0.1,
                50.0,
            ),
        )

        return {f"{self.prefix}a": a_new, f"{self.prefix}b": b_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gPiecewiseChannel"]
        erev = params[f"{self.prefix}ePiecewiseChannel"]
        v = voltages
        a = u[f"{self.prefix}a"]
        b = u[f"{self.prefix}b"]
        return gbar * (a + b) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        a = jnp.where(
            v < -50.0,
            0.1,
            0.9,
        )
        b = jnp.where(
            v < -50.0,
            0.8,
            0.2,
        )
        return {f"{self.prefix}a": a, f"{self.prefix}b": b}


class SK(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gSK": 1e-4,
            f"{self.prefix}eSK": -85.0,
        }
        self.channel_states = {f"{self.prefix}z": 0.0}
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
        z = u[f"{self.prefix}z"]

        # advance state variable z
        z_new = solve_inf_gate_exponential(
            z,
            dt,
            1.0 / (6.92864941342586e-17 * (1 / cai) ** 4.8 + 1.0),
            1.00000000000000,
        )

        return {f"{self.prefix}z": z_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gSK"]
        erev = params[f"{self.prefix}eSK"]
        v = voltages
        z = u[f"{self.prefix}z"]
        return gbar * (z) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        z = 1.0 / (6.92864941342586e-17 * (1 / ca) ** 4.8 + 1.0)
        return {f"{self.prefix}z": z}


class SK_E2(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gSK_E2": 1e-4,
            f"{self.prefix}eSK_E2": -85.0,
        }
        self.channel_states = {f"{self.prefix}z": 0.0}
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
        z = u[f"{self.prefix}z"]

        # advance state variable z
        z_new = solve_inf_gate_exponential(
            z,
            dt,
            jnp.where(
                cai > 1e-07,
                1.0 / (6.92864941342586e-17 * (1 / cai) ** 4.8 + 1.0),
                1.0 / (6.92864941342586e-17 * (1 / (cai + 1e-07)) ** 4.8 + 1.0),
            ),
            1.00000000000000,
        )

        return {f"{self.prefix}z": z_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gSK_E2"]
        erev = params[f"{self.prefix}eSK_E2"]
        v = voltages
        z = u[f"{self.prefix}z"]
        return gbar * (z) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        z = jnp.where(
            ca > 1e-07,
            1.0 / (6.92864941342586e-17 * (1 / ca) ** 4.8 + 1.0),
            1.0 / (6.92864941342586e-17 * (1 / (ca + 1e-07)) ** 4.8 + 1.0),
        )
        return {f"{self.prefix}z": z}


class SKv3_1(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gSKv3_1": 1e-4,
            f"{self.prefix}eSKv3_1": -85.0,
        }
        self.channel_states = {f"{self.prefix}z": 0.0}
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
        z = u[f"{self.prefix}z"]

        # advance state variable z
        z_new = solve_inf_gate_exponential(
            z,
            dt,
            save_exp(0.103092783505155 * v)
            / (save_exp(0.103092783505155 * v) + 6.874610940966),
            4.0
            * save_exp(0.0226551880380607 * v)
            / (save_exp(0.0226551880380607 * v) + 0.348253173014273),
        )

        return {f"{self.prefix}z": z_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gSKv3_1"]
        erev = params[f"{self.prefix}eSKv3_1"]
        v = voltages
        z = u[f"{self.prefix}z"]
        return gbar * (z) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        z = save_exp(0.103092783505155 * v) / (
            save_exp(0.103092783505155 * v) + 6.874610940966
        )
        return {f"{self.prefix}z": z}


class h(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gh": 1e-4,
            f"{self.prefix}eh": -43.0,
        }
        self.channel_states = {f"{self.prefix}hf": 0.0, f"{self.prefix}hs": 0.0}
        self.current_name = "i_"

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
        hf = u[f"{self.prefix}hf"]
        hs = u[f"{self.prefix}hs"]

        # advance state variable hf
        hf_new = solve_inf_gate_exponential(
            hf,
            dt,
            1.0 / (122306.530090586 * save_exp(0.142857142857143 * v) + 1.0),
            40.0000000000000,
        )

        # advance state variable hs
        hs_new = solve_inf_gate_exponential(
            hs,
            dt,
            1.0 / (122306.530090586 * save_exp(0.142857142857143 * v) + 1.0),
            300.000000000000,
        )

        return {f"{self.prefix}hf": hf_new, f"{self.prefix}hs": hs_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gh"]
        erev = params[f"{self.prefix}eh"]
        v = voltages
        hf = u[f"{self.prefix}hf"]
        hs = u[f"{self.prefix}hs"]
        return gbar * (0.8 * hf + 0.2 * hs) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        hf = 1.0 / (122306.530090586 * save_exp(0.142857142857143 * v) + 1.0)
        hs = 1.0 / (122306.530090586 * save_exp(0.142857142857143 * v) + 1.0)
        return {f"{self.prefix}hf": hf, f"{self.prefix}hs": hs}


class test_channel(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gtest_channel": 1e-4,
            f"{self.prefix}etest_channel": -23.0,
        }
        self.channel_states = {
            f"{self.prefix}a00": 0.0,
            f"{self.prefix}a01": 0.0,
            f"{self.prefix}a02": 0.0,
            f"{self.prefix}a10": 0.0,
            f"{self.prefix}a11": 0.0,
            f"{self.prefix}a12": 0.0,
        }
        self.current_name = "i_"

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
        a00 = u[f"{self.prefix}a00"]
        a01 = u[f"{self.prefix}a01"]
        a02 = u[f"{self.prefix}a02"]
        a10 = u[f"{self.prefix}a10"]
        a11 = u[f"{self.prefix}a11"]
        a12 = u[f"{self.prefix}a12"]

        # advance state variable a00
        a00_new = solve_inf_gate_exponential(
            a00,
            dt,
            1.0 / (0.740818220681718 * save_exp(0.01 * v) + 1.0),
            1.00000000000000,
        )

        # advance state variable a01
        a01_new = solve_inf_gate_exponential(
            a01,
            dt,
            1.0 * save_exp(0.01 * v) / (1.0 * save_exp(0.01 * v) + 1.349858807576),
            2.00000000000000,
        )

        # advance state variable a02
        a02_new = solve_inf_gate_exponential(
            a02,
            dt,
            -10.0000000000000,
            1.00000000000000,
        )

        # advance state variable a10
        a10_new = solve_inf_gate_exponential(
            a10,
            dt,
            2.0 / (0.740818220681718 * save_exp(0.01 * v) + 1.0),
            2.00000000000000,
        )

        # advance state variable a11
        a11_new = solve_inf_gate_exponential(
            a11,
            dt,
            2.0 * save_exp(0.01 * v) / (1.0 * save_exp(0.01 * v) + 1.349858807576),
            2.00000000000000,
        )

        # advance state variable a12
        a12_new = solve_inf_gate_exponential(
            a12,
            dt,
            -30.0000000000000,
            3.00000000000000,
        )

        return {
            f"{self.prefix}a00": a00_new,
            f"{self.prefix}a01": a01_new,
            f"{self.prefix}a02": a02_new,
            f"{self.prefix}a10": a10_new,
            f"{self.prefix}a11": a11_new,
            f"{self.prefix}a12": a12_new,
        }

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gtest_channel"]
        erev = params[f"{self.prefix}etest_channel"]
        v = voltages
        a00 = u[f"{self.prefix}a00"]
        a01 = u[f"{self.prefix}a01"]
        a02 = u[f"{self.prefix}a02"]
        a10 = u[f"{self.prefix}a10"]
        a11 = u[f"{self.prefix}a11"]
        a12 = u[f"{self.prefix}a12"]
        return gbar * (5 * a00**3 * a01**3 * a02 + a10**2 * a11**2 * a12) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        a00 = 1.0 / (0.740818220681718 * save_exp(0.01 * v) + 1.0)
        a01 = 1.0 * save_exp(0.01 * v) / (1.0 * save_exp(0.01 * v) + 1.349858807576)
        a02 = -10.0000000000000
        a10 = 2.0 / (0.740818220681718 * save_exp(0.01 * v) + 1.0)
        a11 = 2.0 * save_exp(0.01 * v) / (1.0 * save_exp(0.01 * v) + 1.349858807576)
        a12 = -30.0000000000000
        return {
            f"{self.prefix}a00": a00,
            f"{self.prefix}a01": a01,
            f"{self.prefix}a02": a02,
            f"{self.prefix}a10": a10,
            f"{self.prefix}a11": a11,
            f"{self.prefix}a12": a12,
        }


class test_channel2(Channel):
    # Automatically generated using neat

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.prefix}gtest_channel2": 1e-4,
            f"{self.prefix}etest_channel2": -23.0,
        }
        self.channel_states = {
            f"{self.prefix}a00": 0.0,
            f"{self.prefix}a01": 0.0,
            f"{self.prefix}a10": 0.0,
            f"{self.prefix}a11": 0.0,
        }
        self.current_name = "i_"

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
        a00 = u[f"{self.prefix}a00"]
        a01 = u[f"{self.prefix}a01"]
        a10 = u[f"{self.prefix}a10"]
        a11 = u[f"{self.prefix}a11"]

        # advance state variable a00
        a00_new = solve_inf_gate_exponential(
            a00,
            dt,
            0.300000000000000,
            1.00000000000000,
        )

        # advance state variable a01
        a01_new = solve_inf_gate_exponential(
            a01,
            dt,
            0.500000000000000,
            2.00000000000000,
        )

        # advance state variable a10
        a10_new = solve_inf_gate_exponential(
            a10,
            dt,
            0.400000000000000,
            2.00000000000000,
        )

        # advance state variable a11
        a11_new = solve_inf_gate_exponential(
            a11,
            dt,
            0.600000000000000,
            2.00000000000000,
        )

        return {
            f"{self.prefix}a00": a00_new,
            f"{self.prefix}a01": a01_new,
            f"{self.prefix}a10": a10_new,
            f"{self.prefix}a11": a11_new,
        }

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        gbar = params[f"{self.prefix}gtest_channel2"]
        erev = params[f"{self.prefix}etest_channel2"]
        v = voltages
        a00 = u[f"{self.prefix}a00"]
        a01 = u[f"{self.prefix}a01"]
        a10 = u[f"{self.prefix}a10"]
        a11 = u[f"{self.prefix}a11"]
        return gbar * (0.9 * a00**3 * a01**2 + 0.1 * a10**2 * a11) * (v - erev)

    def init_state(self, states, voltages, params, delta_t):
        v = voltages
        a00 = 0.300000000000000
        a01 = 0.500000000000000
        a10 = 0.400000000000000
        a11 = 0.600000000000000
        return {
            f"{self.prefix}a00": a00,
            f"{self.prefix}a01": a01,
            f"{self.prefix}a10": a10,
            f"{self.prefix}a11": a11,
        }
