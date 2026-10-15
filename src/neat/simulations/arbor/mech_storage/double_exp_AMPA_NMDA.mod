: Double exponential AMPA+NMDA synapse, Arbor version of
: `neuron/mech_storage/double_exp_AMPA_NMDA.mod` (where the magnesium block is
: tabulated, here it is computed exactly).
: `i_rec` duplicates the (non-probeable) current `i` for recording, it is
: assigned after `i` so that modcc infers the conductance from `i`.

NEURON {
    POINT_PROCESS double_exp_AMPA_NMDA
    NONSPECIFIC_CURRENT i
    RANGE tau1, tau2, tau1_NMDA, tau2_NMDA, mg, e, NMDA_ratio
}

PARAMETER {
    tau1 = 0.2 (ms)
    tau2 = 3. (ms)
    tau1_NMDA = 0.2 (ms)
    tau2_NMDA = 43 (ms)
    e = 0 (mV)
    mg = 1 (mM)
    NMDA_ratio = 0.0
}

ASSIGNED {
    factor_AMPA
    factor_NMDA
    i_rec (nA)
}

STATE {
    A_AMPA (uS)
    B_AMPA (uS)
    A_NMDA (uS)
    B_NMDA (uS)
}

INITIAL {
    LOCAL tp_AMPA, tp_NMDA
    A_AMPA = 0
    B_AMPA = 0
    A_NMDA = 0
    B_NMDA = 0
    tp_AMPA = (tau1 * tau2) / (tau2 - tau1) * log(tau2 / tau1)
    factor_AMPA = 1 / (-exp(-tp_AMPA / tau1) + exp(-tp_AMPA / tau2))
    tp_NMDA = (tau1_NMDA * tau2_NMDA) / (tau2_NMDA - tau1_NMDA) * log(tau2_NMDA / tau1_NMDA)
    factor_NMDA = 1 / (-exp(-tp_NMDA / tau1_NMDA) + exp(-tp_NMDA / tau2_NMDA))
    i_rec = 0
}

BREAKPOINT {
    SOLVE betadyn METHOD cnexp
    LOCAL x, f_NMDA, g_AMPA, g_NMDA, g_slope, v0
    x = 0.3 * exp(-0.1 * v)
    f_NMDA = 1 / (1 + x)
    g_AMPA = A_AMPA + B_AMPA
    g_NMDA = A_NMDA + B_NMDA
    : modcc takes the coefficient of `v` in the current as the conductance in
    : the implicit voltage update, treating local variables as constants. We
    : write the current so that this coefficient is the slope conductance
    : di/dv, which NEURON uses, and not the chord conductance.
    g_slope = g_AMPA + f_NMDA * g_NMDA + g_NMDA * (v - e) * 0.1 * x * f_NMDA * f_NMDA
    v0 = v
    i = g_slope * (v - v0) + (g_AMPA + f_NMDA * g_NMDA) * (v0 - e)
    i_rec = i
}

DERIVATIVE betadyn {
    A_AMPA' = -A_AMPA / tau1
    B_AMPA' = -B_AMPA / tau2
    A_NMDA' = -A_NMDA / tau1_NMDA
    B_NMDA' = -B_NMDA / tau2_NMDA
}

NET_RECEIVE(weight) {
    A_AMPA = A_AMPA - weight * factor_AMPA
    B_AMPA = B_AMPA + weight * factor_AMPA
    A_NMDA = A_NMDA - weight * NMDA_ratio * factor_NMDA
    B_NMDA = B_NMDA + weight * NMDA_ratio * factor_NMDA
}
