: Single exponential AMPA+NMDA synapse, Arbor version of
: `neuron/mech_storage/exp_AMPA_NMDA.mod` (where the magnesium block is
: tabulated, here it is computed exactly).
: `i_rec` duplicates the (non-probeable) current `i` for recording, it is
: assigned after `i` so that modcc infers the conductance from `i`.

NEURON {
    POINT_PROCESS exp_AMPA_NMDA
    NONSPECIFIC_CURRENT i
    RANGE tau, tau_NMDA, mg, e, NMDA_ratio
}

PARAMETER {
    tau = 10 (ms)
    tau_NMDA = 200 (ms)
    e = 0 (mV)
    mg = 1 (mM)
    NMDA_ratio = 0.0
}

ASSIGNED {
    i_rec (nA)
}

STATE {
    g (uS)
    g_NMDA (uS)
}

INITIAL {
    g = 0
    g_NMDA = 0
    i_rec = 0
}

BREAKPOINT {
    SOLVE betadyn METHOD cnexp
    LOCAL x, B, g_slope, v0
    x = exp(0.062 * -v) * (mg / 3.57)
    B = 1 / (1 + x)
    : modcc takes the coefficient of `v` in the current as the conductance in
    : the implicit voltage update, treating local variables as constants. We
    : write the current so that this coefficient is the slope conductance
    : di/dv, which NEURON uses, and not the chord conductance g + g_NMDA * B.
    g_slope = g + g_NMDA * B + g_NMDA * (v - e) * 0.062 * x * B * B
    v0 = v
    i = g_slope * (v - v0) + (g + g_NMDA * B) * (v0 - e)
    i_rec = i
}

DERIVATIVE betadyn {
    g' = -g / tau
    g_NMDA' = -g_NMDA / tau_NMDA
}

NET_RECEIVE(weight) {
    g = g + weight
    g_NMDA = g_NMDA + weight * NMDA_ratio
}
