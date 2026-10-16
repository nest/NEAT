: Current based synapse with double exponential profile, Arbor version of
: `neuron/mech_storage/epsc_double_exp.mod`.
: `i_rec` duplicates the (non-probeable) current `i` for recording, it is
: assigned after `i` so that modcc infers the conductance from `i`.

NEURON {
    POINT_PROCESS epsc_double_exp
    NONSPECIFIC_CURRENT i
    RANGE tau1, tau2
}

PARAMETER {
    tau1 = 0.2 (ms)
    tau2 = 3.0 (ms)
}

ASSIGNED {
    factor
    i_rec (nA)
}

STATE {
    A (nA)
    B (nA)
}

INITIAL {
    LOCAL tp
    A = 0
    B = 0
    tp = (tau1 * tau2) / (tau2 - tau1) * log(tau2 / tau1)
    factor = 1 / (-exp(-tp / tau1) + exp(-tp / tau2))
    i_rec = 0
}

BREAKPOINT {
    SOLVE dyn METHOD cnexp
    i = A - B
    i_rec = i
}

DERIVATIVE dyn {
    A' = -A / tau1
    B' = -B / tau2
}

NET_RECEIVE(weight) {
    A = A + weight * factor
    B = B + weight * factor
}
