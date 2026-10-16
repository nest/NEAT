: Double exponential conductance based synapse, equivalent to NEURON's
: built-in `Exp2Syn`.
: `i_rec` duplicates the (non-probeable) current `i` for recording, it is
: assigned after `i` so that modcc infers the conductance from `i`.

NEURON {
    POINT_PROCESS Exp2Syn
    NONSPECIFIC_CURRENT i
    RANGE tau1, tau2, e
}

PARAMETER {
    tau1 = 0.1 (ms)
    tau2 = 10 (ms)
    e = 0 (mV)
}

ASSIGNED {
    factor
    i_rec (nA)
}

STATE {
    A (uS)
    B (uS)
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
    i = (B - A) * (v - e)
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
