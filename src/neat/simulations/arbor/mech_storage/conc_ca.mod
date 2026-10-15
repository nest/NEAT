: Exponential calcium concentration dynamics, Arbor version of
: `neuron/mech_storage/conc_ca.mod`.

NEURON {
    SUFFIX conc_ca
    USEION ca READ ica WRITE cai
    RANGE tau, gamma, inf
}

PARAMETER {
    gamma = 0.0259106803503
    tau = 80.0 (ms)
    inf = 0.0001 (mM)
}

STATE {
    cai (mM)
}

INITIAL {
    cai = inf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
}

DERIVATIVE states {
    cai' = - gamma * ica - (cai - inf) / tau
}
