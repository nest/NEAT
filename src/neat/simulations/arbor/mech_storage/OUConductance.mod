: Ornstein-Uhlenbeck conductance, Arbor version of
: `neuron/mech_storage/OUConductance.mod`.
:
:     d g_ou / dt = (mean - g_ou) / tau + sqrt(2 / tau) * stdev * W
:
: with W white noise, integrated with the Euler-Maruyama method by Arbor. The
: process is switched on by an event with positive weight and off by an event
: with negative weight. `i_rec` duplicates the (non-probeable) current `i` for
: recording, it is assigned after `i` so that modcc infers the conductance
: from `i`.

NEURON {
    POINT_PROCESS OUConductance
    NONSPECIFIC_CURRENT i
    RANGE mean, stdev, tau, e
}

PARAMETER {
    e = 0. (mV)
    tau = 100. (ms)
    mean = 0 (uS)
    stdev = 1 (uS)
}

ASSIGNED {
    i_rec (nA)
}

STATE {
    g_ou (uS)
    active
}

WHITE_NOISE {
    W
}

INITIAL {
    g_ou = 0
    active = -1
    i_rec = 0
}

BREAKPOINT {
    SOLVE oup METHOD stochastic
    i = g_ou * (v - e)
    i_rec = i
}

DERIVATIVE oup {
    g_ou' = step_right(active) * ((mean - g_ou) / tau + sqrt(2 / tau) * stdev * W)
}

NET_RECEIVE(weight) {
    if (weight >= 0) {
        g_ou = mean
        active = 1
    }
    else {
        g_ou = 0
        active = -1
    }
}
