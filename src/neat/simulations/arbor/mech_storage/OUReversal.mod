: Conductance with an Ornstein-Uhlenbeck reversal potential, Arbor version of
: `neuron/mech_storage/OUReversal.mod`.
:
:     d e_ou / dt = (mean - e_ou) / tau + sqrt(2 / tau) * stdev * W
:
: with W white noise, integrated with the Euler-Maruyama method by Arbor. The
: process is switched on by an event with positive weight and off by an event
: with negative weight. `i_rec` duplicates the (non-probeable) current `i` for
: recording, it is assigned after `i` so that modcc infers the conductance
: from `i`.

NEURON {
    POINT_PROCESS OUReversal
    NONSPECIFIC_CURRENT i
    RANGE mean, stdev, tau, g
}

PARAMETER {
    g = 0. (uS)
    tau = 100. (ms)
    mean = 0 (mV)
    stdev = 1 (mV)
}

ASSIGNED {
    i_rec (nA)
}

STATE {
    e_ou (mV)
    active
}

WHITE_NOISE {
    W
}

INITIAL {
    e_ou = mean
    active = -1
    i_rec = 0
}

BREAKPOINT {
    SOLVE oup METHOD stochastic
    i = step_right(active) * g * (v - e_ou)
    i_rec = i
}

DERIVATIVE oup {
    e_ou' = step_right(active) * ((mean - e_ou) / tau + sqrt(2 / tau) * stdev * W)
}

NET_RECEIVE(weight) {
    e_ou = mean
    if (weight >= 0) {
        active = 1
    }
    else {
        active = -1
    }
}
