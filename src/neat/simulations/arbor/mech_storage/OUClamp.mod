: Ornstein-Uhlenbeck current, Arbor version of `neuron/mech_storage/OUClamp.mod`.
:
:     d i_ou / dt = (mean - i_ou) / tau + sqrt(2 / tau) * stdev * W
:
: with W white noise, integrated with the Euler-Maruyama method by Arbor. Arbor
: mechanisms have no access to the simulation time, so the process is switched
: on by an event with positive weight and off by an event with negative weight
: (cf. Arbor's `ou_input` mechanism). `i_rec` duplicates the (non-probeable)
: current `i` for recording, it is assigned after `i` so that modcc infers
: the conductance from `i`.

NEURON {
    POINT_PROCESS OUClamp
    NONSPECIFIC_CURRENT i
    RANGE mean, stdev, tau
}

PARAMETER {
    tau = 100. (ms)
    mean = 0 (nA)
    stdev = 1 (nA)
}

ASSIGNED {
    i_rec (nA)
}

STATE {
    i_ou (nA)
    active
}

WHITE_NOISE {
    W
}

INITIAL {
    i_ou = 0
    active = -1
    i_rec = 0
}

BREAKPOINT {
    SOLVE oup METHOD stochastic
    i = -i_ou
    i_rec = i
}

DERIVATIVE oup {
    i_ou' = step_right(active) * ((mean - i_ou) / tau + sqrt(2 / tau) * stdev * W)
}

NET_RECEIVE(weight) {
    if (weight >= 0) {
        i_ou = mean
        active = 1
    }
    else {
        i_ou = 0
        active = -1
    }
}
