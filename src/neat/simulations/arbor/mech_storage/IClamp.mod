: Constant current injection, the Arbor version of NEURON's built-in `IClamp`.
: Arbor mechanisms have no access to the simulation time, so the current is
: switched on by an event with positive weight and off by an event with
: negative weight. `i_rec` duplicates the (non-probeable) current `i` for
: recording. In contrast to Arbor's built-in `i_clamp`, whose current can only
: be recorded as the total stimulus current in a control volume, the current of
: each instance can be recorded separately.

NEURON {
    POINT_PROCESS IClamp
    NONSPECIFIC_CURRENT i
    RANGE amp
}

PARAMETER {
    amp = 0 (nA)
}

ASSIGNED {
    i_rec (nA)
}

STATE {
    active
}

INITIAL {
    active = 0
    i_rec = 0
}

BREAKPOINT {
    i = -active * amp
    i_rec = i
}

NET_RECEIVE(weight) {
    if (weight >= 0) {
        active = 1
    }
    else {
        active = 0
    }
}
