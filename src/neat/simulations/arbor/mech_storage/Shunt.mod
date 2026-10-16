: Static conductance, Arbor version of `neuron/mech_storage/Shunt.mod`.
: `i_rec` duplicates the (non-probeable) current `i` for recording, it is
: assigned after `i` so that modcc infers the conductance from `i`.

NEURON {
    POINT_PROCESS Shunt
    NONSPECIFIC_CURRENT i
    RANGE g, e
}

PARAMETER {
    g = 1 (uS)
    e = 0 (mV)
}

ASSIGNED {
    i_rec (nA)
}

BREAKPOINT {
    i = g * (v - e)
    i_rec = i
}
