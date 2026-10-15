: Single electrode voltage clamp with series resistance `rs`, a simplified
: Arbor version of NEURON's built-in `SEClamp` with a single clamping level
: `amp1`. Arbor mechanisms have no access to the simulation time, so the clamp
: is active from the start of the simulation and switched off by an event.
: `i_rec` duplicates the (non-probeable) current `i` for recording, it is
: assigned after `i` so that modcc infers the conductance from `i`.

NEURON {
    POINT_PROCESS SEClamp
    NONSPECIFIC_CURRENT i
    RANGE rs, amp1
}

PARAMETER {
    rs = 1 (MOhm)
    amp1 = 0 (mV)
}

ASSIGNED {
    i_rec (nA)
}

STATE {
    active
}

INITIAL {
    active = 1
    i_rec = 0
}

BREAKPOINT {
    i = active * (v - amp1) / rs
    i_rec = i
}

NET_RECEIVE(weight) {
    active = 0
}
