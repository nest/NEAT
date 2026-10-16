: Sinusoidal current injection, Arbor version of
: `neuron/mech_storage/SinClamp.mod`. Arbor mechanisms have no access to the
: simulation time, so the current is switched on by an event with positive
: weight, which resets the phase `theta` of the sinusoid, and off by an event
: with negative weight. `theta` grows at a constant rate, which the `cnexp`
: method integrates exactly. `i_rec` duplicates the (non-probeable) current
: `i` for recording.

NEURON {
    POINT_PROCESS SinClamp
    NONSPECIFIC_CURRENT i
    RANGE pkamp, freq, phase, bias
}

PARAMETER {
    pkamp = 1 (nA)
    freq = 1 (Hz)
    phase = 0 (rad)
    bias = 0 (nA)
}

ASSIGNED {
    i_rec (nA)
}

STATE {
    active
    theta (rad)
}

INITIAL {
    active = 0
    theta = phase
    i_rec = 0
}

BREAKPOINT {
    SOLVE dyn METHOD cnexp
    i = -active * (pkamp * sin(theta) + bias)
    i_rec = i
}

DERIVATIVE dyn {
    : freq is in Hz, time in ms
    theta' = 2e-3 * 3.141592653589793 * freq
}

NET_RECEIVE(weight) {
    if (weight >= 0) {
        active = 1
        theta = phase
    }
    else {
        active = 0
    }
}
