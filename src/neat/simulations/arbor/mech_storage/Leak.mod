: Passive leak current. Arbor's built-in `pas` has a global reversal
: potential, whereas NEAT requires a reversal that can differ between
: locations.

NEURON {
    SUFFIX Leak
    NONSPECIFIC_CURRENT i
    RANGE g, e
}

PARAMETER {
    g = 0.001 (S/cm2)
    e = -70 (mV)
}

BREAKPOINT {
    i = g * (v - e)
}
