COMMENT
Noise current characterized by gaussian distribution 
with mean mean and standerd deviation stdev.

Borrows from NetStim's code so it can be linked with an external instance 
of the Random class in order to generate output that is independent of 
other instances of InGau.

User specifies the time at which the noise starts, 
and the duration of the noise.
Since a new value is drawn at each time step, 
should be used only with fixed time step integration.
ENDCOMMENT

NEURON {
    POINT_PROCESS WNClamp
    NONSPECIFIC_CURRENT i
    RANGE mean, stdev
    RANGE delay, dur
    THREADSAFE : true only if every instance has its own distinct Random
    POINTER donotuse
}

UNITS {
    (nA) = (nanoamp)
}

PARAMETER {
    delay (ms) : delay until noise starts
    dur (ms) <0, 1e9> : duration of noise
    mean = 0 (nA)
    stdev = 1 (nA)
}

ASSIGNED {
    dt (ms)
    on
    per (ms)
    ival (nA)
    i (nA)
    donotuse
}

INITIAL {
    per = dt
    on = 0
    ival = 0
    i = 0
    net_send(delay, 1)
}

PROCEDURE seed(x) {
    set_seed(x)
}

BEFORE BREAKPOINT {
    i = -ival
: printf("time %f \ti %f\n", t, ival)
}

BREAKPOINT { : this block must exist so that a current is actually generated
}

NET_RECEIVE (w) {
    if (dur>0) {
        if (flag==1) {
            if (on==0) { : turn on
                on=1
                net_send(dur,1) : to turn it off
:                ival = (hi-lo)*urand() + lo : first sample
                ival = stdev*grand() + mean : first sample
                net_send(per, 2) : prepare for next sample
            } else {
                if (on==1) { : turn off
                    on=0
                    ival = 0
                }
            }
        }
        if (flag==2) {
            if (on==1) {
                ival = stdev*grand() + mean
: printf("time %f \ti %f\n", t, ival)
                net_send(per, 2) : prepare for next sample
            }
        }
    }
}

VERBATIM
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
ENDVERBATIM

: FUNCTION erand() {
: FUNCTION urand() {
FUNCTION grand() {
VERBATIM
    if (_p_donotuse) {
        /*
         : Supports separate independent but reproducible streams for
         : each instance. However, the corresponding hoc Random
         : distribution MUST be set to Random.uniform(0,1)
         */
//            _lerand = nrn_random_pick(_p_donotuse);
//            _lurand = nrn_random_pick(_p_donotuse);
            _lgrand = nrn_random_pick(_p_donotuse);
    }else{
        /* only can be used in main thread */
        if (_nt != nrn_threads) {
hoc_execerror("multithread random in InUnif"," only via hoc Random");
        }
ENDVERBATIM
        : the old standby. Cannot use if reproducible parallel sim
        : independent of nhost or which host this instance is on
        : is desired, since each instance on this cpu draws from
        : the same stream
:        erand = exprand(1)
:        urand = scop_random()
        grand = normrand(0,1)
: printf("%f\n", grand)
VERBATIM
    }
ENDVERBATIM
}

PROCEDURE noiseFromRandom() {
VERBATIM
 {
    void** pv = (void**)(&_p_donotuse);
    if (ifarg(1)) {
        *pv = nrn_random_arg(1);
    }else{
        *pv = (void*)0;
    }
 }
ENDVERBATIM
}
