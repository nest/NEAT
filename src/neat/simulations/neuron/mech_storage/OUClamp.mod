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

Random generation code adapted to support CoreNEURON, following
https://github.com/neuronsimulator/testcorenrn/blob/master/mod/Gfluct3.mod
ENDCOMMENT


NEURON {
    THREADSAFE
    POINT_PROCESS OUClamp
    NONSPECIFIC_CURRENT i
    RANGE mean, stdev, tau, dt_usr, delay, dur, seed_usr
	THREADSAFE
	BBCOREPOINTER donotuse
}

VERBATIM
#if NRNBBCORE /* running in CoreNEURON */

#define IFNEWSTYLE(arg) arg

#else /* running in NEURON */

/*
   1 means noiseFromRandom was called when _ran_compat was previously 0 .
   2 means noiseFromRandom123 was called when _ran_compat was
previously 0.
*/
static int _ran_compat; /* specifies the noise style for all instances */
#define IFNEWSTYLE(arg) if(_ran_compat == 2) { arg }

#endif /* running in NEURON */ 
ENDVERBATIM  

UNITS {
    (nA) = (nanoamp)
}

PARAMETER {
    delay (ms) : delay until noise starts
    dur (ms) <0, 1e9> : duration of noise
    tau = 100.(ms)
    mean = 0 (nA)
    stdev = 1 (nA)
    seed_usr = 42 (1)
    dt_usr = .1 (ms)
}

ASSIGNED {
    :dt (ms)
    on
    per (ms)
    ival (nA)
    ivar (nA)
    i (nA)
    flag1
    exp_decay
    amp_gauss       (nA)
    donotuse
}

INITIAL {

	VERBATIM
	  if (_p_donotuse) {
	    /* only this style initializes the stream on finitialize */
	    IFNEWSTYLE(nrnran123_setseq((nrnran123_State*)_p_donotuse, 0, 0);)
	  }
	ENDVERBATIM

    on = 0
    ivar = 0
    i = 0
    flag1 = 0
    exp_decay = exp(-dt_usr/tau) : exp(-dt/tau)
    amp_gauss = stdev * sqrt(1. - exp(-2.*dt_usr/tau)) : stdev * sqrt(1. - exp(-2.*dt/tau))
    new_seed(seed_usr)
}

FUNCTION mynormrand(mean, std) {
VERBATIM
	if (_p_donotuse) {
		// corresponding hoc Random distrubution must be Random.normal(0,1)
		double x;
#if !NRNBBCORE
		if (_ran_compat == 2) {
			x = nrnran123_normal((nrnran123_State*)_p_donotuse);
		}else{		
			x = nrn_random_pick((Rand*)_p_donotuse);
		}
#else
		#pragma acc routine(nrnran123_normal) seq
		x = nrnran123_normal((nrnran123_State*)_p_donotuse);
#endif
		x = _lmean + _lstd*x;
		return x;
	}
#if !NRNBBCORE
ENDVERBATIM
	mynormrand = normrand(mean, std)
VERBATIM
#endif
ENDVERBATIM
}


PROCEDURE new_seed(seed) {		: procedure to set the seed
VERBATIM
#if !NRNBBCORE
ENDVERBATIM
	set_seed(seed)
	VERBATIM
	  printf("Setting random generator with seed = %g\n", _lseed);
	ENDVERBATIM
VERBATIM  
#endif
ENDVERBATIM
}


PROCEDURE noiseFromRandom() {
VERBATIM
#if !NRNBBCORE
 {
	Rand** pv = (Rand**)(&_p_donotuse);
	if (_ran_compat == 2) {
		fprintf(stderr, "Gfluct3.noiseFromRandom123 was previously called\n");
		assert(0);
	} 
	_ran_compat = 1;
	if (ifarg(1)) {
		*pv = nrn_random_arg(1);
	}else{
		*pv = (Rand*)0;
	}
 }
#endif
ENDVERBATIM
}


PROCEDURE noiseFromRandom123() {
VERBATIM
#if !NRNBBCORE
 {
	nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);
	if (_ran_compat == 1) {
		fprintf(stderr, "Gfluct3.noiseFromRandom was previously called\n");
		assert(0);
	}
	_ran_compat = 2;
	if (*pv) {
		nrnran123_deletestream(*pv);
		*pv = (nrnran123_State*)0;
	}
	if (ifarg(3)) {
		*pv = nrnran123_newstream3((uint32_t)*getarg(1), (uint32_t)*getarg(2), (uint32_t)*getarg(3));
	}else if (ifarg(2)) {
		*pv = nrnran123_newstream((uint32_t)*getarg(1), (uint32_t)*getarg(2));
	}
 }
#endif
ENDVERBATIM
}


BREAKPOINT {
    SOLVE oup
    i = - ivar
}

PROCEDURE oup() {
    if (t < delay) {
        ivar = 0.
    }
    else { 
        if (flag1 == 0) {
            flag1 = 1
            ivar = mean
        }
        if (t < delay+dur) {
            ivar = mean + exp_decay * (ivar-mean) + amp_gauss * mynormrand(0., 1.)
        }
        else {  
            ivar = 0.
        }
    }
}


VERBATIM
static void bbcore_write(double* x, int* d, int* xx, int *offset, _threadargsproto_) {
	/* error if using the legacy normrand */
	if (!_p_donotuse) {
		fprintf(stderr, "Gfluct3: cannot use the legacy normrand generator for the random stream.\n");
		assert(0);
	}
	if (d) {
		uint32_t* di = ((uint32_t*)d) + *offset;
#if !NRNBBCORE
		if (_ran_compat == 1) { 
			char which;
			Rand** pv = (Rand**)(&_p_donotuse);
			/* error if not using Random123 generator */
			if (!nrn_random_isran123(*pv, di, di+1, di+2)) {
				fprintf(stderr, "Gfluct3: Random123 generator is required\n");
				assert(0);
			}
			/* because coreneuron psolve may not start at t=0 also need the sequence */
			nrn_random123_getseq(*pv, di+3, &which);
			di[4] = (int)which;
		}else{
#else
	{
#endif
			char which;
			nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);
			nrnran123_getids3(*pv, di, di+1, di+2);
			nrnran123_getseq(*pv, di+3, &which);
			di[4] = (int)which;
		}
		/*printf("Gfluct3 bbcore_write %d %d %d %d %d\n", di[0], di[1], di[2], di[3], di[4]);*/
	}
	*offset += 5;
}

static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
	uint32_t* di = ((uint32_t*)d) + *offset;
	nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);
#if !NRNBBCORE
	assert(_ran_compat == 2);
#endif
	if (pv) {
		nrnran123_deletestream(*pv);
	}
	*pv = nrnran123_newstream3(di[0], di[1], di[2]);
	nrnran123_setseq(*pv, di[3], (char)di[4]);
	*offset += 5;
}
ENDVERBATIM
