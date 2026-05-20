TITLE kinetic NMDA receptor model

COMMENT
-----------------------------------------------------------------------------

	Kinetic model of NMDA receptors
	===============================

	10-state gating model:
	Kampa et al. (2004) J Physiol

	  U -- Cl  --  O
         \   | \	    \
          \  |  \      \
         UMg --  ClMg - OMg
		 |	|
		D1	|
		 | \	|
		D2  \	|
		   \	D1Mg
		    \	|
			D2Mg
-----------------------------------------------------------------------------

  Based on voltage-clamp recordings of NMDA receptor-mediated currents in
  nucleated patches of  rat neocortical layer 5 pyramidal neurons (Kampa 2004),
  this model was fit with AxoGraph directly to experimental recordings in
  order to obtain the optimal values for the parameters.

-----------------------------------------------------------------------------

  This mod file does not include mechanisms for the release and time course
  of transmitter; it should to be used in conjunction with a sepearate mechanism
  to describe the release of transmitter and tiemcourse of the concentration
  of transmitter in the synaptic cleft (to be connected to pointer C here).

-----------------------------------------------------------------------------

  See details of NEURON kinetic models in:

  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of
  synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition;
  edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1996.


  Written by Bjoern Kampa in 2004

-----------------------------------------------------------------------------

  Rates modified for near physiological temperatures with Q10 values from
  O.Cais et al 2008, Mg unbinding from Vargas-Caballero 2003, opening and
  closing from Lester and Jahr 1992.

  Tiago Branco 2010

-----------------------------------------------------------------------------

ENDCOMMENT


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	POINT_PROCESS NMDA_Mg_T
	POINTER C
	RANGE U, Cl, D1, D2, O, UMg, ClMg, D1Mg, D2Mg, OMg
	RANGE g, gmax, rb, rmb, rmu, rbMg,rmc1b,rmc1u,rmc2b,rmc2u, Erev, mg
	RANGE Rb, Ru, Rd1, Rr1, Rd2, Rr2, Ro, Rc, Rmb, Rmu
	RANGE RbMg, RuMg, Rd1Mg, Rr1Mg, Rd2Mg, Rr2Mg, RoMg, RcMg
	RANGE Rmd1b,Rmd1u,Rmd2b,Rmd2u,rmd1b,rmd1u,rmd2b,rmd2u
	RANGE Rmc1b,Rmc1u,Rmc2b,Rmc2u
	RANGE vmin, vmax, valence, memb_fraction
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(umho) = (micromho)
	(mM) = (milli/liter)
	(uM) = (micro/liter)
}

PARAMETER {

	Erev	= 5    	(mV)	: reversal potential
	gmax	= 500  	(pS)	: maximal conductance
	mg	= 1  	(mM)	: external magnesium concentration
	vmin 	= -120	(mV)
	vmax 	= 100	(mV)
	valence = -2		: parameters of voltage-dependent Mg block
	memb_fraction = 0.8

: Rates

	Rb		= 10e-3    	(/uM /ms)	: binding
	Ru		= 0.02016 	(/ms)	: unbinding
	Ro		= 46.5e-3   	(/ms)	: opening
	Rc		= 91.6e-3   	(/ms)	: closing
	Rd1		= 0.02266  	(/ms)	: fast desensitisation
	Rr1		= 0.00736  	(/ms)	: fast resensitisation
	Rd2 		= 0.004429	(/ms)	: slow desensitisation
	Rr2 		= 0.0023	(/ms)	: slow resensitisation
	Rmb		= 0.05e-3	(/uM /ms)	: Mg binding Open
	Rmu		= 12800e-3	(/ms)	: Mg unbinding Open
	Rmc1b		= 0.00005e-3	(/uM /ms)	: Mg binding Closed
	Rmc1u		= 0.06	(/ms)	: Mg unbinding Closed
	Rmc2b		= 0.00005e-3	(/uM /ms)	: Mg binding Closed2
	Rmc2u		= 0.06	(/ms)	: Mg unbinding Closed2
	Rmd1b		= 0.00005e-3	(/uM /ms)	: Mg binding Desens1
	Rmd1u		= 0.06	(/ms)	: Mg unbinding Desens1
	Rmd2b		= 0.00005e-3	(/uM /ms)	: Mg binding Desens2
	Rmd2u		= 0.06	(/ms)	: Mg unbinding Desens2
	RbMg		= 10e-3		(/uM /ms)	: binding with Mg
	RuMg		= 0.06156	(/ms)	: unbinding with Mg
	RoMg		= 46.5e-3		(/ms)	: opening with Mg
	RcMg		= 91.6e-3	(/ms)	: closing with Mg
	Rd1Mg		= 0.02163	(/ms)	: fast desensitisation with Mg
	Rr1Mg		= 0.004002	(/ms)	: fast resensitisation with Mg
	Rd2Mg		= 0.002678	(/ms)	: slow desensitisation with Mg
	Rr2Mg		= 0.001932	(/ms)	: slow resensitisation with Mg
}

ASSIGNED {
	v		(mV)	: postsynaptic voltage
	i 		(nA)	: current = g*(v - Erev)
	g 		(pS)	: conductance
	C 		(mM)	: pointer to glutamate concentration

	rb		(/ms)   : binding, [glu] dependent
	rmb		(/ms)	: blocking V and [Mg] dependent
	rmu		(/ms)	: unblocking V and [Mg] dependent
	rbMg		(/ms)	: binding, [glu] dependent
	rmc1b		(/ms)	: blocking V and [Mg] dependent
	rmc1u		(/ms)	: unblocking V and [Mg] dependent
	rmc2b		(/ms)	: blocking V and [Mg] dependent
	rmc2u		(/ms)	: unblocking V and [Mg] dependent
	rmd1b		(/ms)	: blocking V and [Mg] dependent
	rmd1u		(/ms)	: unblocking V and [Mg] dependent
	rmd2b		(/ms)	: blocking V and [Mg] dependent
	rmd2u		(/ms)	: unblocking V and [Mg] dependent
}

STATE {
	: Channel states (all fractions)
	U		: unbound
	Cl		: closed
	D1		: desensitised 1
	D2		: desensitised 2
	O		: open
	UMg		: unbound with Mg
	ClMg		: closed with Mg
	D1Mg		: desensitised 1 with Mg
	D2Mg		: desensitised 2 with Mg
	OMg		: open with Mg
}

INITIAL {
	U = 1
}

BREAKPOINT {
	SOLVE kstates METHOD sparse

	g = gmax * O
	i = (1e-6) * g * (v - Erev)
}

KINETIC kstates {

	rb 	= Rb 	* (1e3) * C
	rbMg 	= RbMg 	* (1e3) * C
	rmb 	= Rmb 	* mg * (1e3) * exp((v-40) * valence * memb_fraction /25)
	rmu 	= Rmu 	* exp((-1)*(v-40) * valence * (1-memb_fraction) /25)
	rmc1b 	= Rmc1b * mg * (1e3) * exp((v-40) * valence * memb_fraction /25)
	rmc1u 	= Rmc1u * exp((-1)*(v-40) * valence * (1-memb_fraction) /25)
	rmc2b 	= Rmc2b * mg * (1e3) * exp((v-40) * valence * memb_fraction /25)
	rmc2u 	= Rmc2u * exp((-1)*(v-40) * valence * (1-memb_fraction) /25)
	rmd1b 	= Rmd1b * mg * (1e3) * exp((v-40) * valence * memb_fraction /25)
	rmd1u 	= Rmd1u * exp((-1)*(v-40) * valence * (1-memb_fraction) /25)
	rmd2b 	= Rmd2b * mg * (1e3) * exp((v-40) * valence * memb_fraction /25)
	rmd2u 	= Rmd2u * exp((-1)*(v-40) * valence * (1-memb_fraction) /25)

	~ U <-> Cl	(rb,Ru)
	~ Cl <-> O	(Ro,Rc)
	~ Cl <-> D1	(Rd1,Rr1)
	~ D1 <-> D2	(Rd2,Rr2)
	~ O <-> OMg	(rmb,rmu)
	~ UMg <-> ClMg 	(rbMg,RuMg)
	~ ClMg <-> OMg 	(RoMg,RcMg)
	~ ClMg <-> D1Mg (Rd1Mg,Rr1Mg)
	~ D1Mg <-> D2Mg (Rd2Mg,Rr2Mg)
	~ U <-> UMg     (rmc1b,rmc1u)
	~ Cl <-> ClMg	(rmc2b,rmc2u)
	~ D1 <-> D1Mg	(rmd1b,rmd1u)
	~ D2 <-> D2Mg	(rmd2b,rmd2u)

	CONSERVE U+Cl+D1+D2+O+UMg+ClMg+D1Mg+D2Mg+OMg = 1
}

