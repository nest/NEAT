NEAT (NEural Analysis Toolkit)
==============================

Introduction
------------

NEAT is a python library for the study, simulation and simplification of
morphological neuron models. NEAT accepts morphologies in the *de facto*
standard .swc format [Cannon1998]_, and implements high-level tools to interact
with and analyze the morphologies.

NEAT also allows for the convenient definition of morphological neuron models.
These models can be simulated, through an interface with the NEURON [Carnevale2004]_ 
and NEST [Gewaltig2007]_ simulators, or can be analyzed with two classical methods: 
*(i)* the separation of variables method [Major1993]_ to obtain impedance kernels 
as a superposition of exponentials and *(ii)* Koch's method to compute impedances
with linearized ion channels analytically in the frequency domain [Koch1985]_.
Furthermore, NEAT implements the neural evaluation tree framework [Wybo2019]_
and an associated C++ simulator, to analyze subunit independence.

Finally, NEAT implements a new and powerful method to simplify morphological
neuron models into compartmental models with few compartments [Wybo2021]_. For
these models, NEAT provides NEURON and NEST interfaces so that they can be
simulated directly.

Documentation
-------------

Documentation for an older version of NEAT (0.9.2) available `here <https://neatdend.readthedocs.io>`_
Note that a new documentation website for the current version (1.0-rc1) is currently
under construction. Please see the changelog (`changelog/v.1.0.md`) for an overview of
the changes.

Installation
------------

**Install**

NEAT can be installed with pip.

.. code-block:: shell

   pip install nest-neat

NEAT can also be installed from the source repository. The following 
instructions are for Linux and Max OSX systems and only use command 
line tools. Please follow the appropriate manuals for Windows systems or
tools with graphical interfaces. 

.. code-block:: shell

   git clone git@github.com:nest/NEAT.git
   cd NEAT
   pip install .

**Post-Install**

Note that if you install NEAT with pip, as above, NEURON will automatically be installed as well.
To use NEAT with `NEST <https://nest-simulator.readthedocs.io/en/stable/index.html>`_, 
you need to manually install NEST and nestml on your system, by following the detailed
`installation instructions <https://nest-simulator.readthedocs.io/en/stable/installation/index.html>`_.

**Testing the installation**

NEAT makes the shell command `neatmodels` available to compile NEAT-defined ion channels
into for NEURON or NEST, so that they can be simulated.
You can test whether this command is available by installing the default ion channels of NEAT:

.. code-block:: shell

    neatmodels install default

This installs the default channels for NEURON. To install them for NEST, use:

.. code-block:: shell

    neatmodels install default -s nest

References
----------

.. [Cannon1998] Cannon et al. (1998) *An online archive of reconstructed hippocampal neurons*, J. Neurosci. methods.
.. [Carnevale2004] Carnevale, Nicholas T. and Hines, Michael L. (2004) *The NEURON book*
.. [Koch1985] Koch, C. and Poggio, T. (1985) *A simple algorithm for solving the cable equation in dendritic trees of arbitrary geometry*, Journal of neuroscience methods, 12(4), pp. 303–315.
.. [Major1993] Major et al. (1993) *Solutions for transients in arbitrarily branching cables: I. Voltage recording with a somatic shunt*, Biophysical journal, 65(1), pp. 423–49.
.. [Martelli03] A. Martelli (2003) *Python in a Nutshell*, O’Reilly Media Inc.
.. [Wybo2019] Wybo, Willem A.M. et al. (2019) *Electrical Compartmentalization in Neurons*, Cell Reports, 26(7), pp. 1759--1773
.. [Wybo2021] Wybo, Willem A.M. et al. (2021) *Data-driven reduction of dendritic morphologies with preserved dendro-somatic responses*, eLife, 10:e60936, pp. 1--26
.. [Gewaltig2007] Gewaltig, Marc-Oliver and Diesmann, Markus. (2007) *NEST (NEural Simulation Tool)*, Scholarpedia, 2(4), pp. 1430
