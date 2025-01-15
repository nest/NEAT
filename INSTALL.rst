Installation
============

**Install**

Note: The following instructions are for Linux and Max OSX systems and only use
command line tools. Please follow the appropriate manuals for Windows systems or
tools with graphical interfaces. The most recent version, NEAT 1.0-rc1 can be 
installed from the master branch.

.. code-block:: shell

   git clone git@github.com:unibe-cns/NEAT.git
   cd NEAT
   pip install .

**Post-Install**

Note that if you install NEAT with pip, as above, NEURON will automatically be installed as well.
To use NEAT with `NEST <https://nest-simulator.readthedocs.io/en/stable/index.html>`_, 
you need to manually install NEST on your system, by following the detailed
`installation instructions <https://nest-simulator.readthedocs.io/en/stable/installation/index.html>`_.

**Testing the installation**

NEAT makes the shell command `neatmodels` available. This command compiles groups of NEAT-defined 
ion channels into for NEURON or NEST models, so that they can be simulated.
You can test whether this command is available by installing a set of default ion channels of NEAT:

.. code-block:: shell

    neatmodels install default

This installs the default channels for NEURON. To install them for NEST, use:

.. code-block:: shell

    neatmodels install default -s nest

To test the installation (requires `pytest`)
::

    pytest

