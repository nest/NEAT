# -*- coding: utf-8 -*-
#
# neuronmodel.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

import os
import time
import copy
import warnings
import platform

import numpy as np

from ...trees.morphtree import MorphLoc
from ...trees.phystree import PhysTree, PhysNode
from ...trees.compartmenttree import CompartmentTree
from ...factorydefaults import DefaultPhysiology

try:
    import neuron
    from neuron import h

    h.load_file("stdlib.hoc")  # contains the lambda rule
    h.load_file("stdrun.hoc")  # basic run control

except ModuleNotFoundError:
    warnings.warn(
        "NEURON not available, importing non-functional h module only for doc generation",
        UserWarning,
    )

    # universal iterable mock object
    class H(object):
        def __init__(self):
            pass

        def __getattr__(self, attr):
            try:
                return super(H, self).__getattr__(attr)
            except AttributeError:
                return self.__global_handler

        def __global_handler(self, *args, **kwargs):
            return H()

        def __iter__(self):  # make iterable
            return self

        def __next__(self):
            raise StopIteration

        def __mul__(self, other):  # make multipliable
            return 1.0

        def __rmul__(self, other):
            return self * other

        def __call__(self, *args, **kwargs):  # make callable
            return H()

    h = H()
    neuron = H()
    np_array = np.array

    def array(*args, **kwargs):
        if isinstance(args[0], H):
            print(args)
            print(kwargs)
            return np.eye(2)
        else:
            return np_array(*args, **kwargs)

    np.array = array


def load_neuron_model(name):
    path = os.path.join(
        os.path.dirname(__file__),
        f"tmp/{name}/{platform.machine()}/.libs/libnrnmech.so",
    )
    if os.path.exists(path):
        h.nrn_load_dll(path)  # load all mechanisms
    else:
        path_name = os.path.join(os.path.dirname(__file__), "tmp/")
        raise FileNotFoundError(
            f"The NEURON model named '{name}' is not installed. "
            f"Run 'neatmodels -h' in the terminal for help on "
            f"installing new NEURON models with NEAT. "
            f"Installed models will be in '{path_name}'."
        )


class MechName(object):
    def __init__(self):
        self.names = {"L": "pas"}
        self.ions = ["ca"]

    def __getitem__(self, key):
        if key in self.names:
            return self.names[key]
        elif key in self.ions:
            return "conc_" + key
        else:
            return "I" + key


mechname = MechName()


class NeuronSimNode(PhysNode):
    """
    Subclass of `neat.PhysNode` that implements functionality to instantiate a
    cylindrical `neuron.h.Section` based on its physiological and geometrical
    parameters.
    """

    def __init__(self, index, p3d=None):
        super().__init__(index, p3d)

    def _make_section(self, factorlambda=1.0, pprint=False):
        compartment = h.Section(name=str(self.index))
        compartment.push()
        # create the compartment
        if self.index == 1:
            compartment.diam = 2.0 * self.R  # um (NEURON takes diam=2*r)
            compartment.L = 2.0 * self.R  # um (to get correct surface)
            compartment.nseg = 1
        else:
            compartment.diam = (
                2.0 * self.R
            )  # section radius [um] (NEURON takes diam = 2*r)
            compartment.L = self.L  # section length [um]
            # set number of segments
            if type(factorlambda) == float:
                # nseg according to NEURON book
                compartment.nseg = int(
                    ((compartment.L / (0.1 * h.lambda_f(100.0)) + 0.9) / 2.0) * 2.0
                    + 1.0
                ) * int(factorlambda)
            else:
                compartment.nseg = factorlambda

        # set parameters
        compartment.cm = self.c_m  # uF/cm^2
        compartment.Ra = self.r_a * 1e6  # MOhm*cm --> Ohm*cm
        # insert membrane currents
        for key, current in self.currents.items():
            if current[0] > 1e-10:
                try:
                    compartment.insert(mechname[key])
                except ValueError as e:
                    raise ValueError(str(e) + f" {mechname[key]}")
                for seg in compartment:
                    exec(
                        "seg." + mechname[key] + ".g = " + str(current[0]) + "*1e-6"
                    )  # uS/cm^2 --> S/cm^2
                    exec("seg." + mechname[key] + ".e = " + str(current[1]))  # mV
        # insert concentration mechanisms
        for ion, params in self.concmechs.items():
            compartment.insert(mechname[ion])
            for seg in compartment:
                for param, value in params.items():
                    if param == "gamma":
                        value *= 1e6
                    exec("seg." + mechname[ion] + "." + param + " = " + str(value))
        h.pop_section()

        if pprint:
            print(self)
            print((">>> compartment length = %.2f um" % compartment.L))
            print((">>> compartment diam = %.2f um" % compartment.diam))
            print((">>> compartment nseg = " + str(compartment.nseg)))

        return compartment

    def _make_shunt(self, compartment):
        if self.g_shunt > 1e-10:
            shunt = h.Shunt(compartment(1.0))
            shunt.g = self.g_shunt  # uS
            shunt.e = self.v_ep  # mV
            return shunt
        else:
            return None


class NeuronSimTree(PhysTree):
    """
    Tree class to define NEURON (Carnevale & Hines, 2004) based on `neat.PhysTree`.

    Attributes
    ----------
    sections: dict of hoc sections
        Storage for hoc sections. Keys are node indices.
    shunts: list of hoc mechanisms
        Storage container for shunts
    syns: list of hoc mechanisms
        Storage container for synapses
    iclamps: list of hoc mechanisms
        Storage container for current clamps
    vclamps: lis of hoc mechanisms
        Storage container for voltage clamps
    vecstims: list of hoc mechanisms
        Storage container for vecstim objects
    netcons: list of hoc mechanisms
        Storage container for netcon objects
    vecs: list of hoc vectors
        Storage container for hoc spike vectors
    dt: float
        timestep of the simulator ``[ms]``
    t_calibrate: float
        Time for the model to equilibrate``[ms]``. Not counted as part of the
        simulation.
    factor_lambda : int or float
        If int, the number of segments per section. If float, multiplies the
        number of segments given by the standard lambda rule (Carnevale, 2004)
        to give the number of compartments simulated (default value 1. gives
        the number given by the lambda rule)
    v_init: float
        The initial voltage at which the model is initialized ``[mV]``

    A `NeuronSimTree` can be extended easily with custom point process mechanisms.
    Just make sure that you store the point process in an existing appropriate
    storage container or in a custom storage container, since if all references
    to the hocobject disappear, the object itself will be deleted as well.

    .. code-block:: python
        class CustomSimTree(NeuronSimTree):
            def addCustomPointProcessMech(self, loc, **kwargs):
                loc = MorphLoc(loc, self)

                # create the point process
                pp = h.custom_point_process(self.sections[loc['node']](loc['x']))
                pp.arg1 = kwargs['arg1']
                pp.arg2 = kwargs['arg2']
                ...

                self.storage_container_for_point_process.append(pp)

    If you define a custom storage container, make sure that you overwrite the
    `__init__()` and `delete_model()` functions to make sure it is created and
    deleted properly.
    """

    def __init__(self, arg=None, types=[1, 3, 4]):
        # neuron storage
        self.sections = {}
        self.shunts = []
        self.syns = []
        self.iclamps = []
        self.vclamps = []
        self.vecstims = []
        self.netcons = []
        self.vecs = []
        # simulation parameters
        self.dt = 0.1  # ms
        self.t_calibrate = 0.0  # ms
        self.factor_lambda = 1.0
        self.v_init = -75.0  # mV
        self.indstart = 0
        # initialize the tree structure
        super().__init__(arg=arg, types=types)

    def create_corresponding_node(self, node_index, p3d=None):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return NeuronSimNode(node_index, p3d=p3d)

    def set_simulation_parameters(
        self, factor_lambda=1.0, t_calibrate=0.0, dt=0.025, v_init=-75.0
    ):
        # simulation parameters
        self.dt = dt  # ms
        self.t_calibrate = t_calibrate  # ms
        self.indstart = int(t_calibrate / dt)
        self.factor_lambda = factor_lambda
        self.v_init = v_init  # mV

    def init_model(
        self, dt=0.025, t_calibrate=0.0, v_init=-75.0, factor_lambda=1.0, pprint=False
    ):
        """
        Initialize hoc-objects to simulate the neuron model implemented by this
        tree.

        Parameters
        ----------
        dt: float (default is ``.025`` ms)
            Timestep of the simulation
        t_calibrate: float (default ``0.`` ms)
            The calibration time; time model runs without input to reach its
            equilibrium state before the true simulation starts
        v_init: float (default ``-75.`` mV)
            The initial voltage at which the model is initialized
        factor_lambda: float or int (default 1.)
            If int, the number of segments per section. If float, multiplies the
            number of segments given by the standard lambda rule (Carnevale, 2004)
            to give the number of compartments simulated (default value 1. gives
            the number given by the lambda rule)
        pprint: bool (default ``False``)
            Whether or not to print info on the NEURON model's creation
        """
        self.set_simulation_parameters(
            dt=dt,
            t_calibrate=t_calibrate,
            v_init=v_init,
            factor_lambda=factor_lambda,
        )
        # reset all storage
        self.delete_model()
        # create the NEURON model
        self._create_neuron_tree(pprint=pprint)

    def delete_model(self):
        """
        Delete all stored hoc-objects
        """
        # reset all storage
        self.sections = {}
        self.shunts = []
        self.syns = []
        self.iclamps = []
        self.vclamps = []
        self.vecstims = []
        self.netcons = []
        self.vecs = []
        self.store_locs([{"node": 1, "x": 0.0}], "rec locs")

    def _create_neuron_tree(self, pprint):
        for node in self:
            # create the NEURON section
            compartment = node._make_section(self.factor_lambda, pprint=pprint)
            # connect with parent section
            if not self.is_root(node):
                compartment.connect(self.sections[node.parent_node.index], 1, 0)
            # store
            self.sections.update({node.index: compartment})
            # create a static shunt
            shunt = node._make_shunt(compartment)
            if shunt is not None:
                self.shunts.append(shunt)

    def set_rec_locs(self, locs):
        self.store_locs(rec_locs, "rec locs")

    def add_shunt(self, loc, g, e_r):
        """
        Adds a static conductance at a given location

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the shunt.
        g: float
            The conductance of the shunt (uS)
        e_r: float
            The reversal potential of the shunt (mV)
        """
        loc = MorphLoc(loc, self)
        # create the shunt
        shunt = h.Shunt(self.sections[loc["node"]](loc["x"]))
        shunt.g = g
        shunt.e = e_r
        # store the shunt
        self.shunts.append(shunt)

    def add_double_exp_current(self, loc, tau1, tau2):
        """
        Adds a double exponential input current at a given location

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the current waveform (ms)
        tau2: float
            Decay time of the current waveform (ms)
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.epsc_double_exp(self.sections[loc["node"]](loc["x"]))
        syn.tau1 = tau1
        syn.tau2 = tau2
        # store the synapse
        self.syns.append(syn)

    def add_exp_synapse(self, loc, tau, e_r):
        """
        Adds a single-exponential conductance-based synapse

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau: float
            Decay time of the conductance window (ms)
        e_r: float
           Reversal potential of the synapse (mV)
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.exp_AMPA_NMDA(self.sections[loc["node"]](loc["x"]))
        syn.tau = tau
        syn.e = e_r
        # store the synapse
        self.syns.append(syn)

    def add_double_exp_synapse(self, loc, tau1, tau2, e_r):
        """
        Adds a double-exponential conductance-based synapse

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the conductance window (ms)
        tau2: float
            Decay time of the conductance window (ms)
        e_r: float
            Reversal potential of the synapse (mV)
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.Exp2Syn(self.sections[loc["node"]](loc["x"]))
        syn.tau1 = tau1
        syn.tau2 = tau2
        syn.e = e_r
        # store the synapse
        self.syns.append(syn)

    def add_nmda_synapse(self, loc, tau, tau_nmda, e_r=0.0, nmda_ratio=1.7):
        """
        Adds a single-exponential conductance-based synapse with an AMPA and an
        NMDA component

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau: float
            Decay time of the AMPA conductance window (ms)
        tau_nmda: float
            Decay time of the NMDA conductance window (ms)
        e_r: float (optional, default ``0.`` mV)
           Reversal potential of the synapse (mV)
        nmda_ratio: float (optional, default 1.7)
            The ratio of the NMDA over AMPA component. Means that the maximum of
            the NMDA conductance window is ``nmda_ratio`` times the maximum of
            the AMPA conductance window.
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.exp_AMPA_NMDA(self.sections[loc["node"]](loc["x"]))
        syn.tau = tau
        syn.tau_NMDA = tau_nmda
        syn.e = e_r
        syn.NMDA_ratio = nmda_ratio
        # store the synapse
        self.syns.append(syn)

    def add_double_exp_nmda_synapse(
        self, loc, tau1, tau2, tau1_nmda, tau2_nmda, e_r=0.0, nmda_ratio=1.7
    ):
        """
        Adds a double-exponential conductance-based synapse with an AMPA and an
        NMDA component

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the AMPA conductance window (ms)
        tau2: float
            Decay time of the AMPA conductance window (ms)
        tau1_nmda: float
            Rise time of the NMDA conductance window (ms)
        tau2_nmda: float
            Decay time of the NMDA conductance window (ms)
        e_r: float (optional, default ``0.`` mV)
           Reversal potential of the synapse (mV)
        nmda_ratio: float (optional, default 1.7)
            The ratio of the NMDA over AMPA component. Means that the maximum of
            the NMDA conductance window is ``nmda_ratio`` times the maximum of
            the AMPA conductance window.
        """
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.double_exp_AMPA_NMDA(self.sections[loc["node"]](loc["x"]))
        syn.tau1 = tau1
        syn.tau2 = tau2
        syn.tau1_NMDA = tau1_nmda
        syn.tau2_NMDA = tau2_nmda
        syn.e = e_r
        syn.NMDA_ratio = nmda_ratio
        # store the synapse
        self.syns.append(syn)

    def add_i_clamp(self, loc, amp, delay, dur):
        """
        Injects a DC current step at a given lcoation

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the current.
        amp: float
            The amplitude of the current (nA)
        delay: float
            The delay of the current step onset (ms)
        dur: float
            The duration of the current step (ms)
        """
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.IClamp(self.sections[loc["node"]](loc["x"]))
        iclamp.delay = delay + self.t_calibrate  # ms
        iclamp.dur = dur  # ms
        iclamp.amp = amp  # nA
        # store the iclamp
        self.iclamps.append(iclamp)

    def add_sin_clamp(self, loc, amp, delay, dur, bias, freq, phase):
        """
        Injects a sinusoidal current at a given lcoation

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the current.
        amp: float
            The amplitude of the current (nA)
        delay: float
            The delay of the current onset (ms)
        dur: float
            The duration of the current (ms)
        bias: float
            Constant baseline added to the sinusoidal waveform (nA)
        freq: float
            Frequency of the sinusoid (Hz)
        phase: float
            Phase of the sinusoid (rad)
        """
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.SinClamp(self.sections[loc["node"]](loc["x"]))
        iclamp.delay = delay + self.t_calibrate  # ms
        iclamp.dur = dur  # ms
        iclamp.pkamp = amp  # nA
        iclamp.bias = bias  # nA
        iclamp.freq = freq  # Hz
        iclamp.phase = phase  # rad
        # store the iclamp
        self.iclamps.append(iclamp)

    def add_ou_clamp(self, loc, tau, mean, stdev, delay, dur, seed=None):
        """
        Injects a Ornstein-Uhlenbeck current at a given lcoation

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau: float
            Time-scale of the OU process (ms)
        mean: float
            Mean of the OU process (nA)
        stdev: float
            Standard deviation of the OU process (nA)
        delay: float
            The delay of current onset from the start of the simulation (ms)
        dur: float
            The duration of the current input (ms)
        seed: int, optional
            Seed for the random number generator
        """
        seed = np.random.randint(1e16) if seed is None else seed
        loc = MorphLoc(loc, self)
        # create the current clamp
        if tau > 1e-9:
            iclamp = h.OUClamp(self.sections[loc["node"]](loc["x"]))
            iclamp.tau = tau
        else:
            iclamp = h.WNclamp(self.sections[loc["node"]](loc["x"]))
        iclamp.mean = mean  # nA
        iclamp.stdev = stdev  # nA
        iclamp.delay = delay + self.t_calibrate  # ms
        iclamp.dur = dur  # ms
        iclamp.seed_usr = seed  # ms
        iclamp.dt_usr = self.dt  # ms
        # store the iclamp
        self.iclamps.append(iclamp)

    def add_ou_conductance(self, loc, tau, mean, stdev, e_r, delay, dur, seed=None):
        """
        Injects a Ornstein-Uhlenbeck conductance at a given location

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the conductance.
        tau: float
            Time-scale of the OU process (ms)
        mean: float
            Mean of the OU process (uS)
        stdev: float
            Standard deviation of the OU process (uS)
        e_r: float
            Reversal of the current (mV)
        delay: float
            The delay of current onset from the start of the simulation (ms)
        dur: float
            The duration of the current input (ms)
        seed: int, optional
            Seed for the random number generator
        """
        seed = np.random.randint(1e16) if seed is None else seed
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.OUConductance(self.sections[loc["node"]](loc["x"]))
        iclamp.tau = tau
        iclamp.mean = mean  # uS
        iclamp.stdev = stdev  # uS
        iclamp.e_r = e_r  # mV
        iclamp.delay = delay + self.t_calibrate  # ms
        iclamp.dur = dur  # ms
        iclamp.seed_usr = seed  # ms
        iclamp.dt_usr = self.dt  # ms
        # store the iclamp
        self.iclamps.append(iclamp)

    def add_ou_reversal(self, loc, tau, mean, stdev, g_val, delay, dur, seed=None):
        seed = np.random.randint(1e16) if seed is None else seed
        loc = MorphLoc(loc, self)
        # create the current clamp
        iclamp = h.OUReversal(self.sections[loc["node"]](loc["x"]))
        iclamp.tau = tau  # ms
        iclamp.mean = mean  # mV
        iclamp.stdev = stdev  # mV
        iclamp.g = g_val  # uS
        iclamp.delay = delay + self.t_calibrate  # ms
        iclamp.dur = dur  # ms
        iclamp.seed_usr = seed  # ms
        iclamp.dt_usr = self.dt  # ms
        # store the iclamp
        self.iclamps.append(iclamp)

    def add_v_clamp(self, loc, e_c, dur):
        """
        Adds a voltage clamp at a given location

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            The location of the conductance.
        e_c: float
            The clamping voltage (mV)
        dur: float, ms
            The duration of the voltage clamp
        """
        loc = MorphLoc(loc, self)
        # add the voltage clamp
        vclamp = h.SEClamp(self.sections[loc["node"]](loc["x"]))
        vclamp.rs = 0.01
        vclamp.dur1 = dur
        vclamp.amp1 = e_c
        # store the vclamp
        self.vclamps.append(vclamp)

    def set_spiketrain(self, syn_index, syn_weight, spike_times):
        """
        Each hoc point process that receive spikes through should by appended to
        the synapse stack (stored under the list `self.syns`).

        Default :class:`NeuronSimTree` point processes that are added to
        `self.syns` are:
        - `self.add_double_exp_current()`
        - `self.addExpSyn()`
        - `self.addDoubleExpSyn()`
        - `self.addDoubleExpSyn()`
        - `self.add_nmda_synapse()`
        - `self.add_double_exp_nmda_synapse()`

        With this function, these synapse can be set to receive a specific spike
        train.

        Parameters
        ----------
        syn_index: int
            index of the point process in the synapse stack
        syn_weight: float
            weight of the synapse (maximal value of the conductance window)
        spike_times: list or `np.array` of floats
            the spike times
        """
        # add spiketrain
        spks = np.array(spike_times) + self.t_calibrate
        spks_vec = h.Vector(spks.tolist())
        vecstim = h.VecStim()
        vecstim.play(spks_vec)
        netcon = h.NetCon(vecstim, self.syns[syn_index], 0, self.dt, syn_weight)
        # store the objects
        self.vecs.append(spks_vec)
        self.vecstims.append(vecstim)
        self.netcons.append(netcon)

    def run(
        self,
        t_max,
        downsample=1,
        dt_rec=None,
        record_from_syns=False,
        record_from_iclamps=False,
        record_from_vclamps=False,
        record_from_channels=False,
        record_v_deriv=False,
        record_concentrations=[],
        record_currents=[],
        spike_rec_loc=None,
        spike_rec_thr=-20.0,
        pprint=False,
    ):
        """
        Run the NEURON simulation. Records at all locations stored
        under the name 'rec locs' on `self` (see `MorphTree.store_locs()`)

        Parameters
        ----------
        t_max: float
            Duration of the simulation
        downsample: int (> 0)
            Records the state of the model every `downsample` time-steps
        dt_rec: float or None
            recording time step (if `None` is given, defaults to the simulation
            time-step)
        record_from_syns: bool (default ``False``)
            Record currents of synapstic point processes (in `self.syns`).
            Accessible as `np.ndarray` in the output dict under key 'i_syn'
        record_from_iclamps: bool (default ``False``)
            Record currents of iclamps (in `self.iclamps`)
            Accessible as `np.ndarray` in the output dict under key 'i_clamp'
        record_from_vclamps: bool (default ``False``)
            Record currents of vclamps (in `self.vclamps`)
            Accessible as `np.ndarray` in the output dict under key 'i_vclamp'
        record_from_channels: bool (default ``False``)
            Record channel state variables from `neat` defined channels in `self`,
            at locations stored under 'rec locs'
            Accessible as `np.ndarray` in the output dict under key 'chan'
        record_v_deriv: bool (default ``False``)
            Record voltage derivative at locations stored under 'rec locs'
            Accessible as `np.ndarray` in the output dict under key 'dv_dt'
        record_concentrations: list (default ``[]``)
            Record ion concentration at locations stored under 'rec locs'
            Accessible as `np.ndarray` in the output dict with as key the ion's
            name
        record_currents: list (default ``[]``)
            Record ion currents at locations stored under 'rec locs'
            Accessible as `np.ndarray` in the output dict with as key the ion's
            name
        spike_rec_loc: `neat.MorphLoc` (default ``None``)
            Record the output spike times from this location
        spike_rec_thr: float
            Spike threshold

        Returns
        -------
        dict
            Dictionary with the results of the simulation. Contains time and
            voltage as `np.ndarray` at locations stored under the name '
            rec locs', respectively with keys 't' and 'v_m'. Also contains
            traces of other recorded variables if the option to record them was
            set to ``True``
        """
        assert isinstance(downsample, int) and downsample > 0
        if dt_rec is None:
            dt_rec = self.dt
        indstart = int(self.t_calibrate / dt_rec)

        # simulation time recorder
        res = {"t": h.Vector()}
        res["t"].record(h._ref_t, dt_rec)

        # voltage recorders
        res["v_m"] = []
        for loc in self.get_locs("rec locs"):
            res["v_m"].append(h.Vector())
            res["v_m"][-1].record(self.sections[loc["node"]](loc["x"])._ref_v, dt_rec)

        # synapse current recorders
        if record_from_syns:
            res["i_syn"] = []
            for syn in self.syns:
                res["i_syn"].append(h.Vector())
                res["i_syn"][-1].record(syn._ref_i, dt_rec)

        # current clamp current recorders
        if record_from_iclamps:
            res["i_clamp"] = []
            for iclamp in self.iclamps:
                res["i_clamp"].append(h.Vector())
                res["i_clamp"][-1].record(iclamp._ref_i, dt_rec)

        # voltage clamp current recorders
        if record_from_vclamps:
            res["i_vclamp"] = []
            for vclamp in self.vclamps:
                res["i_vclamp"].append(h.Vector())
                res["i_vclamp"][-1].record(vclamp._ref_i, dt_rec)

        # channel state variable recordings
        if record_from_channels:
            res["chan"] = {}
            channel_names = self.get_channels_in_tree()
            for channel_name in channel_names:
                res["chan"][channel_name] = {
                    str(var): [] for var in self.channel_storage[channel_name].statevars
                }
                for loc in self.get_locs("rec locs"):
                    for ind, varname in enumerate(
                        self.channel_storage[channel_name].statevars
                    ):
                        var = str(varname)
                        # assure xcoordinate is refering to proper neuron section (not endpoint)
                        xx = loc["x"]
                        if xx < 1e-3:
                            xx += 1e-3
                        elif xx > 1.0 - 1e-3:
                            xx -= 1e-3
                        # create the recorder
                        try:
                            rec_vec = h.Vector()
                            exec(
                                "rec_vec.record(self.sections[loc[0]](xx)."
                                + mechname[channel_name]
                                + "._ref_"
                                + str(var)
                                + f", {dt_rec})"
                            )
                        except AttributeError:
                            # the channel does not exist here
                            rec_vec = None
                            rec_vec = []
                        res["chan"][channel_name][var].append(rec_vec)

        if len(record_concentrations) > 0:
            for c_ion in record_concentrations:
                res[c_ion] = []
                for loc in self.get_locs("rec locs"):
                    try:
                        rec_vec = h.Vector()
                        exec(
                            "rec_vec.record(self.sections[loc['node']](loc['x'])._ref_"
                            + c_ion
                            + f"i, {dt_rec})"
                        )
                    except AttributeError:
                        rec_vec = None
                        rec_vec = []
                    res[c_ion].append(rec_vec)

        if len(record_currents) > 0:
            for c_ion in record_currents:
                curr_name = f"i{c_ion}"
                res[curr_name] = []
                for loc in self.get_locs("rec locs"):
                    try:
                        rec_vec = h.Vector()
                        exec(
                            f"rec_vec.record(self.sections[loc['node']](loc['x'])._ref_{curr_name}, {dt_rec})"
                        )
                    except AttributeError:
                        rec_vec = None
                        rec_vec = []
                    res[curr_name].append(rec_vec)

        # record voltage derivative
        if record_v_deriv:
            res["dv_dt"] = []
            for ii, loc in enumerate(self.get_locs("rec locs")):
                res["dv_dt"].append(h.Vector())
                # res['dv_dt'][-1].deriv(res['v_m'][ii], self.dt)
        if spike_rec_loc is not None:
            # add spike detector at spike_rec_loc
            self.spike_detector = h.NetCon(
                self.sections[spike_rec_loc[0]](spike_rec_loc[1])._ref_v,
                None,
                sec=self.sections[spike_rec_loc[0]],
            )
            self.spike_detector.threshold = spike_rec_thr
            res["spikes"] = h.Vector()
            self.spike_detector.record(res["spikes"])

        # initialize
        # neuron.celsius=37.
        h.dt = self.dt
        # simulate
        if pprint:
            print(">>> Simulating the NEURON model for " + str(t_max) + " ms. <<<")
        start = time.process_time()
        h.finitialize(self.v_init)
        h.continuerun(t_max + self.t_calibrate)
        stop = time.process_time()
        if pprint:
            print(">>> Elapsed time: " + str(stop - start) + " seconds. <<<")
        runtime = stop - start

        # contains default concentrations
        default_phys = DefaultPhysiology()
        # compute derivative
        if "dv_dt" in res:
            for ii, loc in enumerate(self.get_locs("rec locs")):
                res["dv_dt"][ii].deriv(res["v_m"][ii], dt_rec, 2)
                res["dv_dt"][ii] = np.array(res["dv_dt"][ii])[indstart:][::downsample]
            res["dv_dt"] = np.array(res["dv_dt"])
        # cast recordings into numpy arrays
        res["t"] = np.array(res["t"])[indstart:][::downsample] - self.t_calibrate
        for key in (
            set(res.keys())
            - {"t", "chan", "dv_dt", "spikes"}
            - default_phys.conc.keys()
        ):
            if key in res and len(res[key]) > 0:
                arrlist = []
                for reslist in res[key]:
                    arr = (
                        np.array(reslist)[indstart:][::downsample]
                        if len(reslist) > 0
                        else np.zeros_like(res["t"])
                    )
                    arrlist.append(arr)
                res[key] = np.array(arrlist)
                # res[key] = np.array([np.array(reslist)[indstart:][::downsample] \
                #                      for reslist in res[key]])
                if key in ("i_syn", "i_clamp", "i_vclamp"):
                    res[key] *= -1.0
        # cast concentration recordings into numpy arrays, substitute default concentration
        # if no concentration recording is found
        for key in default_phys.conc.keys():
            if key in res and len(res[key]) > 0:
                arrlist = []
                for reslist in res[key]:
                    arr = (
                        np.array(reslist)[indstart:][::downsample]
                        if len(reslist) > 0
                        else np.ones_like(res["t"]) * default_phys.conc[key]
                    )
                    arrlist.append(arr)
                res[key] = np.array(arrlist)
                # res[key] = np.array([np.array(reslist)[indstart:][::downsample] \
                #                      for reslist in res[key]])
                if key in ("i_syn", "i_clamp", "i_vclamp"):
                    res[key] *= -1.0
        # cast channel recordings into numpy arrays
        if "chan" in res:
            for channel_name in channel_names:
                channel = self.channel_storage[channel_name]
                for ind0, varname in enumerate(channel.statevars):
                    var = str(varname)
                    for ind1 in range(len(self.get_locs("rec locs"))):
                        res["chan"][channel_name][var][ind1] = np.array(
                            res["chan"][channel_name][var][ind1]
                        )[indstart:][::downsample]
                        if len(res["chan"][channel_name][var][ind1]) == 0:
                            res["chan"][channel_name][var][ind1] = np.zeros_like(
                                res["t"]
                            )
                    res["chan"][channel_name][var] = np.array(
                        res["chan"][channel_name][var]
                    )
                # compute P_open
                # sv = np.zeros((len(channel.statevars), len(self.get_locs('rec locs')), len(res['t'])))
                sv = {}
                for varname in channel.statevars:
                    var = str(varname)
                    sv[var] = res["chan"][channel_name][var]
                res["chan"][channel_name]["p_open"] = channel.compute_p_open(
                    res["v_m"], **sv
                )
        # cast spike recording to numpy array
        if "spikes" in res:
            res["spikes"] = np.array(list(res["spikes"])) - self.t_calibrate
            self.spike_detector = None
            del self.spike_detector

        res["runtime"] = runtime

        return res

    def calc_e_eq(self, t_dur=100.0, set_v_ep=True):
        """
        Compute the equilibrium potentials in the middle (``x=0.5``) of each node.

        Parameters
        ----------
        t_dur: float (optional, default ``100.`` ms)
            The duration of the simulation
        set_v_ep: bool (optional, default ``True``)
            Store the equilibrium potential as the ``PhysNode.v_ep`` attribute
        """
        self.init_model(
            dt=self.dt,
            t_calibrate=self.t_calibrate,
            v_init=self.v_init,
            factor_lambda=self.factor_lambda,
        )
        self.store_locs([(n.index, 0.5) for n in self], name="rec locs")
        res = self.run(t_dur)
        v_eq = res["v_m"][:-1]
        if set_v_ep:
            for node, e in zip(self, v_eq):
                node.set_v_ep(v_eq)

        return v_eq

    def calc_impedance_matrix(
        self,
        loc_arg,
        i_amp=0.001,
        t_dur=100.0,
        pplot=False,
        factor_lambda=1.0,
        t_calibrate=0.0,
        dt=0.025,
        v_init=-75.0,
    ):
        locs = self.convert_loc_arg_to_locs(loc_arg)
        z_mat = np.zeros((len(locs), len(locs)))
        for ii, loc0 in enumerate(locs):
            for jj, loc1 in enumerate(locs):
                self.init_model(
                    dt=dt,
                    t_calibrate=t_calibrate,
                    v_init=v_init,
                    factor_lambda=factor_lambda,
                )
                self.add_i_clamp(loc0, i_amp, 0.0, t_dur)
                self.store_locs([loc0, loc1], "rec locs", warn=False)
                # simulate
                res = self.run(t_dur)
                self.delete_model()
                # voltage deflections
                v_trans = res["v_m"][1][-int(1.0 / self.dt)] - res["v_m"][1][0]
                # compute impedances
                z_mat[ii, jj] = v_trans / i_amp
                if pplot:
                    import matplotlib.pyplot as pl

                    pl.figure()
                    pl.plot(res["t"], res["v_m"][1])
                    pl.show()

        return z_mat

    def calc_zt(
        self,
        loc0,
        loc1,
        i_amp=0.001,
        dt_pulse=0.1,
        dstep=-2,
        t_max=100.0,
        factor_lambda=1.0,
        t_calibrate=0.0,
        dt=0.025,
        v_init=-75.0,
    ):
        """
        Computes the impulse response kernel between two locations by measuring the
        voltage at `loc1` in response to an input current pulse at `loc0`.

        Parameters
        ----------
        loc0: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer kernel is computed
        loc1: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer kernel is computed
        i_amp : float, optional
            amplitude of the input current pulse [nA], by default 0.001
        dt_pulse : float, optional
            duration of the input current pulse, by default 0.1
        dstep : int, optional
            offset form t=0 in no. of timesteps, by default -2
        t_max : float, optional
            simulation time, by default 100.0
        factor_lambda : float, optional
            If int, the number of segments per section. If float, multiplies the
            number of segments given by the standard lambda rule (Carnevale, 2004)
            to give the number of compartments simulated (default value 1. gives
            the number given by the lambda rule), by default 1.0
        t_calibrate : float, optional
            Time for the model to equilibrate``[ms]``. Not counted as part of the
            simulation., by default 0.0
        dt : float, optional
            Timestep of the simulation, by default 0.025
        v_init : float, optional
            The initial voltage at which the model is initialized ``[mV]``
            Returns, by default -75.0

        Returns
        -------
        np.array
            The time array in [ms]
        np.array
            The impulse response kernel in [MOhm/ms]
        """
        (loc0, loc1) = self.convert_loc_arg_to_locs([loc0, loc1])
        self.set_simulation_parameters(
            dt=dt, t_calibrate=t_calibrate, v_init=v_init, factor_lambda=factor_lambda
        )
        t0 = 5.0
        j0 = int(t0 / self.dt)
        nt = int(t_max / self.dt) - 1
        i0 = int(dt_pulse / self.dt)
        if dstep < -i0:
            dstep = -i0
        self.init_model(
            dt=dt,
            t_calibrate=t_calibrate,
            v_init=v_init,
            factor_lambda=factor_lambda,
        )
        self.add_i_clamp(loc0, i_amp, t0, dt_pulse)
        self.store_locs([loc0, loc1], "rec locs", warn=False)
        # simulate
        res = self.run(t_max + dt_pulse + 3.0 * t0)
        self.delete_model()
        # voltage deflections
        v_trans = (
            res["v_m"][1][j0 + i0 + dstep : j0 + i0 + dstep + nt] - res["v_m"][1][0]
        )
        # compute impedances
        z_trans = v_trans / (i_amp * dt_pulse)

        return res["t"][i0 + dstep : i0 + dstep + nt], z_trans

    def calc_impulse_response_matrix(
        self,
        loc_arg,
        i_amp=0.001,
        dt_pulse=0.1,
        dstep=-2,
        t_max=100.0,
        factor_lambda=1.0,
        t_calibrate=0.0,
        dt=0.025,
        v_init=-75.0,
    ):
        """
        Computes matrix of impulse response kernels between any pairs of locations
        in `loc_arg` by measuring the voltage in response to an input current pulse.

        Parameters
        ----------
        loc_arg : `list` of locations or string
            if `list` of locations, specifies the locations for which the
            impulse response kernels are evaluated, if ``string``, specifies the
            name under which a set of location is stored
        i_amp : float, optional
            amplitude of the input current pulse [nA], by default 0.001
        dt_pulse : float, optional
            duration of the input current pulse, by default 0.1
        dstep : int, optional
            offset form t=0 in no. of timesteps, by default -2
        t_max : float, optional
            simulation time, by default 100.0
        factor_lambda : float, optional
            If int, the number of segments per section. If float, multiplies the
            number of segments given by the standard lambda rule (Carnevale, 2004)
            to give the number of compartments simulated (default value 1. gives
            the number given by the lambda rule), by default 1.0
        t_calibrate : float, optional
            Time for the model to equilibrate``[ms]``. Not counted as part of the
            simulation., by default 0.0
        dt : float, optional
            Timestep of the simulation, by default 0.025
        v_init : float, optional
            The initial voltage at which the model is initialized ``[mV]``
            Returns, by default -75.0

        Returns
        -------
        np.array
            The time array in [ms]
        np.ndarray (ndim=3)
            The impulse response kernel in [MOhm/ms], first dimension corresponds to time,
            last two dimansion correspond to the lcoations
        """
        self.set_simulation_parameters(
            dt=dt, t_calibrate=t_calibrate, v_init=v_init, factor_lambda=factor_lambda
        )
        locs = self.convert_loc_arg_to_locs(loc_arg)
        t0 = 5.0
        j0 = int(t0 / self.dt)
        nt = int(t_max / self.dt) - 1
        i0 = int(dt_pulse / self.dt)
        if dstep < -i0:
            dstep = -i0
        zk_mat = np.zeros((nt, len(locs), len(locs)))
        for ii, loc_in in enumerate(locs):
            self.init_model(
                dt=dt,
                t_calibrate=t_calibrate,
                v_init=v_init,
                factor_lambda=factor_lambda,
            )
            self.add_i_clamp(loc_in, i_amp, t0, dt_pulse)
            self.store_locs(locs, "rec locs", warn=False)
            # simulate
            res = self.run(t_max + dt_pulse + 3.0 * t0)
            self.delete_model()
            # voltage deflections
            v_trans = (
                res["v_m"][:, j0 + i0 + dstep : j0 + i0 + dstep + nt]
                - res["v_m"][:, 0:1]
            )
            # compute impulse response kernels
            zk_mat[:, ii, :] = v_trans.T / (i_amp * dt_pulse)
        return res["t"][i0 + dstep : i0 + dstep + nt], zk_mat


class NeuronCompartmentNode(NeuronSimNode):
    """
    Subclass of `NeuronSimNode` that defines a cylinder with fake geometry in
    NEURON to result in the effective simulation as a single compartment.
    """

    def __init__(self, index):
        super().__init__(index)

    def get_child_nodes(self, skip_inds=[]):
        return super().get_child_nodes(skip_inds=skip_inds)

    def _make_section(self, pprint=False):
        compartment = neuron.h.Section(name=str(self.index))
        compartment.push()
        # create the compartment
        if "points_3d" in self.content:
            points = self.content["points_3d"]
            h.pt3dadd(*points[0], sec=compartment)
            h.pt3dadd(*points[1], sec=compartment)
            h.pt3dadd(*points[2], sec=compartment)
            h.pt3dadd(*points[3], sec=compartment)
        else:
            compartment.diam = (
                2.0 * self.R
            )  # section radius [um] (NEURON takes diam = 2*r)
            compartment.L = self.L  # section length [um]
        # set number of segments to one
        compartment.nseg = 1

        # set parameters
        compartment.cm = self.c_m  # uF/cm^2
        compartment.Ra = self.r_a * 1e6  # MOhm*cm --> Ohm*cm
        # insert membrane currents
        for key, current in self.currents.items():
            if current[0] > 1e-10:
                compartment.insert(mechname[key])
                for seg in compartment:
                    exec(
                        "seg." + mechname[key] + ".g = " + str(current[0]) + "*1e-6"
                    )  # uS/cm^2 --> S/cm^2
                    exec("seg." + mechname[key] + ".e = " + str(current[1]))  # mV
        # insert concentration mechanisms
        for ion, params in self.concmechs.items():
            compartment.insert(mechname[ion])
            for seg in compartment:
                for param, value in params.items():
                    exec("seg." + mechname[ion] + "." + param + " = " + str(value))
        h.pop_section()

        if pprint:
            print(self)
            print((">>> compartment length = %.2f um" % compartment.L))
            print((">>> compartment diam = %.2f um" % compartment.diam))
            print((">>> compartment nseg = " + str(compartment.nseg)))

        return compartment


class NeuronCompartmentTree(NeuronSimTree):
    """
    Creates a `neat.NeuronCompartmentTree` to simulate reduced compartmentment
    models from a `neat.CompartmentTree`.

    Parameters
    ----------
    ctree: `neat.CompartmentTree`
        The tree containing the parameters of the reduced compartmental model
        to be simulated
    fake_c_m: float
        Fake value for the membrance capacitance density, rescales cylinder
        surface
    fake_r_a: float
        Fake value for the axial resistance, rescales cylinder length

    Attributes
    ----------
    equivalent_locs: list of tuples
        'Fake' locations corresponding to each compartment, which are
        used to insert hoc point process at the compartments using
        the same functions definitions as for as for a morphological
        `neat.NeuronSimTree`.

    Notes
    -----
    - Note that this class inherits from `neat.NeuronSimTree` and *not* from
    `neat.CompartmentTree`. This is because NEAT defines a fake morphology to
    implement the compartment model in NEURON, and also to reuse the functionality
    implemented by `neat.NeuronSimTree`. Any function that is not explicitly
    redefined from `neat.NeuronSimTree` can be called in the same way for this
    compartment model.
    - Locations to this class can be provided either as fake morphology locations
    -- i.e. a tuple `(node.index, x-location in [0,1])` -- where the value of the
    x-location is ignored since the nodes here are single compartments, as in
    the `neat.CompartmentTree`, and not cylinders, as in `neat.MorphTree` or
    subclasses, or as location indices, where the index corresponds to the location
    in the original list of locations from which the `neat.CompartmentTree` was
    derived.
    """

    def __init__(self, ctree, fake_c_m=1.0, fake_r_a=100.0 * 1e-6, method=2):

        try:
            assert issubclass(ctree.__class__, CompartmentTree)
        except AssertionError as e:
            raise ValueError(
                "`neat.NeuronCompartmentTree` can only be instantiated "
                "from a `neat.CompartmentTree` or derived class"
            )
        super().__init__(ctree, types=[1, 3, 4])
        self.equivalent_locs = ctree.get_equivalent_locs()
        self._create_reduced_neuron_model(
            ctree,
            fake_c_m=fake_c_m,
            fake_r_a=fake_r_a,
            method=method,
        )

    def _create_reduced_neuron_model(
        self, ctree, fake_c_m=1.0, fake_r_a=100.0 * 1e-6, method=2
    ):
        # calculate geometry that will lead to correct constants
        arg1, arg2 = ctree.compute_fake_geometry(
            fake_c_m=fake_c_m,
            fake_r_a=fake_r_a,
            factor_r_a=1e-6,
            delta=1e-10,
            method=method,
        )
        if method == 1:
            points = arg1
            surfaces = arg2
            for ii, comp_node in enumerate(ctree):
                pts = points[ii]
                sim_node = self.__getitem__(comp_node.index, skip_inds=[])
                sim_node.set_p3d(
                    np.array(pts[0][:3]), (pts[0][3] + pts[-1][3]) / 2.0, 3
                )

            # fill the tree with the currents
            for ii, sim_node in enumerate(self):
                comp_node = ctree[ii]
                sim_node.currents = {
                    chan: [g / surfaces[comp_node.index], e]
                    for chan, (g, e) in comp_node.currents.items()
                }
                sim_node.concmechs = copy.deepcopy(comp_node.concmechs)
                for concmech in sim_node.concmechs.values():
                    concmech.gamma *= surfaces[comp_node.index] * 1e6
                sim_node.c_m = fake_c_m
                sim_node.r_a = fake_r_a
                sim_node.content["points_3d"] = points[comp_node.index]
        elif method == 2:
            lengths = arg1
            radii = arg2
            surfaces = 2.0 * np.pi * radii * lengths
            for ii, comp_node in enumerate(ctree):
                sim_node = self.__getitem__(comp_node.index, skip_inds=[])
                if self.is_root(sim_node):
                    sim_node.set_p3d(np.array([0.0, 0.0, 0.0]), radii[ii] * 1e4, 1)
                else:
                    sim_node.set_p3d(
                        np.array(
                            [sim_node.parent_node.xyz[0] + lengths[ii] * 1e4, 0.0, 0.0]
                        ),
                        radii[ii] * 1e4,
                        3,
                    )

            # fill the tree with the currents
            for ii, sim_node in enumerate(self):
                comp_node = ctree[ii]
                sim_node.currents = {
                    chan: [g / surfaces[comp_node.index], e]
                    for chan, (g, e) in comp_node.currents.items()
                }
                sim_node.concmechs = copy.deepcopy(comp_node.concmechs)
                for concmech in sim_node.concmechs.values():
                    concmech.gamma *= surfaces[comp_node.index] * 1e6
                sim_node.c_m = fake_c_m
                sim_node.r_a = fake_r_a
                sim_node.R = radii[comp_node.index] * 1e4  # convert to [um]
                sim_node.L = lengths[comp_node.index] * 1e4  # convert to [um]

    # redefinition of bunch of standard functions to not include skip inds by default
    def __getitem__(self, index, skip_inds=[]):
        return super().__getitem__(index, skip_inds=skip_inds)

    def get_nodes(self, recompute_flag=0, skip_inds=[]):
        return super().get_nodes(recompute_flag=recompute_flag, skip_inds=skip_inds)

    def __iter__(self, node=None, skip_inds=[]):
        return super().__iter__(node=node, skip_inds=skip_inds)

    def _find_node(self, node, index, skip_inds=[]):
        return super()._find_node(node, index, skip_inds=skip_inds)

    def _gather_nodes(self, node, node_list=[], skip_inds=[]):
        return super()._gather_nodes(node, node_list=node_list, skip_inds=skip_inds)

    def create_corresponding_node(self, node_index):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return NeuronCompartmentNode(node_index)

    def _create_neuron_tree(self, pprint):
        for node in self:
            # create the NEURON section
            compartment = node._make_section(pprint=pprint)
            # connect with parent section
            if not self.is_root(node):
                compartment.connect(self.sections[node.parent_node.index], 0.5, 0)
            # store
            self.sections.update({node.index: compartment})
            # create a static shunt
            shunt = node._make_shunt(compartment)
            if shunt is not None:
                self.shunts.append(shunt)

    def _convert_loc(self, loc_idx):
        if isinstance(loc_idx, int):
            return self.equivalent_locs[loc_idx]
        else:
            return loc_idx

    def set_rec_locs(self, loc_idxs):
        """
        Set the recording locations

        Parameters
        ----------
        loc_idxs : int, dict, tuple, or `neat.MorphLoc`
            the recording locations
        """
        rec_locs = [self._convert_loc(loc_idx) for loc_idx in loc_idxs]
        self.store_locs(rec_locs, "rec locs")

    def add_shunt(self, loc_idx, *args, **kwargs):
        """
        Adds a static conductance at a given location

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the shunt.
        g: float
            The conductance of the shunt (uS)
        e_r: float
            The reversal potential of the shunt (mV)
        """
        loc = self._convert_loc(loc_idx)
        return super().add_shunt(loc, *args, **kwargs)

    def add_double_exp_current(self, loc_idx, *args, **kwargs):
        """
        Adds a double exponential input current at a given location

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the current waveform (ms)
        tau2: float
            Decay time of the current waveform (ms)
        """
        loc = self._convert_loc(loc_idx)
        return super().add_double_exp_current(loc, *args, **kwargs)

    def add_exp_synapse(self, loc_idx, *args, **kwargs):
        """
        Adds a single-exponential conductance-based synapse

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau: float
            Decay time of the conductance window (ms)
        e_r: float
           Reversal potential of the synapse (mV)
        """
        loc = self._convert_loc(loc_idx)
        return super().add_exp_synapse(loc, *args, **kwargs)

    def add_double_exp_synapse(self, loc_idx, *args, **kwargs):
        """
        Adds a double-exponential conductance-based synapse

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the conductance window (ms)
        tau2: float
            Decay time of the conductance window (ms)
        e_r: float
            Reversal potential of the synapse (mV)
        """
        loc = self._convert_loc(loc_idx)
        return super().add_double_exp_synapse(loc, *args, **kwargs)

    def add_nmda_synapse(self, loc_idx, *args, **kwargs):
        """
        Adds a single-exponential conductance-based synapse with an AMPA and an
        NMDA component

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau: float
            Decay time of the AMPA conductance window (ms)
        tau_nmda: float
            Decay time of the NMDA conductance window (ms)
        e_r: float (optional, default ``0.`` mV)
           Reversal potential of the synapse (mV)
        nmda_ratio: float (optional, default 1.7)
            The ratio of the NMDA over AMPA component. Means that the maximum of
            the NMDA conductance window is ``nmda_ratio`` times the maximum of
            the AMPA conductance window.
        """
        loc = self._convert_loc(loc_idx)
        return super().add_nmda_synapse(loc, *args, **kwargs)

    def add_double_exp_nmda_synapse(self, loc_idx, *args, **kwargs):
        """
        Adds a double-exponential conductance-based synapse with an AMPA and an
        NMDA component

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the AMPA conductance window (ms)
        tau2: float
            Decay time of the AMPA conductance window (ms)
        tau1_nmda: float
            Rise time of the NMDA conductance window (ms)
        tau2_nmda: float
            Decay time of the NMDA conductance window (ms)
        e_r: float (optional, default ``0.`` mV)
           Reversal potential of the synapse (mV)
        nmda_ratio: float (optional, default 1.7)
            The ratio of the NMDA over AMPA component. Means that the maximum of
            the NMDA conductance window is ``nmda_ratio`` times the maximum of
            the AMPA conductance window.
        """
        loc = self._convert_loc(loc_idx)
        return super().add_double_exp_nmda_synapse(loc, *args, **kwargs)

    def add_i_clamp(self, loc_idx, *args, **kwargs):
        """
        Injects a DC current step at a given lcoation

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the current.
        amp: float
            The amplitude of the current (nA)
        delay: float
            The delay of the current step onset (ms)
        dur: float
            The duration of the current step (ms)
        """
        loc = self._convert_loc(loc_idx)
        return super().add_i_clamp(loc, *args, **kwargs)

    def add_sin_clamp(self, loc_idx, *args, **kwargs):
        """
        Injects a sinusoidal current at a given lcoation

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the current.
        amp: float
            The amplitude of the current (nA)
        delay: float
            The delay of the current onset (ms)
        dur: float
            The duration of the current (ms)
        bias: float
            Constant baseline added to the sinusoidal waveform (nA)
        freq: float
            Frequency of the sinusoid (Hz)
        phase: float
            Phase of the sinusoid (rad)
        """
        loc = self._convert_loc(loc_idx)
        return super().add_sin_clamp(loc, *args, **kwargs)

    def add_ou_clamp(self, loc_idx, *args, **kwargs):
        """
        Injects a Ornstein-Uhlenbeck current at a given lcoation

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the current.
        tau: float
            Time-scale of the OU process (ms)
        mean: float
            Mean of the OU process (nA)
        stdev: float
            Standard deviation of the OU process (nA)
        delay: float
            The delay of current onset from the start of the simulation (ms)
        dur: float
            The duration of the current input (ms)
        seed: int, optional
            Seed for the random number generator
        """
        loc = self._convert_loc(loc_idx)
        return super().add_ou_clamp(loc, *args, **kwargs)

    def add_ou_conductance(self, loc_idx, *args, **kwargs):
        """
        Injects a Ornstein-Uhlenbeck conductance at a given location

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the conductance.
        tau: float
            Time-scale of the OU process (ms)
        mean: float
            Mean of the OU process (uS)
        stdev: float
            Standard deviation of the OU process (uS)
        e_r: float
            Reversal of the current (mV)
        delay: float
            The delay of current onset from the start of the simulation (ms)
        dur: float
            The duration of the current input (ms)
        seed: int, optional
            Seed for the random number generator
        """
        loc = self._convert_loc(loc_idx)
        return super().add_ou_conductance(loc, *args, **kwargs)

    def add_ou_reversal(self, loc_idx, *args, **kwargs):
        loc = loc_idx if isinstance(loc_idx, tuple) else self.equivalent_locs[loc_idx]
        return super().add_ou_reversal(loc, *args, **kwargs)

    def add_v_clamp(self, loc_idx, *args, **kwargs):
        """
        Adds a voltage clamp at a given location

        Parameters
        ----------
        loc: int, dict, tuple or `neat.MorphLoc`
            The location of the conductance.
        e_c: float
            The clamping voltage (mV)
        dur: float, ms
            The duration of the voltage clamp
        """
        loc = self._convert_loc(loc_idx)
        return super().add_v_clamp(loc, *args, **kwargs)
