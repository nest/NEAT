# -*- coding: utf-8 -*-
#
# arbormodel.py
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
import copy
import json
import time
import platform
import warnings
import functools
import operator

import numpy as np

from ...trees.morphtree import MorphLoc
from ...trees.phystree import PhysTree, PhysNode
from ...trees.compartmenttree import CompartmentTree
from ...factorydefaults import DefaultPhysiology
from ..simtree import SimTreeMixin

try:
    import arbor
    from arbor import units as U

    _ARBOR_AVAILABLE = True
except ImportError:
    _ARBOR_AVAILABLE = False
    warnings.warn(
        "arbor is not installed. ArborSimTree and ArborCompartmentTree require "
        "arbor to export models.",
        ImportWarning,
    )


class ArborMechanismLoadError(RuntimeError):
    pass


# catalogue compiled with `neatmodels install <name> -s arbor` that is used by
# the Arbor models, set by `load_arbor_model()`
_LOADED_ARBOR_MODEL = {"name": None, "catalogue": None}


def _check_arbor_available():
    if not _ARBOR_AVAILABLE:
        raise ModuleNotFoundError(
            "arbor is not installed, install it with `pip install arbor` to "
            "simulate NEAT models with Arbor."
        )


def _get_arbor_runtime_metadata():
    return {
        "arbor_version": arbor.__version__,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }


def _validate_arbor_build_metadata(model_path):
    metadata_path = os.path.join(model_path, "build_info.json")
    if not os.path.exists(metadata_path):
        raise ArborMechanismLoadError(
            f"Arbor model '{model_path}' is missing build metadata at "
            f"'{metadata_path}'. Recompile the mechanisms with 'neatmodels install'."
        )
    with open(metadata_path, "r") as metadata_file:
        expected_metadata = json.load(metadata_file)

    mismatches = [
        f"{key}: built for '{expected_metadata.get(key)}', current runtime is '{value}'"
        for key, value in _get_arbor_runtime_metadata().items()
        if expected_metadata.get(key) != value
    ]
    if mismatches:
        raise ArborMechanismLoadError(
            f"Arbor model '{model_path}' was built for a different runtime. "
            f"Recompile the mechanisms with 'neatmodels install'. "
            f"Mismatches: {'; '.join(mismatches)}"
        )


def load_arbor_model(name):
    """
    Load the Arbor mechanism catalogue that was compiled with
    ``neatmodels install <name> -s arbor``.

    Arbor models created afterwards (:class:`ArborSimTree`,
    :class:`ArborCompartmentTree`) use the mechanisms of this catalogue. Since
    Arbor does not allow mechanisms with the same name in one catalogue, and
    NEAT catalogues all contain the default NEAT mechanisms, only one NEAT
    catalogue can be active at a time: loading a model replaces the
    previously loaded one.

    Parameters
    ----------
    name: str
        The name of the model, as given to ``neatmodels install``
    """
    _check_arbor_available()
    model_path = os.path.join(os.path.dirname(__file__), "tmp", name)
    catalogue_path = os.path.join(model_path, f"{name}-catalogue.so")
    if not os.path.exists(catalogue_path):
        raise FileNotFoundError(
            f"The Arbor model named '{name}' is not installed. "
            f"Run 'neatmodels -h' in the terminal for help on "
            f"installing new Arbor models with NEAT. "
            f"Installed models will be in "
            f"'{os.path.join(os.path.dirname(__file__), 'tmp/')}'."
        )
    if _LOADED_ARBOR_MODEL["name"] == name:
        return
    _validate_arbor_build_metadata(model_path)
    _LOADED_ARBOR_MODEL["catalogue"] = arbor.load_catalogue(catalogue_path)
    _LOADED_ARBOR_MODEL["name"] = name


def _get_arbor_cable_properties():
    """
    Arbor's global cable cell properties, with the mechanism catalogue loaded
    by :func:`load_arbor_model` and NEAT's default temperature and ion
    concentrations
    """
    if _LOADED_ARBOR_MODEL["catalogue"] is None:
        raise ArborMechanismLoadError(
            "No Arbor model has been loaded, load the mechanisms compiled with "
            "`neatmodels install <name> -s arbor` with "
            "`neat.load_arbor_model(<name>)`."
        )
    catalogue = arbor.default_catalogue()
    catalogue.extend(_LOADED_ARBOR_MODEL["catalogue"], "")

    default_phys = DefaultPhysiology()
    props = arbor.neuron_cable_properties()
    props.catalogue = catalogue
    props.set_property(tempK=(default_phys.temp + 273.15) * U.Kelvin)
    for ion, conc in default_phys.conc.items():
        props.set_ion(
            ion,
            int_con=conc * U.mM,
            ext_con=default_phys.conc_ext[ion] * U.mM,
        )
    return props


def _get_mechanism_name(key):
    """
    Name of the Arbor mechanism corresponding to a current or concentration
    mechanism key on a :class:`PhysNode`
    """
    if key == "L":
        return "Leak"
    if key == "ca":
        return "conc_" + key
    return "I" + key


def _get_n_cv_lambda_rule(node, factor_lambda):
    """
    Number of control volumes on a node according to the lambda rule
    (Carnevale & Hines, 2004), evaluated with the radius, axial resistance and
    membrane capacitance of the node

    Parameters
    ----------
    node: :class:`PhysNode`
        The node
    factor_lambda: int or float
        If int, the number of control volumes. If float, multiplies the
        number of control volumes given by the lambda rule
    """
    if not isinstance(factor_lambda, float):
        return factor_lambda
    # length constant at 100 Hz [um], cf. `lambda_f()` in NEURON's `stdlib.hoc`
    lambda_f = 1e5 * np.sqrt(
        2.0 * node.R / (4.0 * np.pi * 100.0 * node.r_a * 1e6 * node.c_m)
    )
    return int(((node.L / (0.1 * lambda_f) + 0.9) / 2.0) * 2.0 + 1.0) * int(
        factor_lambda
    )


class _ArborSingleCellRecipe(arbor.recipe if _ARBOR_AVAILABLE else object):
    def __init__(self, cell, probes, event_generators, cable_properties):
        super().__init__()
        self._cell = cell
        self._probes = probes
        self._event_generators = event_generators
        self._cable_properties = cable_properties

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self._cell

    def probes(self, gid):
        return self._probes

    def event_generators(self, gid):
        return self._event_generators

    def global_properties(self, kind):
        return self._cable_properties


class ArborSimNode(PhysNode):
    """
    Subclass of :class:`PhysNode` that implements functionality to paint its
    physiological parameters on a region of an Arbor cable cell.
    """

    def __init__(self, index, p3d=None):
        super().__init__(index, p3d)

    def _paint_membrane(self, decor, region, channel_storage):
        """
        Paint the membrane capacitance, ion channels and concentration
        mechanisms of this node

        Parameters
        ----------
        decor: `arbor.decor`
            The decor of the Arbor cable cell
        region: str
            Arbor region expression on which the parameters are painted
        channel_storage: dict
            The ion channels of the tree
        """
        decor.paint(region, cm=self.c_m * 1e-2 * U.F / U.m2)  # uF/cm^2 --> F/m^2
        for key, (g_max, e_rev) in self.currents.items():
            if g_max <= 1e-10:
                continue
            params = {"g": g_max * 1e-6}  # uS/cm^2 --> S/cm^2
            if key == "L" or channel_storage[key]._uses_e_rev:
                params["e"] = e_rev
            decor.paint(region, arbor.density(_get_mechanism_name(key), params))
        for ion, concmech in self.concmechs.items():
            params = dict(concmech.items())
            params["gamma"] *= 1e6
            decor.paint(region, arbor.density(_get_mechanism_name(ion), params))


class ArborSimTree(SimTreeMixin, PhysTree):
    """
    Tree class to simulate morphological neuron models with the Arbor simulator
    (Abi Akar et al., 2019), based on :class:`PhysTree`.

    The API mirrors that of :class:`NeuronSimTree`. Each node is simulated as a
    cylinder (the soma as a cylinder with the surface area of the spherical
    soma), and the ion channels and synapses are simulated with the mechanisms
    of the catalogue loaded with :func:`load_arbor_model`.

    Since Arbor cable cells can not be modified once they are created, point
    processes added with the `add_*()` functions are stored as placements on
    an `arbor.decor` and only instantiated when the model is simulated with
    :func:`ArborSimTree.run`.

    Attributes
    ----------
    shunts, syns, iclamps, vclamps: list of dict
        Storage containers for the point processes that have been added to the
        model. Each entry describes how the current through the point process
        can be recorded. The indices of `syns` are the synapse indices to use
        with :func:`ArborSimTree.set_spiketrain`.
    dt: float
        timestep of the simulator ``[ms]``
    t_calibrate: float
        Time for the model to equilibrate``[ms]``. Not counted as part of the
        simulation.
    factor_lambda : int or float
        If int, the number of control volumes per node. If float, multiplies the
        number of control volumes given by the standard lambda rule (Carnevale,
        2004) to give the number of control volumes simulated (default value 1.
        gives the number given by the lambda rule)
    v_init: float
        The initial voltage at which the model is initialized ``[mV]``

    Notes
    -----
    All point process currents recorded by :func:`ArborSimTree.run`
    (``'i_syn'``, ``'i_clamp'`` and ``'i_vclamp'``) are positive when they
    depolarize the cell, and are evaluated from the model state at the recording
    time. :class:`NeuronSimTree` records the currents of the time step that ends
    at the recording time, and records the electrode currents of its step,
    sinusoidal and voltage clamps (added with `add_i_clamp()`, `add_sin_clamp()`
    and `add_v_clamp()`) with the opposite sign.
    """

    def __init__(self, arg=None, types=[1, 3, 4]):
        self._reset_model_storage()
        # simulation parameters
        self.dt = 0.1  # ms
        self.t_calibrate = 0.0  # ms
        self.factor_lambda = 1.0
        self.v_init = -75.0  # mV
        self.indstart = 0
        # initialize the tree structure
        super().__init__(arg=arg, types=types)

    def _reset_model_storage(self):
        self.segment_tree = None
        self.decor = None
        self.cv_policy = None
        self.segments = {}
        self.shunts = []
        self.syns = []
        self.iclamps = []
        self.vclamps = []
        self.event_generators = []
        self.ou_seeds = []
        self._n_point_processes = 0

    def create_corresponding_node(self, node_index, p3d=None):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return ArborSimNode(node_index, p3d=p3d)

    def set_simulation_parameters(
        self, factor_lambda=1.0, t_calibrate=0.0, dt=0.025, v_init=-75.0
    ):
        """
        Set the simulation parameters

        Parameters
        ----------
        factor_lambda: float or int
            If int, the number of control volumes per node. If float, multiplies
            the number of control volumes given by the lambda rule (Carnevale,
            2004). Defaults to 1.
        t_calibrate: float
            The calibration time ``[ms]``. Defaults to 0.
        dt: float
            Timestep of the simulation ``[ms]``. Defaults to 0.025.
        v_init: float
            The initial voltage ``[mV]``. Defaults to -75.
        """
        self.dt = dt  # ms
        self.t_calibrate = t_calibrate  # ms
        self.indstart = int(t_calibrate / dt)
        self.factor_lambda = factor_lambda
        self.v_init = v_init  # mV

    def init_model(
        self, dt=0.025, t_calibrate=0.0, v_init=-75.0, factor_lambda=1.0, pprint=False
    ):
        """
        Initialize the Arbor morphology, decor and discretization to simulate the
        neuron model implemented by this tree.

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
            If int, the number of control volumes per node. If float, multiplies
            the number of control volumes given by the standard lambda rule
            (Carnevale, 2004) to give the number of control volumes simulated
            (default value 1. gives the number given by the lambda rule)
        pprint: bool (default ``False``)
            Whether or not to print info on the Arbor model's creation
        """
        _check_arbor_available()
        self.set_simulation_parameters(
            dt=dt,
            t_calibrate=t_calibrate,
            v_init=v_init,
            factor_lambda=factor_lambda,
        )
        self.delete_model()
        self._create_arbor_tree(pprint=pprint)

    def delete_model(self):
        """
        Delete the Arbor model and all point processes that were added to it
        """
        self._reset_model_storage()
        self.store_locs([(self.root.index, 0.5)], "rec locs")

    def _create_arbor_tree(self, pprint):
        self.segment_tree = arbor.segment_tree()
        self.decor = arbor.decor()
        policies = []
        distal_points = {}
        for node in self:
            if self.is_root(node):
                # cylinder with the surface of the spherical soma, children
                # are attached to its distal end, as in `NeuronSimTree`
                prox = node.xyz - np.array([node.R, 0.0, 0.0])
                dist = node.xyz + np.array([node.R, 0.0, 0.0])
                parent_segment = arbor.mnpos
                n_cv = 1
            else:
                prox = distal_points[node.parent_node.index]
                direction = node.xyz - node.parent_node.xyz
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 0.0 else np.array([1, 0, 0])
                dist = prox + node.L * direction
                parent_segment = self.segments[node.parent_node.index]
                n_cv = _get_n_cv_lambda_rule(node, self.factor_lambda)
            distal_points[node.index] = dist
            segment = self.segment_tree.append(
                parent_segment,
                arbor.mpoint(*prox, node.R),
                arbor.mpoint(*dist, node.R),
                tag=node.swc_type,
            )
            self.segments[node.index] = segment

            region = f"(segment {segment})"
            self.decor.paint(region, rL=node.r_a * 1e6 * U.Ohm * U.cm)
            node._paint_membrane(self.decor, region, self.channel_storage)
            if node.g_shunt > 1e-10:
                self._add_shunt_at_locset(
                    f"(distal (segment {segment}))", node.g_shunt, node.v_ep
                )
            policies.append(arbor.cv_policy_fixed_per_branch(n_cv, region))

            if pprint:
                print(node)
                print(f">>> segment length = {np.linalg.norm(dist - prox):.2f} um")
                print(f">>> segment radius = {node.R:.2f} um")
                print(f">>> number of control volumes = {n_cv}")

        self.cv_policy = functools.reduce(operator.add, policies)

    def _get_locset(self, loc, avoid_endpoints=False):
        """
        Arbor locset expression of a location

        Parameters
        ----------
        loc: :class:`MorphLoc`
            The location
        avoid_endpoints: bool
            If ``True``, locations at the ends of a node are moved slightly
            inwards, so that they do not fall in the zero-area control volume
            at a fork point, where no mechanisms are present. Defaults to
            ``False``.
        """
        x = loc["x"]
        if avoid_endpoints:
            x = min(max(x, 1e-3), 1.0 - 1e-3)
        return f"(on-components {x} (segment {self.segments[loc['node']]}))"

    def _add_point_process(self, loc, mechanism, params):
        loc = MorphLoc(loc, self)
        label = f"pp{self._n_point_processes}"
        self._n_point_processes += 1
        self.decor.place(self._get_locset(loc), arbor.synapse(mechanism, params), label)
        return {"label": label, "mechanism": mechanism}

    def _add_event(self, label, weight, times):
        schedule = arbor.explicit_schedule([max(t, 0.0) * U.ms for t in times])
        self.event_generators.append(arbor.event_generator(label, weight, schedule))

    def _add_shunt_at_locset(self, locset, g, e_r):
        label = f"pp{self._n_point_processes}"
        self._n_point_processes += 1
        self.decor.place(locset, arbor.synapse("Shunt", g=g, e=e_r), label)
        self.shunts.append({"label": label, "mechanism": "Shunt"})

    def set_rec_locs(self, locs):
        """
        Set the recording locations

        Parameters
        ----------
        locs : list of int, dict, tuple, or :class:`MorphLoc`
            the recording locations
        """
        self.store_locs(locs, "rec locs")

    def add_shunt(self, loc, g, e_r):
        """
        Adds a static conductance at a given location

        Parameters
        ----------
        loc: int, dict, tuple or :class:`MorphLoc`
            The location of the shunt.
        g: float
            The conductance of the shunt (uS)
        e_r: float
            The reversal potential of the shunt (mV)
        """
        self.shunts.append(self._add_point_process(loc, "Shunt", {"g": g, "e": e_r}))

    def add_double_exp_current(self, loc, tau1, tau2):
        """
        Adds a double exponential input current at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the current waveform (ms)
        tau2: float
            Decay time of the current waveform (ms)
        """
        self.syns.append(
            self._add_point_process(
                loc, "epsc_double_exp", {"tau1": tau1, "tau2": tau2}
            )
        )

    def add_exp_synapse(self, loc, tau, e_r):
        """
        Adds a single-exponential conductance-based synapse

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
            The location of the current.
        tau: float
            Decay time of the conductance window (ms)
        e_r: float
           Reversal potential of the synapse (mV)
        """
        self.syns.append(
            self._add_point_process(loc, "exp_AMPA_NMDA", {"tau": tau, "e": e_r})
        )

    def add_double_exp_synapse(self, loc, tau1, tau2, e_r):
        """
        Adds a double-exponential conductance-based synapse

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
            The location of the current.
        tau1: float
            Rise time of the conductance window (ms)
        tau2: float
            Decay time of the conductance window (ms)
        e_r: float
            Reversal potential of the synapse (mV)
        """
        self.syns.append(
            self._add_point_process(
                loc, "Exp2Syn", {"tau1": tau1, "tau2": tau2, "e": e_r}
            )
        )

    def add_nmda_synapse(self, loc, tau, tau_nmda, e_r=0.0, nmda_ratio=1.7):
        """
        Adds a single-exponential conductance-based synapse with an AMPA and an
        NMDA component

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
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
        params = {"tau": tau, "tau_NMDA": tau_nmda, "e": e_r, "NMDA_ratio": nmda_ratio}
        self.syns.append(self._add_point_process(loc, "exp_AMPA_NMDA", params))

    def add_double_exp_nmda_synapse(
        self, loc, tau1, tau2, tau1_nmda, tau2_nmda, e_r=0.0, nmda_ratio=1.7
    ):
        """
        Adds a double-exponential conductance-based synapse with an AMPA and an
        NMDA component

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
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
        params = {
            "tau1": tau1,
            "tau2": tau2,
            "tau1_NMDA": tau1_nmda,
            "tau2_NMDA": tau2_nmda,
            "e": e_r,
            "NMDA_ratio": nmda_ratio,
        }
        self.syns.append(self._add_point_process(loc, "double_exp_AMPA_NMDA", params))

    def add_i_clamp(self, loc, amp, delay, dur):
        """
        Injects a DC current step at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
            The location of the current.
        amp: float
            The amplitude of the current (nA)
        delay: float
            The delay of the current step onset (ms)
        dur: float
            The duration of the current step (ms)
        """
        self.iclamps.append(
            self._add_switched_point_process(loc, "IClamp", {"amp": amp}, delay, dur)
        )

    def add_sin_clamp(self, loc, amp, delay, dur, bias, freq, phase):
        """
        Injects a sinusoidal current at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
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
        params = {"pkamp": amp, "freq": freq, "phase": phase, "bias": bias}
        self.iclamps.append(
            self._add_switched_point_process(loc, "SinClamp", params, delay, dur)
        )

    def _add_switched_point_process(self, loc, mechanism, params, delay, dur):
        """
        Adds a point process that is switched on by an event with positive
        weight and off by an event with negative weight, as Arbor mechanisms
        have no access to the simulation time.
        """
        point_process = self._add_point_process(loc, mechanism, params)
        # Arbor delivers events at the start of the time step in which they
        # fall, shifting them by half a time step ensures that rounding errors
        # in the event times do not shift them to the preceding time step
        t_onset = delay + self.t_calibrate + self.dt / 2.0
        self._add_event(point_process["label"], 1.0, [t_onset])
        self._add_event(point_process["label"], -1.0, [t_onset + dur])
        return point_process

    def _add_ou_process(self, loc, mechanism, params, delay, dur, seed):
        self.ou_seeds.append(np.random.randint(1e16) if seed is None else seed)
        return self._add_switched_point_process(loc, mechanism, params, delay, dur)

    def add_ou_clamp(self, loc, tau, mean, stdev, delay, dur, seed=None):
        """
        Injects a Ornstein-Uhlenbeck current at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
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
            Seed for the random number generator. Note that Arbor draws all
            noise from a single random stream per simulation, whose seed is
            derived from the seeds of all OU processes in the model.
        """
        params = {"tau": tau, "mean": mean, "stdev": stdev}
        self.iclamps.append(
            self._add_ou_process(loc, "OUClamp", params, delay, dur, seed)
        )

    def add_ou_conductance(self, loc, tau, mean, stdev, e_r, delay, dur, seed=None):
        """
        Injects a Ornstein-Uhlenbeck conductance at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
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
            Seed for the random number generator, see
            :func:`ArborSimTree.add_ou_clamp`
        """
        params = {"tau": tau, "mean": mean, "stdev": stdev, "e": e_r}
        self.iclamps.append(
            self._add_ou_process(loc, "OUConductance", params, delay, dur, seed)
        )

    def add_ou_reversal(self, loc, tau, mean, stdev, g_val, delay, dur, seed=None):
        """
        Adds a static conductance with an Ornstein-Uhlenbeck reversal potential
        at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
            The location of the conductance.
        tau: float
            Time-scale of the OU process (ms)
        mean: float
            Mean of the OU process (mV)
        stdev: float
            Standard deviation of the OU process (mV)
        g_val: float
            The conductance (uS)
        delay: float
            The delay of current onset from the start of the simulation (ms)
        dur: float
            The duration of the current input (ms)
        seed: int, optional
            Seed for the random number generator, see
            :func:`ArborSimTree.add_ou_clamp`
        """
        params = {"tau": tau, "mean": mean, "stdev": stdev, "g": g_val}
        self.iclamps.append(
            self._add_ou_process(loc, "OUReversal", params, delay, dur, seed)
        )

    def add_v_clamp(self, loc, e_c, dur):
        """
        Adds a voltage clamp at a given location

        Parameters
        ----------
        loc: dict, tuple or :class:`MorphLoc`
            The location of the conductance.
        e_c: float
            The clamping voltage (mV)
        dur: float, ms
            The duration of the voltage clamp, starting at the beginning of the
            simulation (including the calibration time)
        """
        vclamp = self._add_point_process(loc, "SEClamp", {"rs": 0.01, "amp1": e_c})
        # shifted by half a time step, see `_add_switched_point_process()`
        self._add_event(vclamp["label"], 1.0, [dur + self.dt / 2.0])
        self.vclamps.append(vclamp)

    def set_spiketrain(self, syn_index, syn_weight, spike_times):
        """
        Let a synapse receive a specific spike train.

        Synapses are stored under `self.syns`, which contains the synapses added
        with:
        - `self.add_double_exp_current()`
        - `self.add_exp_synapse()`
        - `self.add_double_exp_synapse()`
        - `self.add_nmda_synapse()`
        - `self.add_double_exp_nmda_synapse()`

        Parameters
        ----------
        syn_index: int
            index of the point process in the synapse stack
        syn_weight: float
            weight of the synapse (maximal value of the conductance window)
        spike_times: list or `np.array` of floats
            the spike times
        """
        # as in `NeuronSimTree`, spikes reach the synapse after a delay of one
        # time step, the additional half time step ensures that rounding errors
        # do not shift the events to the preceding time step (see
        # `_add_switched_point_process()`)
        spks = np.array(spike_times) + self.t_calibrate + 1.5 * self.dt
        self._add_event(self.syns[syn_index]["label"], syn_weight, spks)

    def _get_simulation_seed(self):
        """
        Arbor has a single seed per simulation, which we derive from the seeds of
        the individual OU processes
        """
        if len(self.ou_seeds) == 0:
            return 0
        return int(
            np.random.SeedSequence([int(s) for s in self.ou_seeds]).generate_state(1)[0]
        )

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
        Run the Arbor simulation. Records at all locations stored
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
            Record ion current densities ``[mA/cm^2]`` at locations stored under
            'rec locs'. Accessible as `np.ndarray` in the output dict with as key
            ``'i'`` + the ion's name
        spike_rec_loc: dict, tuple or :class:`MorphLoc` (default ``None``)
            Record the output spike times from this location
        spike_rec_thr: float
            Spike threshold
        pprint: bool (default ``False``)
            Whether or not to print info on the Arbor simulation

        Returns
        -------
        dict
            Dictionary with the results of the simulation. Contains time and
            voltage as `np.ndarray` at locations stored under the name '
            rec locs', respectively with keys 't' and 'v_m'. Also contains
            traces of other recorded variables if the option to record them was
            set to ``True``
        """
        if self.decor is None:
            raise RuntimeError(
                "The Arbor model has not been initialized, call `init_model()` "
                "before `run()`"
            )
        if not (isinstance(downsample, int) and downsample > 0):
            raise ValueError(
                f"`downsample` should be a positive integer, got {downsample}"
            )
        if dt_rec is None:
            dt_rec = self.dt
        indstart = int(self.t_calibrate / dt_rec)
        t_total = t_max + self.t_calibrate
        rec_locs = self.get_locs("rec locs")

        # probes are identified by their tag, `traces` maps the result keys to
        # the tags of the probes that are recorded for them, and the factors
        # with which the recorded values are multiplied
        probes = []
        traces = {}

        def add_probe(probe_func, *args, factor=1.0, default=0.0):
            tag = f"probe{len(probes)}"
            probes.append(probe_func(*args, tag))
            return (tag, factor, default)

        def add_point_process_probe(point_process):
            # point process currents are in membrane convention, we flip the
            # sign so that depolarizing currents are positive
            return add_probe(
                arbor.cable_probe_point_state,
                point_process["label"],
                point_process["mechanism"],
                "i_rec",
                factor=-1.0,
            )

        traces["v_m"] = [
            add_probe(arbor.cable_probe_membrane_voltage, self._get_locset(loc))
            for loc in rec_locs
        ]
        if record_from_syns:
            traces["i_syn"] = [add_point_process_probe(syn) for syn in self.syns]
        if record_from_iclamps:
            traces["i_clamp"] = [add_point_process_probe(ic) for ic in self.iclamps]
        if record_from_vclamps:
            traces["i_vclamp"] = [add_point_process_probe(vc) for vc in self.vclamps]
        if record_from_channels:
            channel_names = self.get_channels_in_tree()
            chan_traces = {
                channel_name: {
                    str(var): [
                        add_probe(
                            arbor.cable_probe_density_state,
                            self._get_locset(loc, avoid_endpoints=True),
                            _get_mechanism_name(channel_name),
                            str(var),
                        )
                        for loc in rec_locs
                    ]
                    for var in self.channel_storage[channel_name].statevars
                }
                for channel_name in channel_names
            }
        # Arbor only tracks concentrations where they are used by a mechanism,
        # elsewhere the concentration is the default concentration
        default_conc = DefaultPhysiology().conc
        for ion in record_concentrations:
            traces[ion] = [
                add_probe(
                    arbor.cable_probe_ion_int_concentration,
                    self._get_locset(loc),
                    ion,
                    default=default_conc.get(ion, 0.0),
                )
                for loc in rec_locs
            ]
        for ion in record_currents:
            traces[f"i{ion}"] = [
                add_probe(
                    arbor.cable_probe_ion_current_density,
                    self._get_locset(loc, avoid_endpoints=True),
                    ion,
                    factor=0.1,  # A/m^2 --> mA/cm^2
                )
                for loc in rec_locs
            ]

        decor = arbor.decor(self.decor)
        decor.set_property(Vm=self.v_init * U.mV)
        if spike_rec_loc is not None:
            decor.place(
                self._get_locset(MorphLoc(spike_rec_loc, self)),
                arbor.threshold_detector(spike_rec_thr * U.mV),
                "spike_detector",
            )

        cell = arbor.cable_cell(
            self.segment_tree, decor, arbor.label_dict(), self.cv_policy
        )
        recipe = _ArborSingleCellRecipe(
            cell, probes, self.event_generators, _get_arbor_cable_properties()
        )
        sim = arbor.simulation(recipe, seed=self._get_simulation_seed())
        if spike_rec_loc is not None:
            # Arbor only exchanges spikes from sources with outgoing
            # connections, so spikes of an unconnected cell are only recorded
            # locally
            sim.record(arbor.spike_recording.local)
        # as in NEURON, the final time `t_total` is not recorded
        t_stop = t_total - dt_rec / 2.0
        schedule = arbor.regular_schedule(0.0 * U.ms, dt_rec * U.ms, t_stop * U.ms)
        handles = {
            f"probe{ii}": sim.sample((0, f"probe{ii}"), schedule)
            for ii in range(len(probes))
        }

        if pprint:
            print(">>> Simulating the Arbor model for " + str(t_max) + " ms. <<<")
        start = time.perf_counter()
        sim.run(t_total * U.ms, self.dt * U.ms)
        stop = time.perf_counter()
        runtime = stop - start
        if pprint:
            print(">>> Elapsed time: " + str(runtime) + " seconds. <<<")

        samples = {tag: sim.samples(handle) for tag, handle in handles.items()}
        t_arr = np.array(schedule.events(0.0, t_stop))

        def get_trace(tag, factor, default):
            # an empty list of samples signifies that the probed quantity does
            # not exist at the probed location
            if len(samples[tag]) == 0:
                return np.full_like(t_arr, default)
            return factor * samples[tag][0][0][:, 1]

        def assemble(trace_specs):
            arr = np.array([get_trace(*spec) for spec in trace_specs])
            return arr[:, indstart:][:, ::downsample] if arr.ndim == 2 else arr

        res = {"t": t_arr[indstart:][::downsample] - self.t_calibrate}
        for key, trace_specs in traces.items():
            res[key] = assemble(trace_specs)
        if record_v_deriv:
            v_full = np.array([get_trace(*spec) for spec in traces["v_m"]])
            res["dv_dt"] = np.gradient(v_full, dt_rec, axis=1)[:, indstart:][
                :, ::downsample
            ]

        if record_from_channels:
            res["chan"] = {}
            for channel_name in channel_names:
                channel = self.channel_storage[channel_name]
                res["chan"][channel_name] = {
                    var: assemble(trace_specs)
                    for var, trace_specs in chan_traces[channel_name].items()
                }
                sv = {
                    str(var): res["chan"][channel_name][str(var)]
                    for var in channel.statevars
                }
                res["chan"][channel_name]["p_open"] = channel.compute_p_open(
                    res["v_m"], **sv
                )

        if spike_rec_loc is not None:
            res["spikes"] = (
                np.array([spike["time"] for spike in sim.spikes()]) - self.t_calibrate
            )

        res["runtime"] = runtime

        return res


class ArborCompartmentTree(ArborSimTree):
    """
    Creates a :class:`ArborSimTree` to simulate reduced compartment models from
    a :class:`CompartmentTree`.

    Parameters
    ----------
    ctree: :class:`CompartmentTree`
        The tree containing the parameters of the reduced compartmental model
        to be simulated

    Attributes
    ----------
    equivalent_locs: list of tuples
        'Fake' locations corresponding to each compartment, which are
        used to insert point process at the compartments using
        the same functions definitions as for as for a morphological
        :class:`ArborSimTree`.

    Notes
    -----
    Arbor does not expose the parameters of individual control volumes (CVs),
    but derives them from the morphology and the discretization. To make the
    resulting model independent of how Arbor implements the discretization,
    each compartment is constructed out of two cylindrical segments:

    - a "membrane" segment, that carries the membrane capacitance, ion
      channels and point processes of the compartment, and has an axial
      resistance that is negligible compared to the coupling resistances,
    - a "coupling" segment, that connects the membrane segment to the
      membrane segment of the parent compartment, and has a negligible surface
      area, and an axial resistance equal to the coupling resistance with the
      parent compartment.

    The CV boundaries are placed at the proximal ends of the membrane segments,
    so that each CV comprises the membrane segment of one compartment and the
    coupling segments of its children. Since the coupling resistance between
    two CVs is computed from the axial resistance between reference points
    inside both CVs, and all the axial resistance between these points lies in
    the coupling segment, the coupling conductance equals the conductance of
    the coupling segment, irrespective of where exactly Arbor places these
    reference points. That every compartment results in a single CV is verified
    when the model is simulated.

    - Note that this class inherits from :class:`ArborSimTree` and *not* from
      :class:`CompartmentTree`, as is the case for
      :class:`NeuronCompartmentTree`, so that it can reuse the functionality
      of :class:`ArborSimTree`.
    - Locations to this class can be provided either as fake morphology
      locations -- i.e. a tuple `(node.index, x-location in [0,1])` -- where
      the value of the x-location is ignored since the nodes here are single
      compartments, or as location indices, where the index corresponds to the
      location in the original list of locations from which the
      :class:`CompartmentTree` was derived.
    """

    # geometry [um] of the membrane and coupling segments, the radius of the
    # coupling segment results in a relative surface area of 1e-8, and its
    # length ensures that the midpoint of a CV comprising a membrane segment and
    # a single coupling segment lies inside the membrane segment
    R_MEMBRANE = 10.0
    L_MEMBRANE = 10.0
    R_COUPLING = 1e-4
    L_COUPLING = 1e-2
    # resistance of the membrane segments relative to the smallest coupling
    # resistance
    REL_R_MEMBRANE = 1e-9

    def __init__(self, ctree):
        if not issubclass(ctree.__class__, CompartmentTree):
            raise ValueError(
                "`neat.ArborCompartmentTree` can only be instantiated "
                f"from a `neat.CompartmentTree` or derived class, got {ctree}"
            )
        super().__init__(ctree, types=[1, 3, 4])
        self.equivalent_locs = ctree.get_equivalent_locs()
        self._create_reduced_arbor_model(ctree)

    def _create_reduced_arbor_model(self, ctree):
        area = 2.0 * np.pi * self.R_MEMBRANE * self.L_MEMBRANE * 1e-8  # [cm^2]
        g_cs = [cn.g_c for cn in ctree if cn.parent_node is not None]
        # axial resistivity [MOhm*cm] for which the membrane segment has the
        # relative resistance `REL_R_MEMBRANE`
        self.r_a_membrane = (
            self.REL_R_MEMBRANE
            / max(g_cs, default=1.0)
            * np.pi
            * self.R_MEMBRANE**2
            / self.L_MEMBRANE
            * 1e-4
        )
        for comp_node in ctree:
            sim_node = self.__getitem__(comp_node.index, skip_inds=[])
            if self.is_root(sim_node):
                sim_node.set_p3d(np.array([0.0, 0.0, 0.0]), self.R_MEMBRANE, 1)
            else:
                xyz = sim_node.parent_node.xyz + np.array(
                    [self.L_COUPLING + self.L_MEMBRANE, 0.0, 0.0]
                )
                sim_node.set_p3d(xyz, self.R_MEMBRANE, 3)

            sim_node.currents = {
                chan: [g / area, e] for chan, (g, e) in comp_node.currents.items()
            }
            sim_node.concmechs = copy.deepcopy(comp_node.concmechs)
            for concmech in sim_node.concmechs.values():
                concmech.gamma *= area
            sim_node.c_m = comp_node.ca / area
            # axial resistivity [MOhm*cm] of the coupling segment that yields
            # the coupling conductance `g_c` [uS]
            sim_node.r_a = (
                np.pi * self.R_COUPLING**2 / (self.L_COUPLING * comp_node.g_c) * 1e-4
                if sim_node.parent_node is not None
                else self.r_a_membrane
            )

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
        return ArborSimNode(node_index)

    def _create_arbor_tree(self, pprint):
        self.segment_tree = arbor.segment_tree()
        self.decor = arbor.decor()
        boundaries = []
        for node in self:
            x0 = node.xyz[0] - self.L_MEMBRANE
            if self.is_root(node):
                parent_segment = arbor.mnpos
            else:
                coupling_segment = self.segment_tree.append(
                    self.segments[node.parent_node.index],
                    arbor.mpoint(x0 - self.L_COUPLING, 0.0, 0.0, self.R_COUPLING),
                    arbor.mpoint(x0, 0.0, 0.0, self.R_COUPLING),
                    tag=node.swc_type,
                )
                self.decor.paint(
                    f"(segment {coupling_segment})",
                    rL=node.r_a * 1e6 * U.Ohm * U.cm,
                )
                parent_segment = coupling_segment
            segment = self.segment_tree.append(
                parent_segment,
                arbor.mpoint(x0, 0.0, 0.0, self.R_MEMBRANE),
                arbor.mpoint(node.xyz[0], 0.0, 0.0, self.R_MEMBRANE),
                tag=node.swc_type,
            )
            self.segments[node.index] = segment

            region = f"(segment {segment})"
            self.decor.paint(region, rL=self.r_a_membrane * 1e6 * U.Ohm * U.cm)
            node._paint_membrane(self.decor, region, self.channel_storage)
            if node.g_shunt > 1e-10:
                self._add_shunt_at_locset(
                    f"(on-components 0.5 {region})", node.g_shunt, node.v_ep
                )
            boundaries.append(f"(proximal {region})")

            if pprint:
                print(node)

        boundary_locset = (
            f"(join {' '.join(boundaries)})" if len(boundaries) > 1 else boundaries[0]
        )
        self.cv_policy = arbor.cv_policy_explicit(boundary_locset)

    def _check_discretization(self, cell):
        """
        Verify that every compartment is simulated as a single Arbor CV
        """
        cv_data = arbor.cv_data(cell)
        nodes = list(self)
        if cv_data.num_cv != len(nodes):
            raise RuntimeError(
                f"Arbor discretized the {len(nodes)} compartment model into "
                f"{cv_data.num_cv} CVs, the compartment model can not be "
                f"simulated with this Arbor version ({arbor.__version__})"
            )

    def run(self, t_max, spike_rec_loc=None, **kwargs):
        if self.decor is not None:
            cell = arbor.cable_cell(
                self.segment_tree, self.decor, arbor.label_dict(), self.cv_policy
            )
            self._check_discretization(cell)
        if spike_rec_loc is not None:
            spike_rec_loc = self._convert_loc(spike_rec_loc)
        return super().run(t_max, spike_rec_loc=spike_rec_loc, **kwargs)

    run.__doc__ = ArborSimTree.run.__doc__

    def _get_locset(self, loc, avoid_endpoints=False):
        # the x-coordinate is irrelevant, as the compartments are isopotential
        return f"(on-components 0.5 (segment {self.segments[loc['node']]}))"

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
        loc_idxs : list of int, dict, tuple, or :class:`MorphLoc`
            the recording locations
        """
        rec_locs = [self._convert_loc(loc_idx) for loc_idx in loc_idxs]
        self.store_locs(rec_locs, "rec locs")

    def _add_point_process(self, loc_idx, mechanism, params):
        return super()._add_point_process(self._convert_loc(loc_idx), mechanism, params)
