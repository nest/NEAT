# -*- coding: utf-8 -*-
#
# phystree.py
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

import numpy as np

import copy
import warnings

from . import morphtree
from .morphtree import MorphNode, MorphTree, MorphLoc
from .morphtree import computational_tree_decorator
from ..channels import concmechs, ionchannels
from ..factorydefaults import DefaultPhysiology

CFG = DefaultPhysiology()


def comptree_removal_decorator(fun):
    """
    Decorator that provides the safety that the computational tree is removed
    when a function changes the physiological parameters of a tree
    """

    # wrapper to access self
    def wrapped(self, *args, **kwargs):
        with self.as_original_tree:
            res = fun(self, *args, **kwargs)
        self._computational_root = None
        return res

    wrapped.__doc__ = fun.__doc__
    return wrapped


class PhysNode(MorphNode):
    """
    Node associated with `neat.PhysTree`. Stores the physiological parameters
    of the cylindrical segment connecting this node with its parent node

    Attributes
    ----------
    currents: dict {str: [float,float]}
        dict with as keys the channel names and as values lists of length two
        containing as first entry the channels' conductance density (uS/cm^2)
        and as second element the channels reversal (mV) (i.e.:
        {name: [g_max (uS/cm^2), e_rev (mV)]})
        For the leak conductance, the corresponding key is 'L'
    concmechs: dict
        dict containing concentration mechanisms present in the segment
    c_m: float
        The sement's specific membrane capacitance (uF/cm^2)
    r_a: float
        The segment's axial resistance (MOhm*cm)
    g_shunt: float
        Point-like shunt conductance located at x=1 (uS)
    e_eq: float
        Segment's equilibrium potential
    """

    def __init__(
        self, index, p3d=None, c_m=1.0, r_a=100 * 1e-6, g_shunt=0.0, v_ep=-75.0
    ):
        super().__init__(index, p3d)
        # biophysical parameters
        self.currents = {}  # {name: (g_max (uS/cm^2), e_rev (mV))}
        self.concmechs = {}
        self.c_m = c_m  # uF/cm^2
        self.r_a = r_a  # MOhm*cm
        self.g_shunt = g_shunt  # uS
        # expansion points
        self.v_ep = v_ep  # mV
        self.conc_eps = {}  # equilibrium concentration values (mM)

    def set_physiology(self, c_m, r_a, g_shunt=0.0):
        """
        Set the physiological parameters of the current

        Parameters
        ----------
        c_m: float
            the membrance capacitance (uF/cm^2)
        r_a: float
            the axial current (MOhm*cm)
        g_shunt: float
            A point-like shunt, located at x=1 on the node. Defaults to 0.
        """
        self.c_m = c_m  # uF/cm^2
        self.r_a = r_a  # MOhm*cm
        self.g_shunt = g_shunt

    def _add_current(self, channel_name, g_max, e_rev):
        """
        Add an ion channel current at this node. ('L' as `channel_name`
        signifies the leak current)

        Parameters
        ----------
        channel_name: string
            the name of the current
        g_max: float
            the conductance of the current (uS/cm^2)
        e_rev: float
            the reversal potential of the current (mV)
        """
        self.currents[channel_name] = [g_max, e_rev]

    def _construct_conc_args(self, channel):
        """
        Returns the concentration expansion point for the channel, around which
        the conductance is computed.

        Checks if the ion concentration is in
        `self.conc_eps`, and otherwise defaults to the factory default in
        `neat.factorydefaults.DefaultPhysiology`.

        Parameters
        ----------
        channel: `neat.IonChannel` object
            the ion channel

        Returns
        -------
        conc: dict ({str: np.ndarray})
            The concentrations at the expansion points.
        """
        # if concencentration is in expansion point, use it. Otherwise use
        # concentration in equilibrium concentrations (self.conc_eps), if
        # it is there. If not, use default concentration.
        ions = [
            str(ion) for ion in channel.conc
        ]  # convert potential sympy symbols to str
        conc = {ion: self.conc_eps.copy().pop(ion, CFG.conc[ion]) for ion in ions}

        return conc

    def add_conc_mech(self, ion, params={}):
        """
        Add a concentration mechanism at this node.

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        params: dict
            parameters for the concentration mechanism (only used for NEURON model)
        """
        if set(params.keys()) == {"gamma", "tau"}:
            self.concmechs[ion] = concmechs.ExpConcMech(
                ion, params["tau"], params["gamma"]
            )
        else:
            warnings.warn(
                "These parameters do not match any NEAT concentration "
                + "mechanism, no concentration mechanism has been added",
                UserWarning,
            )

    def set_v_ep(self, v_ep):
        """
        Set the equilibrium potential at the node.

        Parameters
        ----------
        e_eq: float
            the equilibrium potential (mV)
        """
        self.v_ep = v_ep

    def set_conc_ep(self, ion, conc):
        """
        Set the equilibrium concentration value at this node

        Parameters
        ----------
        ion: str ('ca', 'k', 'na')
            the ion for which the concentration is to be set
        conc: float
            the concentration value (mM)
        """
        self.conc_eps[ion] = conc

    def fit_leak_current(self, channel_storage, e_eq_target=-75.0, tau_m_target=10.0):
        """ """
        gsum = 0.0
        i_eq = 0.0

        for channel_name in set(self.currents.keys()) - set("L"):
            g, e = self.currents[channel_name]

            # compute channel conductance and current
            p_open = channel_storage[channel_name].compute_p_open(e_eq_target)
            g_chan = g * p_open

            gsum += g_chan
            i_eq += g_chan * (e - e_eq_target)

        if self.c_m / (tau_m_target * 1e-3) < gsum:
            warnings.warn(
                "Membrane time scale is chosen larger than "
                + "possible, adding small leak conductance"
            )
            tau_m_target = self.c_m / (gsum + 20.0)
        else:
            tau_m_target *= 1e-3
        g_l = self.c_m / tau_m_target - gsum
        e_l = e_eq_target - i_eq / g_l
        self.currents["L"] = [g_l, e_l]
        self.v_ep = e_eq_target

    def calc_g_tot(self, channel_storage, channel_names=None, v=None):
        """
        Get the total conductance of the membrane at a steady state given voltage,
        if nothing is given, the equilibrium potential is used to compute membrane
        conductance.

        Parameters
        ----------
            channel_storage: dict {``channel_name``: `channel_instance`}
                dict where all ion channel objects present on the node are stored
            channel_names: List[str]
                the names of the channels to be included included in the
                conductance calculation
            v: float (optional, defaults to `self.v_ep`)
                the potential (in mV) at which to compute the membrane conductance

        Returns
        -------
            float
                the total conductance of the membrane (uS / cm^2)
        """
        if channel_names is None:
            channel_names = channel_names = list(self.currents.keys())

        g_tot = 0.0
        for channel_name in set(self.currents.keys()) & set(channel_names):
            g, e = self.currents[channel_name]
            v = self.v_ep if v is None else v

            if channel_name == "L":
                g_tot += g
            else:
                conc = self._construct_conc_args(channel_storage[channel_name])
                g_tot += g * channel_storage[channel_name].compute_p_open(v, **conc)

        return g_tot

    def calc_i_tot(self, channel_storage, channel_names=None, v=None):
        """
        Get the total conductance of the membrane at a steady state given voltage,
        if nothing is given, the equilibrium potential is used to compute membrane
        conductance.

        Parameters
        ----------
            channel_storage: dict {``channel_name``: `channel_instance`}
                dict where all ion channel objects present on the node are stored
            channel_names: List[str]
                the names of the channels to be included included in the
                conductance calculation
            v: float (optional, defaults to `self.v_ep`)
                the potential (in mV) at which to compute the membrane conductance

        Returns
        -------
            float
                the total conductance of the membrane (uS / cm^2)
        """
        if channel_names is None:
            channel_names = channel_names = list(self.currents.keys())

        i_tot = 0.0
        for channel_name in set(self.currents.keys()) & set(channel_names):
            g, e = self.currents[channel_name]
            v = self.v_ep if v is None else v

            if channel_name == "L":
                i_tot += g * (v - e)
            else:
                conc = self._construct_conc_args(channel_storage[channel_name])
                p_open = channel_storage[channel_name].compute_p_open(v, **conc)
                i_tot += g * p_open * (v - e)

        return i_tot

    def as_passive_membrane(self, channel_storage, channel_names=None, v=None):
        """
        Makes the membrane act as a passive membrane for this node, channels
        are assumed to add a conductance of g_max * p_open to the membrane
        conductance, where p_open is evaluated at the expansion point potential
        stored under `self.v_ep`

        Parameters
        ----------
        channel_storage: dict {``channel_name``: `channel_instance`}
            dict where all ion channel objects present on the node are stored
        channel_names: List[str] or None
            The channels to passify. If not provided, all channels are passified.
        v: float (optional, defaults to `self.v_ep`)
            the potential (in mV) at which to compute the membrane conductance
        """
        if channel_names is None:
            channel_names = list(self.currents.keys())
        # append leak current to channel names
        if "L" not in channel_names:
            channel_names.append("L")

        v = self.v_ep if v is None else v

        # compute the total conductance of the to be passified channels
        g_l = self.calc_g_tot(channel_storage, channel_names=channel_names, v=v)

        # compute the total current of the not to be passified channels
        i_tot = self.calc_i_tot(
            channel_storage,
            channel_names=[key for key in channel_storage if key not in channel_names],
            v=v,
        )

        # remove the passified channels
        for channel_name in channel_names:
            if channel_name == "L":
                continue

            try:
                del self.currents[channel_name]
            except KeyError:
                # the channel was not present at this node anyway
                pass

        self.currents["L"] = [g_l, v + i_tot / g_l]

    def __str__(self, with_parent=True, with_morph_info=False):
        if with_morph_info:
            node_str = super().__str__(with_parent=with_parent)
        else:
            node_str = super(MorphNode, self).__str__(with_parent=with_parent)

        node_str += (
            f" --- "
            f"r_a = {self.r_a:1.6g} MOhm*cm, "
            f"c_m = {self.c_m:1.6g} uF/cm^2, "
            f"v_ep = {self.v_ep:1.6g} mV, "
        )
        if self.g_shunt > 1e-10:
            f"g_shunt = {self.g_shunt:1.6g} uS,"
        node_str += ", ".join(
            [
                f"(g_{c} = {g:1.6g} uS/cm^2, e_{c} = {e:1.6g} mV)"
                for c, (g, e) in self.currents.items()
            ]
        )
        return node_str

    def _get_repr_dict(self):
        repr_dict = super()._get_repr_dict()
        repr_dict.update(
            {
                "currents": {
                    c: (f"({g:1.6g}, {e:1.6g})") for c, (g, e) in self.currents.items()
                },
                "concmechs": self.concmechs,
                "c_m": f"{self.c_m:1.6g}",
                "r_a": f"{self.r_a:1.6g}",
                "g_shunt": f"{self.g_shunt:1.6g}",
                "v_ep": f"{self.v_ep:1.6g}",
                "conc_eps": {
                    ion: f"{conc:1.6g}" for ion, conc in self.conc_eps.items()
                },
            }
        )
        return repr_dict

    def __repr__(self):
        return repr(self._get_repr_dict())


class PhysTree(MorphTree):
    """
    Adds physiological parameters to `neat.MorphTree` and convenience functions
    to set them across the morphology. Initialized in the same way as
    `neat.MorphTree`

    Functions for setting ion channels densities are applied to the original tree,
    which can cause the computational tree to be out of sync. To avoid this, the
    computational tree is always removed by these functions. It can be set
    afterwards with `PhysTree.set_comp_tree()`

    Attributes
    ----------
    channel_storage: dict {str: `neat.IonChannel`}
        Stores the user defined ion channels present in the tree
    ions: set {str}
        The ions for which a concentration mechanism is present in the tree
    """

    def __init__(self, arg=None, types=[1, 3, 4]):
        self.channel_storage = {}
        self.ions = set()
        super().__init__(arg=arg, types=types)
        # set basic physiology parameters (c_m = 1.0 uF/cm^2 and r_a = 0.0001 MOhm*cm),
        # but only when `arg` is a `neat.PhysNode`
        if issubclass(type(arg), PhysNode):
            for node in self:
                node.set_physiology(1.0, 100.0 / 1e6)

    def _get_repr_dict(self):
        ckeys = list(self.channel_storage.keys())
        ckeys.sort()
        return {"channel_storage": ckeys}

    def __repr__(self):
        repr_str = super().__repr__()
        return repr_str + repr(self._get_repr_dict())

    def _reset_channel_storage(self):
        new_channel_storage = {}
        for node in self:
            for channel_name in node.currents:
                if channel_name not in new_channel_storage and channel_name != "L":
                    new_channel_storage[channel_name] = self.channel_storage[
                        channel_name
                    ]

        self.channel_storage = new_channel_storage

    def create_corresponding_node(
        self, node_index, p3d=None, c_m=1.0, r_a=100 * 1e-6, g_shunt=0.0, v_ep=-75.0
    ):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return PhysNode(node_index, p3d=p3d)

    @comptree_removal_decorator
    def as_passive_membrane(self, channel_names=None, node_arg=None):
        """
        Makes the membrane act as a passive membrane (for the nodes in
        ``node_arg``), channels are assumed to add a conductance of
        g_max * p_open to the membrane conductance, where p_open for each node
        is evaluated at the expansion point potential stored in that node,
        i.e. `PhysNode.v_ep` (see `PhysTree.set_v_ep()`).

        Parameters
        ----------
        channel_names: List[str] or None
            The channels to passify. If not provided, all channels are passified.
        node_arg: optional
            see documentation of :func:`MorphTree.convert_node_arg_to_nodes`.
            Defaults to None. The nodes for which the membrane is set to
            passive
        """
        for node in self.convert_node_arg_to_nodes(node_arg):
            node.as_passive_membrane(self.channel_storage, channel_names=channel_names)

        self._reset_channel_storage()

    def _distr2Float(self, distr, node, argname=""):
        if isinstance(distr, float):
            val = distr
        elif isinstance(distr, dict):
            val = distr[node.index]
        elif hasattr(distr, "__call__"):
            d2s = self.path_length({"node": node.index, "x": 0.5}, (1.0, 0.5))
            val = distr(d2s)
        else:
            raise TypeError(
                argname + " argument should be a float, dict " + "or a callable"
            )
        return val

    @comptree_removal_decorator
    def set_v_ep(self, v_ep_distr, node_arg=None):
        """
        Set the voltage expansion points throughout the tree.

        Note that these need not correspond to the actual equilibrium potentials
        in the absence of input, but rather the (node-specific) voltage around
        which the possible expansions are computed.

        Parameters
        ----------
        v_ep_distr: float, dict or :func:`float -> float`
            The expansion point potentials [mV]
        """
        for node in self.convert_node_arg_to_nodes(node_arg):
            e = self._distr2Float(v_ep_distr, node, argname="`v_ep_distr`")
            node.set_v_ep(e)

    @comptree_removal_decorator
    def set_conc_ep(self, ion, conc_eq_distr, node_arg=None):
        """
        Set the concentration expansion points throughout the tree.

        Note that these need not correspond to the actual equilibrium concentrations
        in the absence of input, but rather the (node-specific) concentrations around
        which the possible expansions are computed.

        Parameters
        ----------
        conc_eq_distr: float, dict or :func:`float -> float`
            The expansion point concentrations [mM]
        """
        for node in self.convert_node_arg_to_nodes(node_arg):
            conc = self._distr2Float(conc_eq_distr, node, argname="`conc_eq_distr`")
            node.set_conc_ep(ion, conc)

    @comptree_removal_decorator
    def set_physiology(self, c_m_distr, r_a_distr, g_s_distr=None, node_arg=None):
        """
        Set specifice membrane capacitance, axial resistance and (optionally)
        static point-like shunt conductances in the tree. Capacitance is stored
        at each node as the attribute 'c_m' (uF/cm2) and axial resistance as the
        attribute 'r_a' (MOhm*cm)

        Parameters
        ----------
        c_m_distr: float, dict or :func:`float -> float`
            specific membrance capacitance
        r_a_distr: float, dict or :func:`float -> float`
            axial resistance
        g_s_distr: float, dict, :func:`float -> float` or None (optional, default
            is `None`)
            point like shunt conductances (placed at `(node.index, 1.)` for the
            nodes in ``node_arg``). By default no shunt conductances are added
        node_arg: optional
            see documentation of :func:`MorphTree.convert_node_arg_to_nodes`.
            Defaults to None
        """
        for node in self.convert_node_arg_to_nodes(node_arg):
            c_m = self._distr2Float(c_m_distr, node, argname="`c_m_distr`")
            r_a = self._distr2Float(r_a_distr, node, argname="`r_a_distr`")
            g_s = (
                self._distr2Float(g_s_distr, node, argname="`g_s_distr`")
                if g_s_distr is not None
                else 0.0
            )
            assert int(np.sign(g_s)) != -1
            node.set_physiology(c_m, r_a, g_s)

    @comptree_removal_decorator
    def set_leak_current(self, g_l_distr, e_l_distr, node_arg=None):
        """
        Set the parameters of the leak current. At each node, leak is stored
        under the attribute `node.currents['L']` at a tuple `(g_l, e_l)` with
        `g_l` the conductance [uS/cm^2] and `e_l` the reversal [mV]

        parameters:
        ----------
        g_l_distr: float, dict or :func:`float -> float`
            If float, the leak conductance is set to this value for all
            the nodes specified in `node_arg`. If it is a function, the input
            must specify the distance from the soma (micron) and the output
            the leak conductance [uS/cm^2] at that distance. If it is a
            dict, keys are the node indices and values the ion leak
            conductances [uS/cm^2].
        e_l_distr: float, dict or :func:`float -> float`
            If float, the reversal [mV] is set to this value for all
            the nodes specified in `node_arg`. If it is a function, the input
            must specify the distance from the soma [um] and the output
            the reversal at that distance. If it is a
            dict, keys are the node indices and values the ion reversals.
        node_arg: optional
            see documentation of :func:`MorphTree.convert_node_arg_to_nodes`.
            Defaults to None
        """
        for node in self.convert_node_arg_to_nodes(node_arg):
            g_l = self._distr2Float(g_l_distr, node, argname="`g_l_distr`")
            e_l = self._distr2Float(e_l_distr, node, argname="`e_l_distr`")
            assert int(np.sign(g_l)) != -1
            node._add_current("L", g_l, e_l)

    @comptree_removal_decorator
    def add_channel_current(self, channel, g_max_distr, e_rev_distr, node_arg=None):
        """
        Adds a channel to the morphology. At each node, the channel is stored
        under the attribute `node.currents[channel.__class__.__name__]` as a
        tuple `(g_max, e_rev)` with `g_max` the maximal conductance [uS/cm^2]
        and `e_rev` the reversal [mV]

        Parameters
        ----------
        channel_name: :class:`IonChannel`
            The ion channel
        g_max_distr: float, dict or :func:`float -> float`
            If float, the maximal conductance is set to this value for all
            the nodes specified in `node_arg`. If it is a function, the input
            must specify the distance from the soma (micron) and the output
            the ion channel density (uS/cm^2) at that distance. If it is a
            dict, keys are the node indices and values the ion channel
            densities (uS/cm^2).
        e_rev_distr: float, dict or :func:`float -> float`
            If float, the reversal (mV) is set to this value for all
            the nodes specified in `node_arg`. If it is a function, the input
            must specify the distance from the soma (micron) and the output
            the reversal at that distance. If it is a
            dict, keys are the node indices and values the ion reversals.
        node_arg: optional
            see documentation of :func:`MorphTree.convert_node_arg_to_nodes`.
            Defaults to None
        """
        if not isinstance(channel, ionchannels.IonChannel):
            raise IOError("`channel` argmument needs to be of class `neat.IonChannel`")
        channel_name = channel.__class__.__name__

        nodes_with_channel = self.convert_node_arg_to_nodes(node_arg)
        if len(nodes_with_channel) > 0:
            self.channel_storage[channel_name] = channel

        # add the ion channel to the nodes
        for node in self.convert_node_arg_to_nodes(node_arg):
            g_max = self._distr2Float(g_max_distr, node, argname="`g_max_distr`")
            e_rev = self._distr2Float(e_rev_distr, node, argname="`e_rev_distr`")
            assert int(np.sign(g_max)) != -1
            node._add_current(channel_name, g_max, e_rev)

    def get_channels_in_tree(self):
        """
        Returns list of strings of all channel names in the tree

        Returns
        -------
        list of string
            the channel names
        """
        return list(self.channel_storage.keys())

    @comptree_removal_decorator
    def add_conc_mech(self, ion, params={}, node_arg=None):
        """
        Add a concentration mechanism to the tree

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        params: dict
            parameters for the concentration mechanism
        node_arg:
            see documentation of :func:`MorphTree.convert_node_arg_to_nodes`.
            Defaults to None
        """
        self.ions.add(ion)
        for node in self.convert_node_arg_to_nodes(node_arg):
            node.add_conc_mech(ion, params=params)

    @comptree_removal_decorator
    def fit_leak_current(self, e_eq_target_distr, tau_m_target_distr, node_arg=None):
        """
        Fits the leak current to fix equilibrium potential and membrane time-
        scale.

        !!! Should only be called after all ion channels have been added !!!

        Parameters
        ----------
        e_eq_target_distr: float, dict or :func:`float -> float`
            The target reversal potential (mV). If float, the target reversal is
            set to this value for all the nodes specified in `node_arg`. If it
            is a function, the input must specify the distance from the soma (um)
            and the output the target reversal at that distance. If it is a
            dict, keys are the node indices and values the target reversals
        tau_m_target_distr: float, dict or :func:`float -> float`
            The target membrane time-scale (ms). If float, the target time-scale is
            set to this value for all the nodes specified in `node_arg`. If it
            is a function, the input must specify the distance from the soma (um)
            and the output the target time-scale at that distance. If it is a
            dict, keys are the node indices and values the target time-scales
        node_arg:
            see documentation of :func:`MorphTree.convert_node_arg_to_nodes`.
            Defaults to None
        """
        for node in self.convert_node_arg_to_nodes(node_arg):
            e_eq_target = self._distr2Float(
                e_eq_target_distr, node, argname="`g_max_distr`"
            )
            tau_m_target = self._distr2Float(
                tau_m_target_distr, node, argname="`e_rev_distr`"
            )
            assert tau_m_target > 0.0
            node.fit_leak_current(
                e_eq_target=e_eq_target,
                tau_m_target=tau_m_target,
                channel_storage=self.channel_storage,
            )

    def _evaluate_comp_criteria(self, node, eps=1e-8, rbool=False):
        """
        Return ``True`` if relative difference in any physiological parameters
        between node and child node is larger than margin ``eps``.

        Overrides the `MorphTree._evaluate_comp_criteria()` function called by
        `MorphTree.set_comp_tree()`.

        Parameters
        ----------
        node: ::class::`MorphNode`
            node that is compared to parent node
        eps: float (optional, default ``1e-8``)
            the margin

        return
        ------
        bool
        """
        rbool = super()._evaluate_comp_criteria(node, eps=eps, rbool=rbool)

        if not rbool:
            cnode = node.child_nodes[0]
            rbool = np.abs(node.r_a - cnode.r_a) > eps * np.max([node.r_a, cnode.r_a])
        if not rbool:
            rbool = np.abs(node.c_m - cnode.c_m) > eps * np.max([node.c_m, cnode.c_m])
        if not rbool:
            rbool = set(node.currents.keys()) != set(cnode.currents.keys())
        if not rbool:
            for chan_name, channel in node.currents.items():
                if not rbool:
                    rbool = np.abs(
                        channel[0] - cnode.currents[chan_name][0]
                    ) > eps * np.max(
                        [np.abs(channel[0]), np.abs(cnode.currents[chan_name][0])]
                    )
                if not rbool:
                    rbool = np.abs(
                        channel[1] - cnode.currents[chan_name][1]
                    ) > eps * np.max(
                        [np.abs(channel[1]), np.abs(cnode.currents[chan_name][1])]
                    )
        if not rbool:
            rbool = node.g_shunt > 0.001 * eps

        return rbool

    def create_new_tree(self, loc_arg, name="new tree", fake_soma=False, new_tree=None):
        """
        Creates a new tree where the provided location in `loc_arg` are now the nodes.
        Note that if the soma is not in the list of locations, a common root location
        might be added if necessary.

        Distance relations between locations are maintained (note that this
        relation is stored in `L` attribute of `neat.MorphNode`, the `p3d`
        attribute containing the 3d coordinates does not maintain distances)

        The radius of a node is taken as the average radius between the location
        associated with the node and the location associated with the parent node,
        weighted by the lengths of all individual nodes.

        Physiological parameters are copied from the original node on which the
        new node is located.

        Parameters
        ----------
        loc_arg: list of `neat.MorphLoc` or string
            the locations. If list of locs, they will be stored under the name
            `new_tree`
        name: str (default 'new tree')
            The name under which the locations associated to the tree are stored.
        fake_soma: bool (default `False`)
            if `True`, finds the common root of the set of locations and
            uses that as the soma of the new tree. If `False`, the real soma
            is used.
        new_tree: `None` or instance of subclass of `neat.MorphTree`
            The new tree instance.

        Returns
        -------
        `neat.MorphTree`
            The new tree.
        """
        if new_tree is not None and not issubclass(type(new_tree), PhysTree):
            raise ValueError(
                f"`new_tree` is an instance of {new_tree.__class__}, "
                f"but should be a subclass of <class 'neat.PhysTree'>."
            )

        new_tree = super().create_new_tree(
            loc_arg, name, fake_soma=fake_soma, new_tree=new_tree
        )
        new_locs = self.get_locs(name)

        new_channels = set()
        new_ions = set()
        for new_node in new_tree:

            loc = new_locs[new_node.content["loc idx"]]
            orig_node = self[loc["node"]]

            # copy over physiological parameters
            new_node.c_m = orig_node.c_m
            new_node.r_a = orig_node.r_a
            new_node.g_shunt = orig_node.g_shunt

            new_node.v_ep = orig_node.v_ep
            new_node.conc_eps = copy.deepcopy(orig_node.conc_eps)

            new_node.currents = copy.deepcopy(orig_node.currents)
            new_node.concmechs = copy.deepcopy(orig_node.concmechs)

            new_channels.update(set(new_node.currents.keys()))
            new_ions.update(set(new_node.concmechs.keys()))

        new_tree.channel_storage = {
            cname: channel
            for cname, channel in self.channel_storage.items()
            if cname in new_channels
        }
        new_tree.ions = new_ions

        return new_tree

    @computational_tree_decorator
    def create_finite_difference_tree(self, dx_max=15.0, name="dont store"):
        """
        Create a ::class::`neat.CompartmentTree` whose parameters implement the
        second order finite difference approximation for the morphology.

        Parameters
        ----------
        dx_max: float
            Maximum distance step between compartments (in [um]). By default,
            each node of this tree will correspond to at least one compartment,
            and thus one node in the comparment tree. If the length of a node
            exceeds `dx_max`, there will be the smallest possible number of
            equally spaced comparments so that the distance between them does
            not exceed `dx_max`. Note that if the computational tree is active,
            the computational nodes will be taken as a reference for placing
            the compartment locations.
        name: string
            If given, stores the compartment locations in this tree. Default
            is to not store the locations.

        Returns
        -------
        comptree: ::class::`neat.CompartmentTree`
            The compartment tree
        locs: list of ::class::`neat.MorphLoc`
            The location corresponding to the compartments of the finite
            difference approximation
        """

        # create the list of compartment locations for FD approximation
        # locs = []
        # for node in self:
        #     if self.is_root(node):
        #         locs.append(
        #             MorphLoc((node.index, 0.5), self, set_as_comploc=set_as_comploc)
        #         )
        #     else:
        #         n_comp = np.ceil(node.L / dx_max).astype(int)

        #         for cc in range(1, n_comp + 1):
        #             new_loc = MorphLoc(
        #                 (node.index, cc / n_comp), self, set_as_comploc=set_as_comploc
        #             )
        #             locs.append(new_loc)

        locs = self.distribute_locs_finite_diff(dx_max=dx_max, name=name)

        aux_tree = self.create_new_tree(locs, new_tree=PhysTree())
        fd_tree = self.create_compartment_tree(locs)

        for ii, (fd_node, aux_node, loc) in enumerate(zip(fd_tree, aux_tree, locs)):
            assert aux_node.content["loc idx"] == fd_node.loc_idx

            # unit conversion [um] -> [cm]
            R_ = aux_node.R * 1e-4
            L_ = aux_node.L * 1e-4

            if fd_tree.is_root(fd_node):
                # for the soma we apply the spherical approximation
                surf = 4.0 * np.pi * R_**2
            else:
                # for other nodes we apply the cylindrical approximation
                # but take only half of it (half for the current node, half
                # for the parent
                surf = 2.0 * np.pi * R_ * L_ / 2.0

            # set finite difference values for current node
            fd_node.ca = surf * aux_node.c_m
            fd_node.currents = {
                chan: (surf * g, e) for chan, (g, e) in aux_node.currents.items()
            }

            for chan in aux_node.currents:
                if chan not in fd_tree.channel_storage and chan in self.channel_storage:
                    fd_tree.channel_storage[chan] = copy.deepcopy(
                        self.channel_storage[chan]
                    )

            if not fd_tree.is_root(fd_node):
                fd_node.g_c = np.pi * R_**2 / (aux_node.r_a * L_)

                # add finite difference contributions to parent
                fd_parent = fd_node.parent_node
                fd_parent.ca += surf * aux_node.c_m

                for chan in aux_node.currents:
                    g_node = surf * aux_node.currents[chan][0]
                    e_node = aux_node.currents[chan][1]

                    if chan in fd_parent.currents:
                        g_parent = fd_parent.currents[chan][0]
                        e_parent = fd_parent.currents[chan][1]
                    else:
                        g_parent = 0.0
                        e_parent = aux_node.currents[chan][1]

                    if g_parent + g_node > 1e-10:
                        fd_parent.currents[chan] = (
                            g_parent + g_node,
                            (g_parent * e_parent + g_node * e_node)
                            / (g_parent + g_node),
                        )
                    else:
                        fd_parent.currents[chan] = (0.0, e_parent)

        # set concentration mechanisms in separate pass
        for ii, (fd_node, aux_node, loc) in enumerate(zip(fd_tree, aux_tree, locs)):
            for ion in aux_node.concmechs:
                ion_factors_aux = 0.0
                ion_factors_fd = 0.0

                for cname in aux_node.currents:
                    if cname != "L" and self.channel_storage[cname].ion == ion:
                        ion_factors_aux += aux_node.currents[cname][0]
                        ion_factors_fd += fd_node.currents[cname][0]
                # if no channels carry the `ion`, revert to leak to compute rescale factors
                if ion_factors_fd < 1e-12:
                    ion_factors_aux = aux_node.currents["L"][0]
                    ion_factors_fd = fd_node.currents["L"][0]

                fd_node.concmechs[ion] = copy.deepcopy(aux_node.concmechs[ion])
                fd_node.concmechs[ion].gamma *= ion_factors_aux / ion_factors_fd

        return fd_tree, locs
