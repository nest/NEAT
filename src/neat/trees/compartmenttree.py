# -*- coding: utf-8 -*-
#
# compartmenttree.py
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
import scipy.linalg as la
import scipy.optimize as so

from .stree import SNode, STree
from ..tools import kernelextraction as ke
from ..channels import ionchannels, concmechs
from ..factorydefaults import DefaultPhysiology

import copy
import warnings
import itertools
from operator import mul
from functools import reduce
from typing import Literal


CFG = DefaultPhysiology()


class CompartmentNode(SNode):
    """
    Implements a node for `CompartmentTree`

    Attributes
    ----------
    loc_idx: int
        The index of the location in the list of locations on which this reduction was
        based
    ca: float
        capacitance of the compartment (uF)
    g_l: float
        leak conductance at the compartment (uS)
    g_c: float
        Coupling conductance of compartment with parent compartment (uS).
        Ignore if node is the root
    e_eq: float
        equilibrium potential at the compartment
    currents: dict {str: [g_bar, e_rev]}
        dictionary with as keys the channel names and as elements lists of length
        two with contain the maximal conductance (uS) and the channels'
        reversal potential in (mV)
    concmechs: dict {str: `neat.channels.concmechs.ConcMech`}
        dictionary with as keys the ion names and as values the concentration
        mechanisms governing the concentration of each ion channel
    expansion_points: dict {str: np.ndarray}
        dictionary with as keys the channel names and as elements the state
        variables of the ion channel around which to compute the linearizations
    """

    def __init__(self, index, loc_idx=None, ca=1.0, g_c=0.0, g_l=1e-2, e_eq=-75.0):
        super().__init__(index)
        # location index this node corresponds to
        self._loc_idx = loc_idx
        # compartment params
        self.ca = ca  # capacitance (uF)
        self.g_c = g_c  # coupling conductance (uS)
        self.e_eq = e_eq  # equilibrium potential (mV)
        self.conc_eqs = {}  # equilibrium concentration values (mM)
        self.currents = {
            "L": [g_l, e_eq]
        }  # ion channel conductance (uS) and reversals (mV)
        self.concmechs = {}
        self.expansion_points = {}

    def set_loc_idx(self, loc_idx):
        self._loc_idx = loc_idx

    def get_loc_idx(self):
        if self._loc_idx is None:
            raise AttributeError(
                "`self.loc_idx` is undefined, this node has "
                + "not been associated with a location"
            )
        else:
            return self._loc_idx

    loc_idx = property(get_loc_idx, set_loc_idx)

    def __str__(self, with_parent=False, with_children=False):
        node_string = super(CompartmentNode, self).__str__()
        if self.parent_node is not None:
            node_string += (
                ", Parent: " + super(CompartmentNode, self.parent_node).__str__()
            )
        node_string += (
            " --- (g_c = %.12f uS, " % self.g_c
            + ", ".join(
                [
                    "g_" + cname + " = %.12f uS" % cpar[0]
                    for cname, cpar in self.currents.items()
                ]
            )
            + ", c = %.12f uF)" % self.ca
        )
        return node_string

    def set_conc_eq(self, ion, conc):
        """
        Set the equilibrium concentration value at this node

        Parameters
        ----------
        ion: str ('ca', 'k', 'na')
            the ion for which the concentration is to be set
        conc: float
            the concentration value (mM)
        """
        self.conc_eqs[ion] = conc

    def _add_current(self, channel_name, e_rev):
        """
        Add an ion channel current at this node. ('L' as `channel_name`
        signifies the leak current)

        Parameters
        ----------
        channel_name: string
            the name of the current
        e_rev: float
            the reversal potential of the current (mV)
        """
        self.currents[channel_name] = [0.0, e_rev]

    def add_conc_mech(self, ion, **kwargs):
        """
        Add a concentration mechanism at this node.

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        kwargs: dict
            parameters for the concentration mechanism that are not used in the
            fits (only used for NEURON model)
        """
        if "tau" in kwargs:
            self.concmechs[ion] = concmechs.ExpConcMech(ion, kwargs["tau"], 0.0)
        else:
            warnings.warn(
                "These parameters do not match any NEAT concentration "
                + "mechanism, no concentration mechanism has been added",
                UserWarning,
            )

    def set_expansion_point(self, channel_name, statevar):
        """
        Set the choice for the state variables of the ion channel around which
        to linearize.

        Note that when adding an ion channel to the node, the default expansion
        point setting is to linearize around the asymptotic values for the state
        variables at the equilibrium potential store in `self.e_eq`.
        Hence, this function only needs to be called to change that setting.

        Parameters
        ----------
        channel_name: string
            the name of the ion channel
        statevar: dict of float
            values of the state variable expansion point
        """
        if statevar is None:
            statevar = {}
        self.expansion_points[channel_name] = statevar

    def get_expansion_point(self, channel_name):
        try:
            return self.expansion_points[channel_name]
        except KeyError:
            self.expansion_points[channel_name] = {}
            return self.expansion_points[channel_name]

    def _construct_channel_args(self, channel):
        """
        Returns the expansion points for the channel, around which the
        linearization in computed.

        For voltage, checks if 'v' key is in `self.expansion_points`, otherwise
        defaults to `self.e_eq`.

        For concentrations, checks if the ion is in `self.expansion_points`,
        otherwise checks if a concentration of the ion is given in
        `self.conc_eqs`, and otherwise defaults to the factory default in
        `neat.channels.ionchannels`.

        Parameters
        ----------
        channel: `neat.IonChannel` object
            the ion channel

        Returns
        v: float or np.ndarray
            The voltage values for the expansion points
        sv: dict ({str: np.ndarray})
            The state variables and/or concentrations at the expansion points.
        """
        # check if linearistation needs to be computed around expansion point
        sv = self.get_expansion_point(channel.__class__.__name__).copy()

        # if voltage is not in expansion point, use equilibrium potential
        v = sv.pop("v", self.e_eq)

        # if concencentration is in expansion point, use it. Otherwise use
        # concentration in equilibrium concentrations (self.conc_eqs), if
        # it is there. If not, use default concentration.
        ions = [
            str(ion) for ion in channel.conc
        ]  # convert potential sympy symbols to str
        conc = {
            ion: sv.pop(ion, self.conc_eqs.copy().pop(ion, CFG.conc[ion]))
            for ion in ions
        }
        sv.update(conc)

        return v, sv

    def calc_membrane_conductance_terms(
        self, channel_storage, freqs=0.0, v=None, channel_names=None
    ):
        """
        Contribution of linearized ion channel to conductance matrix

        Parameters
        ----------
        channel_storage: dict of ion channels
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized
        freqs: np.ndarray (ndim = 1, dtype = complex or float) or float or complex
            The frequencies at which the impedance terms are to be evaluated
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the ion channels that have to be included in the
            conductance term

        Returns
        -------
        dict of np.ndarray or float or complex
            Each entry in the dict is of the same type as ``freqs`` and is the
            conductance term of a channel
        """
        if channel_names is None:
            channel_names = list(self.currents.keys())

        cond_terms = {}
        if "L" in channel_names:
            cond_terms["L"] = 1.0  # leak conductance has 1 as prefactor

        for channel_name in set(channel_names) - set("L"):
            e = self.currents[channel_name][1]
            # get the ionchannel object
            channel = channel_storage[channel_name]

            v, sv = self._construct_channel_args(channel)
            # add linearized channel contribution to membrane conductance
            cond_terms[channel_name] = -channel.compute_lin_sum(v, freqs, e, **sv)

        return cond_terms

    def calc_membrane_concentration_terms(
        self,
        ion,
        channel_storage,
        freqs=0.0,
        v=None,
        channel_names=None,
        fit_type="gamma",
    ):
        """
        Contribution of linearized concentration dependence to conductance matrix

        Parameters
        ----------
        ion: str
            The ion for which the concentration terms are to be calculated
        channel_storage: dict of ion channels
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized
        freqs: np.ndarray (ndim = 1, dtype = complex or float) or float or complex
            The frequencies at which the impedance terms are to be evaluated
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the ion channels that have to be included in the
            conductance term

        Returns
        -------
        dict of np.ndarray or float or complex
            Each entry in the dict is of the same type as ``freqs`` and is the
            conductance term of a channel
        """
        if channel_names is None:
            channel_names = list(self.currents.keys())

        conc_write_channels = np.zeros_like(freqs)
        conc_read_channels = np.zeros_like(freqs)

        for channel_name in channel_names:
            if channel_name == "L":
                continue

            g, e = self.currents[channel_name]
            channel = channel_storage[channel_name]

            v, sv = self._construct_channel_args(channel)

            # if the channel adds to ion channel current, add it here
            if channel.ion == ion:
                conc_write_channels = conc_write_channels - g * channel.compute_lin_sum(
                    v, freqs, e, **sv
                )

            # if channel reads the ion channel current, add it here
            if ion in channel.conc:
                conc_read_channels = conc_read_channels - g * channel.compute_lin_conc(
                    v, freqs, ion, e, **sv
                )

        if fit_type == "gamma":
            return (
                conc_write_channels
                * conc_read_channels
                * self.concmechs[ion].compute_lin(freqs)
            )

        elif fit_type == "tau":
            c0, c1 = self.concmechs[ion].compute_lin_tau_fit(freqs)
            return conc_write_channels * conc_read_channels * c0, c1

        else:
            raise NotImplementedError("Unkown fit type, choose 'gamma' or 'tau'")

    def calc_g_tot(
        self, channel_storage, v=None, channel_names=None, p_open_channels=None
    ):
        """
        Compute the total conductance of a set of channels evaluated at a given
        voltage

        Parameters
        ----------
        channel_storage: dict {str: `neat.IonChannel`}
            Dictionary of all ion channels on the `neat.CompartmentTree`
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the channel that have to be included in the calculation
        p_open_channels: dict {str: float}, optional
            The open probalities of the channels. Custom set of open
            probabilities. Overwrites both `self.expansion_point` and `v`.
            Defaults to `None`.

        Returns
        -------
        float: the total conductance
        """
        if channel_names is None:
            channel_names = list(self.currents.keys())

        # compute total conductance around `self.e_eq`
        g_tot = self.currents["L"][0] if "L" in channel_names else 0.0

        for channel_name in channel_names:
            if channel_name == "L":
                continue

            g, e = self.currents[channel_name]
            channel = channel_storage[channel_name]

            v, sv = self._construct_channel_args(channel)

            # open probability
            if p_open_channels is None:
                p_o = channel.compute_p_open(v, **sv)
            else:
                p_o = p_open_channels[channel_name]

            # add to total conductance
            g_tot = g_tot + g * p_o

        return g_tot

    def calc_i_tot(
        self, channel_storage, v=None, channel_names=None, p_open_channels={}
    ):
        """
        Compute the total current of a set of channels evaluated at a given
        voltage

        Parameters
        ----------
        channel_storage: dict {str: `neat.IonChannel`}
            Dictionary of all ion channels on the `neat.CompartmentTree`
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the channel that have to be included in the calculation
        p_open_channels: dict {str: float}, optional
            The open probalities of the channels. Custom set of open
            probabilities. Overwrites probabilities given by both
            `self.expansion_point` and `v`. Defaults to `None`.

        Returns
        -------
        float: the total conductance
        """

        if channel_names is None:
            channel_names = list(self.currents.keys())

        i_tot = 0.0
        for channel_name in channel_names:
            g, e = self.currents[channel_name]

            if channel_name == "L":
                v = self.e_eq
                i_tot = i_tot + g * (v - e)

                continue

            channel = channel_storage[channel_name]
            v, sv = self._construct_channel_args(channel)

            if channel_name not in p_open_channels:
                i_tot = i_tot + g * channel.compute_p_open(v, **sv) * (v - e)

            else:
                i_tot = i_tot + g * p_open_channels[channel_name] * (v - e)

        return i_tot

    def calc_linear_statevar_terms(self, channel_storage, v=None, channel_names=None):
        """
        Contribution of linearized ion channel to conductance matrix

        Parameters
        ----------
        channel_storage: dict of ion channels
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized
        freqs: np.ndarray (ndim = 1, dtype = complex or float) or float or complex
            The frequencies at which the impedance terms are to be evaluated
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the ion channels that have to be included in the
            conductance term

        Returns
        -------
        dict of np.ndarray or float or complex
            Each entry in the dict is of the same type as ``freqs`` and is the
            conductance term of a channel
        """
        if channel_names is None:
            channel_names = list(self.currents.keys())

        svar_terms = {}
        for channel_name in set(channel_names) - set("L"):
            g, e = self.currents[channel_name]

            # get the ionchannel object
            channel = channel_storage[channel_name]
            v, sv = self._construct_channel_args(channel)

            # add linearized channel contribution to membrane conductance
            dp_dx = channel.compute_derivatives(v, **sv)[0]

            svar_terms[channel_name] = {}
            for svar, dp_dx_ in dp_dx.items():
                svar_terms[channel_name][svar] = g * dp_dx_ * (e - v)

        return svar_terms

    def _add_linear_system_terms(
        self, cc, V2V, Y2V, V2Y, Y2Y, channel_storage, channel_names=None
    ):
        """
        Contribution of linearized ion channel to conductance matrix

        Parameters
        ----------
        channel_storage: dict of ion channels
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized
        freqs: np.ndarray (ndim = 1, dtype = complex or float) or float or complex
            The frequencies at which the impedance terms are to be evaluated
        v: float (optional, default is None which evaluates at `self.e_eq`)
            The potential at which to compute the total conductance
        channel_names: list of str
            The names of the ion channels that have to be included in the
            conductance term

        Returns
        -------
        dict of np.ndarray or float or complex
            Each entry in the dict is of the same type as ``freqs`` and is the
            conductance term of a channel
        """
        if channel_names is None:
            channel_names = list(self.currents.keys())

        ii = self.index
        if self.parent_node != None:
            pp = self.parent_node.index
            V2V[pp, pp] -= self.g_c
            V2V[ii, pp] += self.g_c
            V2V[pp, ii] += self.g_c
        V2V[ii, ii] -= self.currents["L"][0] + self.g_c

        for channel_name in set(channel_names) - set("L"):
            g, e = self.currents[channel_name]

            # get the ionchannel object
            channel = channel_storage[channel_name]
            v, sv = self._construct_channel_args(channel)
            n_sv = len(channel.statevars)
            sv_idxs = list(range(cc, cc + n_sv))

            # add linearized channel contribution to membrane conductance
            p_o = channel.compute_p_open(v, **sv)
            dp_dx, df_dv, df_dx = channel.compute_derivatives(v, **sv)

            dp_dx = np.array([dp_dx[sv] for sv in channel.ordered_statevars])
            df_dv = np.array([df_dv[sv] for sv in channel.ordered_statevars])
            df_dx = np.array([df_dx[sv] for sv in channel.ordered_statevars])

            V2V[ii, ii] -= g * p_o
            Y2V[ii, cc : cc + n_sv] += g * dp_dx * (e - v)
            V2Y[cc : cc + n_sv, ii] += df_dv * 1e3  # convert to 1 / s
            Y2Y[sv_idxs, sv_idxs] += df_dx * 1e3  # convert to 1 / s

            cc += n_sv

        for child in self.child_nodes:
            child._add_linear_system_terms(cc, V2V, Y2V, V2Y, Y2Y, channel_storage)

    def __str__(self, with_parent=True, with_morph_info=False):
        node_str = super().__str__(with_parent=with_parent)

        node_str += (
            f" --- "
            f"loc_idx = {self._loc_idx}, "
            f"g_c = {self.g_c} uS, "
            f"ca = {self.ca} uF, "
            f"e_eq = {self.e_eq} mV, "
        )

        node_str += ", ".join(
            [f"(g_{c} = {g} uS, e_{c} = {e} mV)" for c, (g, e) in self.currents.items()]
        )

        return node_str

    def _get_repr_dict(self):
        repr_dict = super()._get_repr_dict()
        repr_dict.update(
            {
                "loc_idx": self._loc_idx,
                "ca": f"{self.ca:1.6g}",
                "g_c": f"{self.g_c:1.6g}",
                "e_eq": f"{self.e_eq:1.6g}",
                "conc_eqs": self.conc_eqs,
                "currents": {
                    c: (f"{g:1.6g}, {e:1.6g}") for c, (g, e) in self.currents.items()
                },
                "concmechs": self.concmechs,
                "expansion_points": self.expansion_points,
            }
        )
        return repr_dict

    def __repr__(self):
        return repr(self._get_repr_dict())


class CompartmentTree(STree):
    """
    Abstract tree that implements physiological parameters for reduced
    compartmental models. Also implements the matrix algebra to fit physiological
    parameters to impedance matrices

    Attributes
    ----------
    channel_storage: dict {str: `neat.IonChannel`}
        Stores the user defined ion channels present in the tree
    """

    def __init__(self, arg=None):
        self.channel_storage = {}
        super().__init__(arg)
        # for fitting the model
        self.reset_fit_data()

    def _get_repr_dict(self):
        ckeys = list(self.channel_storage.keys())
        ckeys.sort()
        return {"channel_storage": ckeys}

    def __repr__(self):
        repr_str = super().__repr__()
        return repr_str + repr(self._get_repr_dict())

    def create_corresponding_node(self, index, ca=1.0, g_c=0.0, g_l=1e-2):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
        node_index: int
            index of the new node
        """
        return CompartmentNode(index, ca=ca, g_c=g_c, g_l=g_l)

    def get_nodes_from_loc_idxs(self, *args):
        """
        find the nodes that correspond(s) to a (list of) location
        index (indices)

        Parameters
        ----------
        args: `int` or `list` of `int`
            location indices

        Returns
        -------
        `neat.CompartmentNode` or `list` of `neat.CompartmentNode
        """
        nodes = []

        idxs = args[0]
        was_int = False
        if isinstance(idxs, int):
            idxs = [idxs]
            was_int = True

        for idx in idxs:

            found = False
            for node in self:
                if node.loc_idx == idx:
                    nodes.append(node)
                    found = True
                    break

            if not found:
                raise IndexError(f"Location index {idx} not in tree")

        if was_int:
            return nodes[0]
        else:
            return nodes

    def _reset_channel_storage(self):
        new_channel_storage = {}
        for node in self:
            for channel_name in node.currents:
                if channel_name not in new_channel_storage and channel_name != "L":
                    new_channel_storage[channel_name] = self.channel_storage[
                        channel_name
                    ]

        self.channel_storage = new_channel_storage

    def set_e_eq(self, e_eq, indexing="locs"):
        """
        Set the equilibrium potential at all nodes on the compartment tree

        Parameters
        ----------
        e_eq: float or np.array of floats
            The equilibrium potential(s). If a float, the same potential is set
            at every node. If a numpy array, must have the same length as `self`
        indexing: 'locs' or 'tree'
            The ordering of the equilibrium potentials. If 'locs', assumes the
            equilibrium potentials are in the order of the list of locations
            to which the tree is fitted. If 'tree', assumes they are in the order
            of which nodes appear during iteration
        """
        if isinstance(e_eq, float) or isinstance(e_eq, int):
            e_eq = e_eq * np.ones(len(self), dtype=float)
        elif indexing == "locs":
            e_eq = self._permute_to_tree(np.array(e_eq))

        for ii, node in enumerate(self):
            node.e_eq = e_eq[ii]

    def get_e_eq(self, indexing="locs"):
        """
        Get the equilibrium potentials at each node.

        Parameters
        ----------
        indexing: 'locs' or 'tree'
            The ordering of the returned array. If 'locs', returns the array
            in the order of the list of locations to which the tree is fitted.
            If 'tree', returns the array in the order in which nodes appear
            during iteration

        Returns
        -------
        np.array
            The equilibrium potentials
        """
        e_eq = np.array([node.e_eq for node in self])
        if indexing == "locs":
            e_eq = self._permuteToLocs(e_eq)
        return e_eq

    def set_conc_eq(self, ion, conc_eq, indexing="locs"):
        """
        Set the equilibrium concentrations at all nodes in the compartment tree

        Parameters
        ----------
        conc_eq: `np.array` or float
            The equilibrium concentrations [mM]
        """
        if isinstance(conc_eq, float) or isinstance(conc_eq, int):
            conc_eq = conc_eq * np.ones(len(self), dtype=float)
        elif indexing == "locs":
            conc_eq = self._permute_to_tree(np.array(conc_eq))

        for ii, node in enumerate(self):
            node.set_conc_eq(ion, conc_eq[ii])

    def get_conc_eq(self, ion, indexing="locs"):
        """
        Get the equilibrium concentrations of 'ion' at each node.

        Parameters
        ----------
        ion: str
            The ion for which to get the concentrations
        indexing: 'locs' or 'tree'
            The ordering of the returned array. If 'locs', returns the array
            in the order of the list of locations to which the tree is fitted.
            If 'tree', returns the array in the order in which nodes appear
            during iteration

        Returns
        -------
        np.array
            The equilibrium concentrations
        """
        conc_eq = np.array([node.conc_eqs[ion] for node in self])
        if indexing == "locs":
            conc_eq = self._permuteToLocs(conc_eq)
        return conc_eq

    def set_expansion_points(self, expansion_points):
        """
        Set the choice for the state variables of the ion channel around which
        to linearize.

        Note that when adding an ion channel to the tree, the default expansion
        point setting is to linearize around the asymptotic values for the state
        variables at the equilibrium potential store in `self.e_eq`.
        Hence, this function only needs to be called to change that setting.

        Parameters
        ----------
        expansion_points: dict {`channel_name`: ``None`` or dict}
            dictionary with as keys `channel_name` the name of the ion channel
            and as value its expansion point
        """
        to_tree_inds = self.permute_to_tree_idxs()
        for channel_name, expansion_point in expansion_points.items():
            # if one set of state variables, set throughout neuron
            if expansion_point is None:
                eps = [None for _ in self]
            else:
                eps = [{} for _ in self]
                for svar, exp_p in expansion_point.items():
                    if np.ndim(exp_p) == 0:
                        for ep in eps:
                            ep[svar] = exp_p
                    else:
                        assert len(exp_p) == len(self)
                        for ep, ep_ in zip(eps, exp_p[to_tree_inds]):
                            ep[svar] = ep_

            for node, ep in zip(self, eps):
                node.set_expansion_point(channel_name, ep)

    def remove_expansion_points(self):
        for node in self:
            node.expansion_points = {}

    def fit_e_leak(self):
        """
        Fit the leak reversal potential to obtain the stored equilibirum potentials
        as resting membrane potential
        """
        e_l_0 = self.get_e_eq(indexing="tree")
        # compute the solutions
        fun = self._fun_e_leak_fit(e_l_0)
        jac = self._jac_e_leak_fit(e_l_0)
        e_l = np.linalg.solve(jac, -fun + np.dot(jac, e_l_0))
        # set the leak reversals
        for ii, node in enumerate(self):
            node.currents["L"] = [node.currents["L"][0], e_l[ii]]

    def _fun_e_leak_fit(self, e_l):
        # set the leak reversal potentials
        for ii, node in enumerate(self):
            node.currents["L"] = [node.currents["L"][0], e_l[ii]]
        # compute the function values (currents)
        fun_vals = np.zeros(len(self))
        for ii, node in enumerate(self):
            fun_vals[ii] += node.calc_i_tot(self.channel_storage)
            # add the parent node coupling term
            if node.parent_node is not None:
                fun_vals[ii] += node.g_c * (node.e_eq - node.parent_node.e_eq)
            # add the child node coupling terms
            for cnode in node.child_nodes:
                fun_vals[ii] += cnode.g_c * (node.e_eq - cnode.e_eq)
        return fun_vals

    def _jac_e_leak_fit(self, e_l):
        for ii, node in enumerate(self):
            node.currents["L"][1] = e_l[ii]
        jac_vals = np.array([-node.currents["L"][0] for node in self])
        return np.diag(jac_vals)

    def add_channel_current(self, channel, e_rev):
        """
        Add an ion channel current to the tree

        Parameters
        ----------
        channel_name: string
            The name of the channel type
        e_rev: float
            The reversal potential of the ion channel [mV]
        """

        channel_name = channel.__class__.__name__
        self.channel_storage[channel_name] = channel
        for ii, node in enumerate(self):
            node._add_current(channel_name, e_rev)

    def add_conc_mech(self, ion, params={}):
        """
        Add a concentration mechanism to the tree

        Parameters
        ----------
        ion: string
            the ion the mechanism is for
        params: dict
            parameters for the concentration mechanism
        """
        for node in self:
            node.add_conc_mech(ion, params=params)

    def permute_to_tree_idxs(self):
        """
        Returns index array for the permutation of location indices to tree indices
        """
        return np.array([node.loc_idx for node in self])

    def _permute_to_tree(self, mat):
        """
        Permutes the input array, which is assumed to be ordered according to the
        location list, to the tree order
        """
        index_arr = self.permute_to_tree_idxs()
        if mat.ndim == 1:
            return mat[index_arr]
        else:
            return mat[..., index_arr, :][..., :, index_arr]

    def permute_to_locs_idxs(self):
        """
        Return an index array that can be used to permute matrices that follow to tree
        order to the location list order
        """
        loc_idxs = np.array([node.loc_idx for node in self])
        return np.argsort(loc_idxs)

    def _permuteToLocs(self, mat):
        index_arr = self.permute_to_locs_idxs()
        if mat.ndim == 1:
            return mat[index_arr]
        else:
            return mat[..., index_arr, :][..., :, index_arr]

    def get_equivalent_locs(self):
        """
        Get list of fake locations in the same order as original list of locations
        to which the compartment tree was fitted.

        Returns
        -------
        list of tuple
            Tuple has the form `(node.index, .5)`
        """
        loc_idxs = [node.loc_idx for node in self]
        index_arr = np.argsort(loc_idxs)
        locs_unordered = [(node.index, 0.5) for node in self]
        return [locs_unordered[ind] for ind in index_arr]

    def calc_impedance_matrix(
        self, freqs=0.0, channel_names=None, indexing="locs", use_conc=False
    ):
        """
        Constructs the impedance matrix of the model for each frequency provided
        in `freqs`. This matrix is evaluated at the equilibrium potentials
        stored in each node

        Parameters
        ----------
        freqs: np.array (dtype = complex) or float
            Frequencies at which the matrix is evaluated [Hz]
        channel_names: ``None`` (default) or `list` of `str`
            The channels to be included in the matrix. If ``None``, all
            channels present on the tree are included in the calculation
        use_conc: bool
            wheter or not to use the concentration dynamics
        indexing: 'tree' or 'locs'
            Whether the indexing order of the matrix corresponds to the tree
            nodes (order in which they occur in the iteration) or to the
            locations on which the reduced model is based

        Returns
        -------
            `np.ndarray` (ndim = 3, dtype = complex)
                The first dimension corresponds to the
                frequency, the second and third dimension contain the impedance
                matrix for that frequency
        """
        return np.linalg.inv(
            self.calc_system_matrix(
                freqs=freqs,
                channel_names=channel_names,
                indexing=indexing,
                use_conc=use_conc,
            )
        )

    def calc_impulse_response_matrix(
        self,
        t_inp,
        channel_names=None,
        indexing="locs",
        use_conc=False,
        compute_time_derivative=False,
        method: Literal["", "exp fit", "quadrature"] = "",
    ):
        """
        Computes the matrix of impulse response kernels at a given set of
        locations for all time-points defined in `self.ft.t` (the input times
        provided to `set_impedance()`).

        Parameters
        ----------
        t_inp : `np.array` (`ndim=1`, `dtype=real`)
            The time array at which the kernels are to be evaluated
        channel_names: ``None`` (default) or `list` of `str`
            The channels to be included in the matrix. If ``None``, all
            channels present on the tree are included in the calculation
        use_conc: bool
            wheter or not to use the concentration dynamics
        indexing: 'tree' or 'locs'
            Whether the indexing order of the matrix corresponds to the tree
            nodes (order in which they occur in the iteration) or to the
            locations on which the reduced model is based
        compute_time_derivative: bool
            if ``True``, also returns the time derivatives of the kernels
        method: str ("", "exp fit", "quadrature")
            The method to use when computing the kernel. "quadrature" for
            explicit integration of the inverse Fourrier integral, "exp fit" for
            a frequency domain fit with the Fourrier transforms of time domain
            exponentials, or "" choses the most appropriate method based on the
            case

        Returns
        -------
        `np.ndarray` (``ndim = 3``)
            the matrix of impulse responses, first dimension corresponds to the
            time axis, second and third dimensions contain the impulse response
            in ``[MOhm/ms]`` at that time point

        """
        ft = ke.FourierTools(t_inp)

        zf_mat = self.calc_impedance_matrix(
            freqs=ft.freqs,
            channel_names=channel_names,
            indexing=indexing,
            use_conc=use_conc,
        )

        nt = len(ft.t)  # number of time points
        nl = len(self)  # number of compartments
        zt_mat = np.zeros((nt, nl, nl))
        if compute_time_derivative:
            dzt_dt_mat = np.zeros((nt, nl, nl))

        for ii in range(len(self)):
            for jj in range(len(self)):

                if jj > ii:
                    break

                if compute_time_derivative:
                    zt_mat[:, ii, jj], dzt_dt_mat[:, ii, jj] = ft.inverse_fourier(
                        zf_mat[:, ii, jj],
                        compute_time_derivative=True,
                        method=method,
                    )
                    dzt_dt_mat[:, jj, ii] = dzt_dt_mat[:, ii, jj]

                else:
                    zt_mat[:, ii, jj] = ft.inverse_fourier(
                        zf_mat[:, ii, jj],
                        compute_time_derivative=False,
                        method=method,
                    )
                zt_mat[:, jj, ii] = zt_mat[:, ii, jj]

        if compute_time_derivative:
            return zt_mat, dzt_dt_mat
        else:
            return zt_mat

    def calc_conductance_matrix(self, indexing="locs"):
        """
        Constructs the conductance matrix of the model

        Returns
        -------
        `np.ndarray` (``dtype = float``, ``ndim = 2``)
            the conductance matrix
        """
        g_mat = np.zeros((len(self), len(self)))
        for node in self:
            ii = node.index
            g_mat[ii, ii] += node.calc_g_tot(self.channel_storage) + node.g_c
            if node.parent_node is not None:
                jj = node.parent_node.index
                g_mat[jj, jj] += node.g_c
                g_mat[ii, jj] -= node.g_c
                g_mat[jj, ii] -= node.g_c
        if indexing == "locs":
            return self._permuteToLocs(g_mat)
        elif indexing == "tree":
            return g_mat
        else:
            raise ValueError(
                "invalid argument for `indexing`, " + "has to be 'tree' or 'locs'"
            )

    def calc_system_matrix(
        self,
        freqs=0.0,
        channel_names=None,
        with_ca=True,
        use_conc=False,
        ep_shape=None,
        indexing="locs",
    ):
        """
        Constructs the matrix of conductance and capacitance terms of the model
        for each frequency provided in ``freqs``. this matrix is evaluated at
        the equilibrium potentials stored in each node

        Parameters
        ----------
            freqs: np.array (dtype = complex) or float (default ``0.``)
                Frequencies at which the matrix is evaluated [Hz]
            channel_names: `None` (default) or `list` of `str`
                The channels to be included in the matrix. If `None`, all
                channels present on the tree are included in the calculation
            with_ca: `bool`
                Whether or not to include the capacitive currents
            use_conc: `bool`
                wheter or not to use the concentration dynamics
            indexing: 'tree' or 'locs'
                Whether the indexing order of the matrix corresponds to the tree
                nodes (order in which they occur in the iteration) or to the
                locations on which the reduced model is based

        Returns
        -------
            `np.ndarray` (``ndim = 3, dtype = complex``)
                The first dimension corresponds to the
                frequency, the second and third dimension contain the impedance
                matrix for that frequency
        """
        if channel_names is None:
            channel_names = ["L"] + list(self.channel_storage.keys())

        # ensure that shapes are compatible
        freqs = np.array(freqs)
        if ep_shape is None:
            ep_shape = freqs.shape
        assert np.broadcast(freqs, np.empty(ep_shape)).shape == ep_shape

        s_mat = np.zeros(ep_shape + (len(self), len(self)), dtype=freqs.dtype)
        for node in self:
            ii = node.index

            # set the capacitance contribution
            if with_ca:
                s_mat[..., ii, ii] += freqs * node.ca

            # set the coupling conductances
            s_mat[..., ii, ii] += node.g_c
            if node.parent_node is not None:
                jj = node.parent_node.index
                s_mat[..., jj, jj] += node.g_c
                s_mat[..., ii, jj] -= node.g_c
                s_mat[..., jj, ii] -= node.g_c

            # set the ion channel contributions
            g_terms = node.calc_membrane_conductance_terms(
                self.channel_storage, freqs=freqs, channel_names=channel_names
            )
            s_mat[..., ii, ii] += sum(
                [
                    node.currents[c_name][0] * g_term
                    for c_name, g_term in g_terms.items()
                ]
            )

            # set the concentration contributions
            if use_conc:
                for ion, concmech in node.concmechs.items():
                    c_term = node.calc_membrane_concentration_terms(
                        ion,
                        self.channel_storage,
                        freqs=freqs,
                        channel_names=channel_names,
                    )
                    s_mat[..., ii, ii] += concmech.gamma * c_term

        if indexing == "locs":
            s_mat = self._permuteToLocs(s_mat)
        elif not indexing == "tree":
            raise ValueError(
                "invalid argument for `indexing`, " + "has to be 'tree' or 'locs'"
            )

        return s_mat

    def calc_eigenvalues(self, indexing="tree"):
        """
        Calculates the eigenvalues and eigenvectors of the passive system

        Returns
        -------
        np.ndarray (ndim = 1, dtype = complex)
            the eigenvalues
        np.ndarray (ndim = 2, dtype = complex)
            the right eigenvector matrix
        indexing: 'tree' or 'locs'
            Whether the indexing order of the matrix corresponds to the tree
            nodes (order in which they occur in the iteration) or to the
            locations on which the reduced model is based
        """
        # get the system matrix
        mat = self.calc_system_matrix(
            freqs=0.0, channel_names=["L"], with_ca=False, indexing=indexing
        )
        ca_vec = np.array([node.ca for node in self])
        if indexing == "locs":
            ca_vec = self._permuteToLocs(ca_vec)
        mat /= ca_vec[:, None]
        # compute the eigenvalues
        alphas, phimat = la.eig(mat)
        if max(np.max(np.abs(alphas.imag)), np.max(np.abs(phimat.imag))) < 1e-5:
            alphas = alphas.real
            phimat = phimat.real
        phimat_inv = la.inv(phimat)

        alphas /= -1e3
        phimat_inv /= ca_vec[None, :] * 1e3
        return alphas, phimat, phimat_inv

    def _calc_linear_system_matrix(self, channel_names=None):
        """
        Assume node indices correspond to their order in a depth-first iteration,
        i.e. by using `STree.reset_indices()`.
        """
        assert self.check_ordered()

        N_ = len(self)
        C_ = sum(
            [
                len(self.channel_storage[cname].statevars)
                for node in self
                for cname in node.currents.keys()
                if cname != "L"
            ]
        )

        if channel_names is None:
            channel_names = ["L"] + list(self.channel_storage.keys())

        V2V = np.zeros((N_, N_))
        Y2V = np.zeros((N_, C_))
        V2Y = np.zeros((C_, N_))
        Y2Y = np.zeros((C_, C_))

        self.root._add_linear_system_terms(
            0, V2V, Y2V, V2Y, Y2Y, self.channel_storage, channel_names=channel_names
        )

        return np.block([[V2V, Y2V], [V2Y, Y2Y]])

    def _preprocess_z_mat_arg(self, z_mat_arg):
        if isinstance(z_mat_arg, np.ndarray):
            return [self._permute_to_tree(z_mat_arg)]
        elif isinstance(z_mat_arg, list):
            return [self._permute_to_tree(z_mat) for z_mat in z_mat_arg]
        else:
            raise ValueError(
                "`z_mat_arg` has to be ``np.ndarray`` or list of " + "`np.ndarray`"
            )

    def _preprocess_e_eqs(self, e_eqs, w_e_eqs=None):
        # preprocess e_eqs argument
        if e_eqs is None:
            e_eqs = np.array([self.get_e_eq(indexing="tree")])
        if isinstance(e_eqs, float):
            e_eqs = np.array([e_eqs])
        elif isinstance(e_eqs, list) or isinstance(e_eqs, tuple):
            e_eqs = np.array(e_eqs)
        elif isinstance(e_eqs, np.ndarray):
            pass
        else:
            raise TypeError(
                "`e_eqs` has to be ``float`` or list or "
                + "``np.ndarray`` of ``floats`` or ``np.ndarray``"
            )
        # preprocess the w_e_eqs argument
        if w_e_eqs is None:
            w_e_eqs = np.ones_like(e_eqs)
        elif isinstance(w_e_eqs, float):
            w_e_eqs = np.array([e_eqs])
        elif isinstance(w_e_eqs, list) or isinstance(w_e_eqs, tuple):
            w_e_eqs = np.array(w_e_eqs)
        # check if arrays have the same shape
        assert w_e_eqs.shape[0] == e_eqs.shape[0]

        return e_eqs, w_e_eqs

    def _preprocess_freqs(self, freqs, w_freqs=None, z_mat_arg=None):
        if isinstance(freqs, float) or isinstance(freqs, complex):
            freqs = np.array([freqs])
        if w_freqs is None:
            w_freqs = np.ones_like(freqs)
        else:
            assert w_freqs.shape[0] == freqs.shape[0]
        # convert to 3d matrices if they are two dimensional
        z_mat_arg_ = []
        for z_mat in z_mat_arg:
            if z_mat.ndim == 2:
                z_mat_arg_.append(z_mat[np.newaxis, :, :])
            else:
                z_mat_arg_.append(z_mat)
            assert z_mat_arg_[-1].shape[0] == freqs.shape[0]
        z_mat_arg = z_mat_arg_
        return freqs, w_freqs, z_mat_arg

    def _to_structure_tensor_gmc(self, channel_names):
        g_vec = self._to_vec_gmc(channel_names)
        g_struct = np.zeros((len(self), len(self), len(g_vec)))
        kk = 0  # counter
        for node in self:
            ii = node.index
            g_terms = node.calc_membrane_conductance_terms(
                self.channel_storage, freqs=0.0, channel_names=["L"] + channel_names
            )
            if node.parent_node == None:
                # membrance conductance elements
                for channel_name in channel_names:
                    g_struct[0, 0, kk] += g_terms[channel_name]
                    kk += 1
            else:
                jj = node.parent_node.index
                # coupling conductance element
                g_struct[ii, jj, kk] -= 1.0
                g_struct[jj, ii, kk] -= 1.0
                g_struct[jj, jj, kk] += 1.0
                g_struct[ii, ii, kk] += 1.0
                kk += 1
                # membrance conductance elements
                for channel_name in channel_names:
                    g_struct[ii, ii, kk] += g_terms[channel_name]
                    kk += 1
        return g_struct

    def _to_vec_gmc(self, channel_names):
        """
        Place all conductances to be fitted in a single vector
        """
        g_list = []
        for node in self:
            if node.parent_node is None:
                g_list.extend([node.currents[c_name][0] for c_name in channel_names])
            else:
                g_list.extend(
                    [node.g_c] + [node.currents[c_name][0] for c_name in channel_names]
                )
        return np.array(g_list)

    def _to_tree_gmc(self, g_vec, channel_names):
        kk = 0  # counter
        for ii, node in enumerate(self):
            if node.parent_node is None:
                for channel_name in channel_names:
                    node.currents[channel_name][0] = g_vec[kk]
                    kk += 1
            else:
                node.g_c = g_vec[kk]
                kk += 1
                for channel_name in channel_names:
                    node.currents[channel_name][0] = g_vec[kk]
                    kk += 1

    def _to_structure_tensor_gm(self, freqs, channel_names, all_channel_names=None):
        freqs = np.array(freqs)
        # to construct appropriate channel vector
        if all_channel_names is None:
            all_channel_names = channel_names
        else:
            assert set(channel_names).issubset(all_channel_names)
        g_vec = self._to_vec_gm(all_channel_names)
        g_struct = np.zeros(
            (len(freqs), len(self), len(self), len(g_vec)), dtype=freqs.dtype
        )
        # fill the fit structure
        kk = 0  # counter
        for node in self:
            ii = node.index
            g_terms = node.calc_membrane_conductance_terms(
                self.channel_storage, freqs=freqs, channel_names=channel_names
            )
            # membrance conductance elements
            for channel_name in all_channel_names:
                if channel_name in channel_names:
                    g_struct[:, ii, ii, kk] += g_terms[channel_name]
                kk += 1
        return g_struct

    def _to_vec_gm(self, channel_names):
        """
        Place all conductances to be fitted in a single vector
        """
        g_list = []
        for node in self:
            g_list.extend([node.currents[c_name][0] for c_name in channel_names])
        return np.array(g_list)

    def _to_tree_gm(self, g_vec, channel_names):

        kk = 0  # counter
        for ii, node in enumerate(self):
            for channel_name in channel_names:
                # leack conductance is not allowed to be zero
                # if channel_name == 'L':
                #     g_vec[kk] = max(g_vec[kk], 1e-8)

                node.currents[channel_name][0] = g_vec[kk]
                kk += 1

    def _to_structure_tensor_conc(self, ion, freqs, channel_names, ep_shape):
        # to construct appropriate channel vector
        c_struct = np.zeros(
            ep_shape + (len(self), len(self), len(self)), dtype=freqs.dtype
        )
        # fill the fit structure
        for node in self:
            ii = node.index

            c_term = node.calc_membrane_concentration_terms(
                ion, self.channel_storage, freqs=freqs, channel_names=channel_names
            )
            c_struct[..., ii, ii, ii] += c_term

        return c_struct

    def _to_structure_tensor_conc(
        self, ion, freqs, channel_names, ep_shape, fit_type="gamma"
    ):
        if fit_type == "gamma":
            # to construct appropriate channel vector
            c_terms = np.zeros(ep_shape + (len(self),), dtype=freqs.dtype)
            for node in self:
                ii = node.index

                c_term = node.calc_membrane_concentration_terms(
                    ion,
                    self.channel_storage,
                    freqs=freqs,
                    channel_names=channel_names,
                    fit_type=fit_type,
                )
                c_terms[..., ii] = c_term

            return c_terms

        elif fit_type == "tau":
            # construct conductance vectors for fit
            c_terms0 = np.zeros(ep_shape + (len(self),), dtype=freqs.dtype)
            c_terms1 = np.zeros(ep_shape + (len(self),), dtype=freqs.dtype)
            for node in self:
                ii = node.index

                c0, c1 = node.calc_membrane_concentration_terms(
                    ion,
                    self.channel_storage,
                    freqs=freqs,
                    channel_names=channel_names,
                    fit_type=fit_type,
                )
                c_terms0[..., ii], c_terms1[..., ii] = c0, c1

            return c_terms0, c_terms1

    def _to_vec_conc(self, ion, return_type="gamma"):
        """
        Place concentration mechanisms to be fitted in a single vector
        """
        if return_type == "gamma":
            return np.array([node.concmechs[ion].gamma for node in self])
        elif return_type == "tau":
            return np.array([node.concmechs[ion].tau for node in self])

    def _to_tree_conc(self, c_vec, ion, param_type):
        if param_type == "tau":

            for ii, node in enumerate(self):
                node.concmechs[ion].gamma *= node.concmechs[ion].tau / c_vec[ii]
                node.concmechs[ion].tau = c_vec[ii]

        elif param_type == "gamma":

            for ii, node in enumerate(self):
                node.concmechs[ion].gamma = c_vec[ii]

        else:
            raise NotImplementedError("param_type should be 'tau' or 'gamma'")

    def _to_structure_tensor_c(self, freqs):
        freqs = np.array(freqs)
        c_vec = self._to_vec_c()
        c_struct = np.zeros(
            (len(freqs), len(self), len(self), len(c_vec)), dtype=complex
        )
        for node in self:
            ii = node.index
            # capacitance elements
            c_struct[:, ii, ii, ii] += freqs
        return c_struct

    def _to_vec_c(self):
        return np.array([node.ca for node in self])

    def _to_tree_c(self, c_vec):
        for ii, node in enumerate(self):
            node.ca = c_vec[ii]

    def compute_gmc(self, z_mat_arg, e_eqs=None, channel_names=["L"]):
        """
        Fit the models' membrane and coupling conductances to a given steady
        state impedance matrix.

        Parameters
        ----------
        z_mat_arg: np.ndarray (ndim = 2, dtype = float or complex) or
                   list of np.ndarray (ndim = 2, dtype = float or complex)
            If a single array, represents the steady state impedance matrix,
            If a list of arrays, represents the steady state impedance
            matrices for each equilibrium potential in ``e_eqs``
        e_eqs: np.ndarray (ndim = 1, dtype = float) or float
            The equilibirum potentials in each compartment for each
            evaluation of ``z_mat``
        channel_names: list of string (defaults to ['L'])
            Names of the ion channels that have been included in the impedance
            matrix calculation and for whom the conductances are fit. Default is
            only leak conductance
        """
        z_mat_arg = self._preprocess_z_mat_arg(z_mat_arg)
        e_eqs, _ = self._preprocess_e_eqs(e_eqs)
        assert len(z_mat_arg) == len(e_eqs)
        # do the fit
        mats_feature = []
        vecs_target = []
        for z_mat, e_eq in zip(z_mat_arg, e_eqs):
            # set equilibrium conductances
            self.set_e_eq(e_eq)
            # create the matrices for linear fit
            g_struct = self._to_structure_tensor_gmc(channel_names)
            tensor_feature = np.einsum("ij,jkl->ikl", z_mat, g_struct)
            tshape = tensor_feature.shape
            mat_feature_aux = np.reshape(
                tensor_feature, (tshape[0] * tshape[1], tshape[2])
            )
            vec_target_aux = np.reshape(np.eye(len(self)), (len(self) * len(self),))
            mats_feature.append(mat_feature_aux)
            vecs_target.append(vec_target_aux)
        mat_feature = np.concatenate(mats_feature, 0)
        vec_target = np.concatenate(vecs_target)
        # linear regression fit
        res = la.lstsq(mat_feature, vec_target)
        res = res[0].real
        # coupling and leak conductances are not allowed to be below zero
        g_vec = np.maximum(res, 0.0)
        # set the conductances
        self._to_tree_gmc(g_vec, channel_names)

    def compute_g_channels(
        self,
        channel_names,
        z_mat,
        e_eq,
        freqs,
        sv=None,
        weight=1.0,
        all_channel_names=None,
        other_channel_names=None,
        action="store",
    ):
        """
        Fit the conductances of multiple channels from the given impedance
        matrices, or store the feature matrix and target vector for later use
        (see `action`).

        Parameters
        ----------
        channel_names: list of str
            The names of the ion channels whose conductances are to be fitted
        z_mat: np.ndarray (ndim=3)
            The impedance matrix to which the ion channel is fitted. Shape is
            ``(F, N, N)`` with ``N`` the number of compartments and ``F`` the
            number of frequencies at which the matrix is evaluated
        e_eq: float
            The equilibirum potential at which the impedance matrix was computed
        freqs: np.array
            The frequencies at which `z_mat` is computed (shape is ``(F,)``)
        sv: dict {channel_name: np.ndarray} (optional)
            The state variable expansion point. If ``np.ndarray``, assumes it is
            the expansion point of the channel that is fitted. If dict, the
            expansion points of multiple channels can be specified. An empty dict
            implies the asymptotic points derived from the equilibrium potential
        weight: float
            The relative weight of the feature matrices in this part of the fit
        all_channel_names: list of str or ``None``
            The names of all channels whose conductances will be fitted in a
            single linear least squares fit
        other_channel_names: list of str or ``None`` (default)
            List of channels present in `z_mat`, but whose conductances are
            already fitted. If ``None`` and 'L' is not in `all_channel_names`,
            sets `other_channel_names` to 'L'
        action: 'fit', 'store' or 'return'
            If 'fit', fits the conductances for this feature matrix and target
            vector for directly; only based on `z_mat`; nothing is stored.
            If 'store', stores the feature matrix and target vector to fit later
            on. Relative weight in fit will be determined by `weight`.
            If 'return', returns the feature matrix and target vector. Nothing
            is stored
        """
        # to construct appropriate channel vector
        if all_channel_names is None:
            all_channel_names = channel_names
        else:
            assert set(channel_names).issubset(all_channel_names)
        if other_channel_names is None and "L" not in all_channel_names:
            other_channel_names = ["L"]
        if sv is None:
            sv = {}

        z_mat = self._permute_to_tree(z_mat)
        if isinstance(freqs, float):
            freqs = np.array([freqs])
        # set equilibrium conductances
        self.set_e_eq(e_eq)
        # set channel expansion point
        self.set_expansion_points(sv)
        # feature matrix
        g_struct = self._to_structure_tensor_gm(
            freqs=freqs,
            channel_names=channel_names,
            all_channel_names=all_channel_names,
        )
        tensor_feature = np.einsum("oij,ojkl->oikl", z_mat, g_struct)
        tshape = tensor_feature.shape
        mat_feature = np.reshape(
            tensor_feature, (tshape[0] * tshape[1] * tshape[2], tshape[3])
        )
        # target vector
        g_mat = self.calc_system_matrix(
            freqs, channel_names=other_channel_names, indexing="tree"
        )
        zg_prod = np.einsum("oij,ojk->oik", z_mat, g_mat)
        mat_target = np.eye(len(self))[np.newaxis, :, :] - zg_prod
        vec_target = np.reshape(mat_target, (tshape[0] * tshape[1] * tshape[2],))

        return self._fit_res_action(
            action, mat_feature, vec_target, weight, channel_names=all_channel_names
        )

    def compute_g_single_channel(
        self,
        channel_name,
        z_mat,
        e_eq,
        freqs,
        sv=None,
        weight=1.0,
        all_channel_names=None,
        other_channel_names=None,
        action="store",
    ):
        """
        Fit the conductances of a single channel from the given impedance
        matrices, or store the feature matrix and target vector for later use
        (see `action`).

        Parameters
        ----------
        channel_name: str
            The name of the ion channel whose conductances are to be fitted
        z_mat: np.ndarray (ndim=3)
            The impedance matrix to which the ion channel is fitted. Shape is
            ``(F, N, N)`` with ``N`` the number of compartments and ``F`` the
            number of frequencies at which the matrix is evaluated
        e_eq: float
            The equilibirum potential at which the impedance matrix was computed
        freqs: np.array
            The frequencies at which `z_mat` is computed (shape is ``(F,)``)
        sv: dict or nested dict of float or np.array, or None (default)
            The state variable expansion point. If simple dict, assumes it is
            the expansion point of the channel that is fitted. If nested dict, the
            expansion points of multiple channels can be specified. ``None``
            implies the asymptotic point derived from the equilibrium potential
        weight: float
            The relative weight of the feature matrices in this part of the fit
        all_channel_names: list of str or ``None``
            The names of all channels whose conductances will be fitted in a
            single linear least squares fit
        other_channel_names: list of str or ``None`` (default)
            List of channels present in `z_mat`, but whose conductances are
            already fitted. If ``None`` and 'L' is not in `all_channel_names`,
            sets `other_channel_names` to 'L'
        action: 'fit', 'store' or 'return'
            If 'fit', fits the conductances for this feature matrix and target
            vector for directly; only based on `z_mat`; nothing is stored.
            If 'store', stores the feature matrix and target vector to fit later
            on. Relative weight in fit will be determined by `weight`.
            If 'return', returns the feature matrix and target vector. Nothing
            is stored
        """
        # to construct appropriate channel vector
        if all_channel_names is None:
            all_channel_names = [channel_name]
        else:
            assert channel_name in all_channel_names
        if other_channel_names is None and "L" not in all_channel_names:
            other_channel_names = ["L"]

        z_mat = self._permute_to_tree(z_mat)
        if isinstance(freqs, float):
            freqs = np.array([freqs])
        if sv is None or not isinstance(list(sv.items())[0], dict):
            # if it is not a nested dict, make nested dict
            sv = {channel_name: sv}
        # set equilibrium conductances
        self.set_e_eq(e_eq)
        # set channel expansion point
        self.set_expansion_points(sv)
        # feature matrix
        g_struct = self._to_structure_tensor_gm(
            freqs=freqs,
            channel_names=[channel_name],
            all_channel_names=all_channel_names,
        )
        tensor_feature = np.einsum("oij,ojkl->oikl", z_mat, g_struct)
        tshape = tensor_feature.shape
        mat_feature = np.reshape(
            tensor_feature, (tshape[0] * tshape[1] * tshape[2], tshape[3])
        )
        # target vector
        g_mat = self.calc_system_matrix(
            freqs, channel_names=other_channel_names, indexing="tree"
        )
        zg_prod = np.einsum("oij,ojk->oik", z_mat, g_mat)
        mat_target = np.eye(len(self))[np.newaxis, :, :] - zg_prod
        vec_target = np.reshape(mat_target, (tshape[0] * tshape[1] * tshape[2],))

        self.remove_expansion_points()

        return self._fit_res_action(
            action, mat_feature, vec_target, weight, channel_names=all_channel_names
        )

    def _set_expansion_points(self, expansion_points):
        """
        Set the choice for the state variables of the ion channel around which
        to linearize.

        Note that when adding an ion channel to the tree, the default expansion
        point setting is to linearize around the asymptotic values for the state
        variables at the equilibrium potential store in `self.e_eq`.
        Hence, this function only needs to be called to change that setting.

        Parameters
        ----------
        expansion_points: dict {`channel_name`: ``None`` or dict}
            dictionary with as keys `channel_name` the name of the ion channel
            and as value its expansion point
        """
        if expansion_points is None:
            expansion_points = {}

        for channel_name, expansion_point in expansion_points.items():
            for node in self:
                node.set_expansion_point(channel_name, expansion_point)

    def compute_c(self, alphas, phimat, weights=None, tau_eps=5.0):
        """
        Fit the capacitances to the eigenmode expansion

        Parameters
        ----------
        alphas: np.ndarray of float or complex (shape=(K,))
            The eigenmode inverse timescales (1/s)
        phimat: np.ndarray of float or complex (shape=(K,C))
            The eigenmode vectors (C the number of compartments)
        weights: np.ndarray (shape=(K,)) or None
            The weights given to each eigenmode in the fit
        """
        alphas = alphas.real
        phimat = phimat.real
        n_c, n_a = len(self), len(alphas)
        assert phimat.shape == (n_a, n_c)
        if weights is None:
            weights = np.ones_like(alphas)
        else:
            weights = weights.real
        # construct the passive conductance matrix
        g_mat = -self.calc_system_matrix(
            freqs=0.0, channel_names=["L"], with_ca=False, indexing="tree"
        )

        # set lower limit for capacitance, fit not always well conditioned
        g_tot = np.array(
            [
                node.calc_g_tot(self.channel_storage, channel_names=["L"])
                for node in self
            ]
        )
        c_lim = g_tot / (-alphas[0] * tau_eps)
        gamma_mat = alphas[:, None] * phimat * c_lim[None, :]

        # construct feature matrix and target vector
        mat_feature = np.zeros((n_a * n_c, n_c))
        vec_target = np.zeros(n_a * n_c)
        for ii, node in enumerate(self):
            mat_feature[ii * n_a : (ii + 1) * n_a, ii] = (
                alphas * phimat[:, ii] * weights
            )
            vec_target[ii * n_a : (ii + 1) * n_a] = (
                np.reshape(
                    np.dot(phimat, g_mat[ii : ii + 1, :].T) - gamma_mat[:, ii : ii + 1],
                    n_a,
                )
                * weights
            )

        # least squares fit
        res = so.nnls(mat_feature, vec_target)[0]
        c_vec = res + c_lim

        self._to_tree_c(c_vec)

    # def fit_capacitances_with_matrix_exp(
    #     self, times: np.ndarray, target_kernels: np.ndarray,
    #     lr=1e-2, steps=1000, device="cpu",
    #     return_loss_history=False, early_stopping=False, patience=50, tol=1e-6
    # ):
    #     """
    #     Fit compartment capacitances to match input resistance kernels
    #     from GreensTreeTime using matrix exponentials.

    #     Parameters
    #     ----------
    #     times : np.ndarray, shape (T,)
    #         Time points at which kernels are evaluated.
    #     target_kernels : np.ndarray, shape (T, n_comp)
    #         Target diagonal kernels from GreensTreeTime
    #         (response at i from injection at i).
    #     lr : float
    #         Learning rate for optimizer.
    #     steps : int
    #         Maximum number of optimization steps.
    #     device : str
    #         PyTorch device ("cpu" or "cuda").
    #     return_loss_history : bool, default False
    #         If True, also return the list of loss values over optimization.
    #     early_stopping : bool, default False
    #         If True, stop optimization early if the loss does not improve.
    #     patience : int, default 50
    #         Number of steps to wait for improvement before stopping.
    #     tol : float, default 1e-6
    #         Minimum improvement in loss to reset patience.

    #     Returns
    #     -------
    #     kernels_pred : np.ndarray, shape (T, n_comp)
    #         Predicted fitted diagonal kernels.
    #     loss_history : list of float, optional
    #         Loss values at each optimization step (only if return_loss_history=True).
    #     """
    #     import torch
    #     n_comp = len(self)

    #     # Convert inputs to torch tensors
    #     times = torch.tensor(times, dtype=torch.float32, device=device)
    #     target_kernels = torch.tensor(target_kernels, dtype=torch.float32, device=device)

    #     # Initial capacitance vector from tree
    #     c_init = self._to_c_vec()  # numpy array (n_comp,)
    #     log_C = torch.nn.Parameter(
    #         torch.log(torch.tensor(c_init, dtype=torch.float32, device=device))
    #     )
    #     optimizer = torch.optim.Adam([log_C], lr=lr)

    #     # Use NEAT's conductance matrix function
    #     G_np = self.calc_conductance_matrix()   # numpy array (n_comp, n_comp)
    #     G = torch.tensor(G_np, dtype=torch.float32, device=device)

    #     loss_history = []
    #     best_loss = float('inf')
    #     steps_since_improvement = 0

    #     for step in range(steps):
    #         optimizer.zero_grad()

    #         # Capacitances
    #         C = torch.exp(log_C)
    #         Cinv = torch.diag(1.0 / C)

    #         # System matrix
    #         A = -Cinv @ G

    #         # Vectorized computation of exp(A * t) for all times
    #         At = A.unsqueeze(0) * times[:, None, None]  # (T, n_comp, n_comp)
    #         expAt = torch.matrix_exp(At)                # (T, n_comp, n_comp)

    #         # Kernel prediction: exp(At) @ C^{-1}, take diagonals
    #         kernels_pred = (expAt @ Cinv).diagonal(dim1=1, dim2=2)  # (T, n_comp)

    #         # Loss: mean squared error vs diagonal target kernels
    #         loss = torch.mean((kernels_pred - target_kernels) ** 2)
    #         loss.backward()
    #         optimizer.step()

    #         # Track loss
    #         loss_value = loss.item()
    #         loss_history.append(loss_value)

    #         if step % max(1, steps // 10) == 0:
    #             print(f"Step {step}, Loss {loss_value:.6e}")

    #         # Early stopping logic
    #         if early_stopping:
    #             if loss_value + tol < best_loss:
    #                 best_loss = loss_value
    #                 steps_since_improvement = 0
    #             else:
    #                 steps_since_improvement += 1
    #                 if steps_since_improvement >= patience:
    #                     print(f"Early stopping at step {step}, loss did not improve for {patience} steps.")
    #                     break

    #     # Write optimized capacitances back into tree
    #     C_opt = C.detach().cpu().numpy()
    #     self._to_tree_c(C_opt)

    #     # Return fitted kernels and optionally loss history
    #     kernels_pred_np = kernels_pred.detach().cpu().numpy()
    #     if return_loss_history:
    #         return kernels_pred_np, loss_history
    #     else:
        
    def _fit_res_action(
            self, action, mat_feature, vec_target, weight, ca_lim=[], **kwargs
        ):
        if action == "fit":
            res = np.linalg.lstsq(mat_feature, vec_target, rcond=None)
            vec_res = res[0].real
            vec_res = np.maximum(vec_res, 0.0)
            # set the conductances
            if "channel_names" in kwargs:
                self._to_tree_gm(vec_res, channel_names=kwargs["channel_names"])
            elif "ion" in kwargs:
                self._to_tree_conc(
                    vec_res, kwargs["ion"], param_type=kwargs["param_type"]
                )
            else:
                raise IOError("Provide 'channel_names' or 'ion' as keyword argument")
        elif action == "return":
            return mat_feature, vec_target
        elif action == "store":
            if "channel_names" in kwargs:
                try:
                    assert self.fit_data["ion"] == ""
                except AssertionError:
                    raise IOError(
                        "Stored fit matrices are concentration mech fits, "
                        + "do not try to store channel conductance fit matrices"
                    )
                if len(self.fit_data["channel_names"]) == 0:
                    self.fit_data["channel_names"] = kwargs["channel_names"]
                else:
                    try:
                        assert self.fit_data["channel_names"] == kwargs["channel_names"]
                    except AssertionError:
                        raise IOError(
                            "`channel_names` does not agree with stored "
                            + "channel names for other fits\n"
                            + "`channel_names`:      "
                            + str(kwargs["channel_names"])
                            + "\nstored channel names: "
                            + str(self.fit_data["channel_names"])
                        )
            elif "ion" in kwargs:
                try:
                    assert len(self.fit_data["channel_names"]) == 0
                except AssertionError:
                    raise IOError(
                        "Stored fit matrices are channel conductance fits, "
                        + "do not try to store concentration fit matrices"
                    )
                if self.fit_data["ion"] == "":
                    self.fit_data["ion"] = kwargs["ion"]
                else:
                    try:
                        assert self.fit_data["ion"] == kwargs["ion"]
                    except AssertionError:
                        raise IOError(
                            "`ion` does not agree with stored ion for "
                            + "other fits:\n"
                            + "`ion`: "
                            + kwargs["ion"]
                            + "\nstored ion: "
                            + self.fit_data["ion"]
                        )

            self.fit_data["mats_feature"].append(mat_feature)
            self.fit_data["vecs_target"].append(vec_target)
            self.fit_data["weights_fit"].append(weight)
        else:
            raise IOError("Undefined action, choose 'fit', 'return' or 'store'.")

    def reset_fit_data(self):
        """
        Delete all stored feature matrices and and target vectors.
        """
        self.fit_data = dict(
            mats_feature=[], vecs_target=[], weights_fit=[], channel_names=[], ion=""
        )

    def run_fit(self):
        """
        Run a linear least squares fit for the conductances concentration
        mechanisms. The obtained conductances are stored on each node. All
        stored feature matrices and and target vectors are deleted.
        """
        fit_data = self.fit_data
        if len(fit_data["mats_feature"]) > 0:
            # apply the weights
            for m_f, v_t, w_f in zip(
                fit_data["mats_feature"],
                fit_data["vecs_target"],
                fit_data["weights_fit"],
            ):
                nn = len(v_t)
                m_f *= w_f / nn
                v_t *= w_f / nn
            # create the fit matrices
            mat_feature = np.concatenate(fit_data["mats_feature"])
            vec_target = np.concatenate(fit_data["vecs_target"])
            # do the fit
            if len(fit_data["channel_names"]) > 0:
                self._fit_res_action(
                    "fit",
                    mat_feature,
                    vec_target,
                    1.0,
                    channel_names=fit_data["channel_names"],
                )
            elif fit_data["ion"] != "":
                self._fit_res_action(
                    "fit", mat_feature, vec_target, 1.0, ion=fit_data["ion"]
                )
            # reset fit data
            self.reset_fit_data()
        else:
            warnings.warn(
                "No fit matrices are stored, no fit has been performed", UserWarning
            )

    def compute_fake_geometry(
        self,
        fake_c_m=1.0,
        fake_r_a=100.0 * 1e-6,
        factor_r_a=1e-6,
        delta=1e-14,
        method=2,
    ):
        """
        Computes a fake geometry so that the neuron model is a reduced
        compartmental model

        Parameters
        ----------
        fake_c_m: float [uF / cm^2]
            fake membrane capacitance value used to compute the surfaces of
            the compartments
        fake_r_a: float [MOhm * cm]
            fake axial resistivity value, used to evaluate the lengths of each
            section to yield the correct coupling constants

        Returns
        -------
        radii, lengths: np.array of floats [cm]
            The radii, lengths, resp. surfaces for the section in NEURON. Array
            index corresponds to NEURON index

        Raises
        ------
        AssertionError
            If the node indices are not ordered consecutively when iterating
        """
        assert self.check_ordered()
        factor_r = 1.0 / np.sqrt(factor_r_a)
        # compute necessary vectors for calculating
        surfaces = np.array([node.ca / fake_c_m for node in self])
        vec_coupling = np.array(
            [1.0] + [1.0 / node.g_c for node in self if node.parent_node is not None]
        )
        if method == 1:
            # find the 3d points to construct the segments' geometry
            p0s = -surfaces
            p1s = np.zeros_like(p0s)
            p2s = np.pi * (factor_r**2 - 1.0) * np.ones_like(p0s)
            p3s = 2.0 * np.pi**2 * vec_coupling / fake_r_a * (1.0 + factor_r)
            # find the polynomial roots
            points = []
            for ii, (p0, p1, p2, p3) in enumerate(zip(p0s, p1s, p2s, p3s)):
                res = np.roots([p3, p2, p1, p0])
                # compute radius and length of first half of section
                radius = res[np.where(res.real > 0.0)[0][0]].real
                radius *= 1e4  # convert [cm] to [um]
                length = (
                    np.pi * radius**2 * vec_coupling[ii] / (fake_r_a * 1e4)
                )  # convert [MOhm*cm] to [MOhm*um]
                # compute the pt3d points
                point0 = [0.0, 0.0, 0.0, 2.0 * radius]
                point1 = [length, 0.0, 0.0, 2.0 * radius]
                point2 = [length * (1.0 + delta), 0.0, 0.0, 2.0 * radius * factor_r]
                point3 = [length * (2.0 + delta), 0.0, 0.0, 2.0 * radius * factor_r]
                points.append([point0, point1, point2, point3])

            return points, surfaces
        elif method == 2:
            radii = np.cbrt(fake_r_a * surfaces / (vec_coupling * (2.0 * np.pi) ** 2))
            lengths = surfaces / (2.0 * np.pi * radii)
            return lengths, radii
        else:
            raise ValueError("Invalid `method` argument, should be 1 or 2")

    def plot_dendrogram(
        self,
        ax,
        plotargs={},
        labelargs={},
        textargs={},
        nodelabels={},
        bbox=None,
        y_max=None,
    ):
        """
        Generate a dendrogram of the NET

        Parameters
        ----------
            ax: `matplotlib.axes`
                the axes object in which the plot will be made
            plotargs : dict (string : value)
                keyword args for the matplotlib plot function, specifies the
                line properties of the dendrogram
            labelargs : dict (string : value)
                keyword args for the matplotlib plot function, specifies the
                marker properties for the node points. Or dict with keys node
                indices, and with values dicts with keyword args for the
                matplotlib function that specify the marker properties for
                specific node points. The entry under key -1 specifies the
                properties for all nodes not explicitly in the keys.
            textargs : dict (string : value)
                keyword args for matplotlib textproperties
            nodelabels: dict (int: string) or None
                labels of the nodes. If None, nodes are named by default
                according to their location indices. If empty dict, no labels
                are added.
            y_max: int, float or None
                specifies the y-scale. If None, the scale is computed from
                ``self``. By default, y=1 is added for each child of a node, so
                if y_max is smaller than the depth of the tree, part of it will
                not be plotted
        """
        # get the number of leafs to determine the dendrogram spacing
        rnode = self.root
        n_branch = self.degree_of_node(rnode)
        l_spacing = np.linspace(0.0, 1.0, n_branch + 1)
        if y_max is None:
            y_max = np.max([self.depth_of_node(n) for n in self.leafs]) + 1.5
        y_min = 0.5
        # plot the dendrogram
        self._expand_dendrogram(
            rnode,
            0.5,
            None,
            0.0,
            l_spacing,
            y_max,
            ax,
            plotargs=plotargs,
            labelargs=labelargs,
            textargs=textargs,
            nodelabels=nodelabels,
            bbox=bbox,
        )
        # limits
        ax.set_ylim((y_min, y_max))
        ax.set_xlim((0.0, 1.0))

        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
        # ax.axison = False

        return y_max

    def _expand_dendrogram(
        self,
        node,
        x0,
        xprev,
        y0,
        l_spacing,
        y_max,
        ax,
        plotargs={},
        labelargs={},
        textargs={},
        nodelabels={},
        bbox=None,
    ):
        # impedance of layer
        ynew = y0 + 1.0
        # plot vertical connection line
        # ax.vlines(x0, y0, ynew, **plotargs)
        if xprev is not None:
            ax.plot([xprev, x0], [y0, ynew], **plotargs)
        # get the child nodes for recursion
        l0 = 0
        for i, cnode in enumerate(node.child_nodes):
            # attribute space on xaxis
            deg = self.degree_of_node(cnode)
            l1 = l0 + deg
            # new quantities
            xnew = (l_spacing[l0] + l_spacing[l1]) / 2.0
            # horizontal connection line limits
            if i == 0:
                xnew0 = xnew
            if i == len(node.child_nodes) - 1:
                xnew1 = xnew
            # recursion
            self._expand_dendrogram(
                cnode,
                xnew,
                x0,
                ynew,
                l_spacing[l0 : l1 + 1],
                y_max,
                ax,
                plotargs=plotargs,
                labelargs=labelargs,
                textargs=textargs,
                nodelabels=nodelabels,
                bbox=None,
            )
            # next index
            l0 = l1
        # add label and maybe text annotation to node
        if node.index in labelargs:
            ax.plot([x0], [ynew], **labelargs[node.index])
        elif -1 in labelargs:
            ax.plot([x0], [ynew], **labelargs[-1])
        else:
            try:
                ax.plot([x0], [ynew], **labelargs)
            except TypeError as e:
                pass
        if textargs:
            if nodelabels != None:
                if node.index in nodelabels:
                    if labelargs == {}:
                        ax.plot([x0], [ynew], **nodelabels[node.index][1])
                        ax.annotate(
                            nodelabels[node.index][0],
                            xy=(x0, ynew),
                            xytext=(x0 + 0.06, ynew),  # +y_max*0.04),
                            bbox=bbox,
                            **textargs,
                        )
                    else:
                        ax.annotate(
                            nodelabels[node.index],
                            xy=(x0, ynew),
                            xytext=(x0 + 0.06, ynew),  # +y_max*0.04),
                            bbox=bbox,
                            **textargs,
                        )
            else:
                ax.annotate(
                    r"$N=" + "".join([str(ind) for ind in node.loc_idxs]) + "$",
                    xy=(x0, ynew),
                    xytext=(x0 + 0.06, ynew),  # +y_max*0.04),
                    bbox=bbox,
                    **textargs,
                )
