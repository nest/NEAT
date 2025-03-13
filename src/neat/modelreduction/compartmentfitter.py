# -*- coding: utf-8 -*-
#
# compartmentfitter.py
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

import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from ..trees.stree import STree
from ..trees.phystree import PhysTree
from ..trees.compartmenttree import CompartmentTree
from ..tools.kernelextraction import Kernel
from ..channels.ionchannels import SPDict
from ..factorydefaults import FitParams, MechParams
from .cachetrees import CachedGreensTree, CachedSOVTree, EquilibriumTree

import copy
import pathlib
import warnings


def _statevar_is_activating(f_statevar):
    """
    check whether a statevar is activating or inactivating

    Parameters
    ----------
    f_statevar: callable
        the activation function of the state variable
    """
    # test voltage values to check whether state variable is activating or
    # inactivating
    v_test = np.array([-43.22, -32.22])

    sv_test = f_statevar(v_test)
    return sv_test[0] < sv_test[1]


def _get_two_variable_holding_potentials(e_hs):
    e_hs_aux_act = list(e_hs)
    e_hs_aux_inact = list(e_hs)
    for ii, e_h1 in enumerate(e_hs):
        for jj, e_h2 in enumerate(e_hs):
            e_hs_aux_act.append(e_h1)
            e_hs_aux_inact.append(e_h2)
    e_hs_aux_act = np.array(e_hs_aux_act)
    e_hs_aux_inact = np.array(e_hs_aux_inact)

    return e_hs_aux_act, e_hs_aux_inact


def get_expansion_points(e_hs, channel, only_e_h=False):
    """
    Returns a list of expansion points around which to compute the impedance
    matrix given a set of holding potentials. If the channel has only one state
    variable, the returned expansion points are at the holding potentials, if
    the channels has two state variables, the returned expansions points are
    are different combinations of the state variable values around the holding
    potentials

    Parameters
    ----------
    e_hs: iterable collection
        The holding potentials around which the expansion points are computed
    channel: `neat.channels.ionchannels.IonChannel`
        The ion channels
    only_e_h: bool
        Only applicable for channels with at least two state variables.
        If True, returned expansion points are always for state variable
        combination evaluated at the same holding potential. Otherwise,
        state variable activations are evaluated at different holding potentials.

    Returns
    -------
    sv_hs: dict
        the expansion points at every holding potential
    """
    if len(channel.statevars) == 1 or only_e_h:
        sv_hs = channel.compute_varinf(e_hs)
        sv_hs["v"] = e_hs
    else:
        # create different combinations of holding potentials
        e_hs_aux_act, e_hs_aux_inact = _get_two_variable_holding_potentials(e_hs)

        sv_hs = SPDict(v=e_hs_aux_act)
        for svar, f_inf in channel.f_varinf.items():
            # check if variable is activation
            if _statevar_is_activating(f_inf):  # variable is activation
                sv_hs[str(svar)] = f_inf(e_hs_aux_act)
            else:  # variable is inactivation
                sv_hs[str(svar)] = f_inf(e_hs_aux_inact)

    return sv_hs


class CompartmentFitter(EquilibriumTree):
    """
    Tree class that streamlines fitting reduced compartmental models

    Attributes
    ----------
    tree: `neat.PhysTree`
        The full tree based on which reductions are made
    fit_cfg: `neat.FitParams`
        The fit parameters
    concmech_cfg: `neat.MechParams`
        The concentration mechanisms parameters
    model_fits: dict of `{str: dict}`
        Data structure with already performed model fits, where keys are the provided names.
        Each entry is a dict of the form
        `{'ctree': neat.CompartmentTree, 'locs': list of neat.MorphLoc}`
    cache_name: str (default '')
        name of files in which intermediate trees required for the fit are
        cached.
    cache_path: str (default '')
        specify a path under which the intermediate files are cached.
        Default is empty string, which means that intermediate files are stored
        in the working directory.
    save_cache: bool (default `True`)
        Save the intermediate results in a cache (using `cache_path` and
        `cache_name`).
    recompute_cache: bool (default `False`)
        Forces recomputing the caches.
    """

    def __init__(self, *args, fit_cfg=None, concmech_cfg=None, **kwargs):
        if (
            len(args) == 0
            or isinstance(args[0], str)
            or isinstance(args[0], pathlib.Path)
            or (
                issubclass(type(args[0]), STree)
                and not issubclass(type(args[0]), PhysTree)
            )
        ):
            call_post_init_in_contructor = False
            # if the initialization argument is not provided (empty tree),
            # or if it is a .swc-filename string,
            # or if it is a tree class that is likely to require further build operations after
            # calling this constructor, we do not call `post_init()` in this constructor, but
            # raise a warning that it has to be called manually
            warnings.warn(
                f"Initialization of a {self.__class__.__name__}"
                f"-instance as a tree that still has to be built, "
                f"be sure to call `{self.__class__.__name__}.post_init()` after building the tree."
            )
        else:
            call_post_init_in_contructor = True

        self.fitted_models = {}

        self.fit_cfg = None
        self.concmech_cfg = None

        super().__init__(*args, **kwargs)

        if self.fit_cfg is None:
            self.fit_cfg = FitParams()
        elif fit_cfg is not None:
            self.fit_cfg = fit_cfg
        if self.concmech_cfg is None:
            self.concmech_cfg = MechParams()
        elif concmech_cfg is not None:
            self.concmech_cfg = concmech_cfg

        if call_post_init_in_contructor:
            self.post_init()

        # boolean flag that is reset the first time `self.fit_passive` is called
        self.use_all_channels_for_passive = True

    def set_cfg(self, fit_cfg=None, concmech_cfg=None):
        self.fit_cfg = fit_cfg
        if fit_cfg is None:
            self.fit_cfg = FitParams()

        self.concmech_cfg = concmech_cfg
        if concmech_cfg is None:
            self.concmech_cfg = MechParams()

    def post_init(self):
        with self.as_original_tree:
            # set the equilibrium potentials in the tree
            self.set_e_eq(pprint=True)

    def get_attributes_excluded_from_cache_override(self):
        """
        Returns a list of attributes that should NOT be overwritten by the cashed tree

        Returns
        -------
        list of str
            Attribute names that should not be overwritten
        """
        return super().get_attributes_excluded_from_cache_override() + [
            "fit_cfg",
            "concmech_cfg",
        ]

    def convert_fit_arg(self, fit_arg):
        """
        Convert a fit argument, which can be a tuple, dict or tuple, to a tuple
        consisting of a `neat.CompartmentTree` that is either fitted, or in the
        process of being fitted, and the corresponding list of locations.

        Parameters
        ----------
        fit_arg : string, dict, or tuple
            If string, the provided argument is interpreted as the fit name.
            If dict, the provided argument is a dictionary of the form
            `{'ctree': neat.CompartmentTree, 'locs': <list of locations>}`.
            If tuple, the provided argument is a tuple of the form
            `(neat.CompartmentTree, <list of locations>}`.

        Returns
        -------
        `neat.CompartmentTree`
            The compartmenttree that is (in the process of being) fitted.
        list of <neat.MorphLoc>
            The corresponding list of fit locations.

        Raises
        ------
        TypeError
            If `fit_arg` does not correspond to one of the above described arguments.
        """
        if isinstance(fit_arg, str):
            return (
                self.fitted_models[fit_arg]["ctree"],
                self.fitted_models[fit_arg]["locs"],
            )
        elif isinstance(fit_arg, dict):
            return fit_arg["ctree"], fit_arg["locs"]
        elif issubclass(type(fit_arg[0]), CompartmentTree):
            return fit_arg[0], fit_arg[1]
        else:
            raise TypeError(
                "Invalid type for `fit_arg`, should be string, "
                "dict with {'ctree': neat.CompartmentTree, 'locs': list of locations}, "
                "or a tuple of (neat.CompartmentTree, list of locations)"
            )

    def _store_fit(self, ctree, locs, fit_name=""):
        if len(fit_name) > 0:
            self.store_locs(locs, name=fit_name)
            self.fitted_models[fit_name] = {
                "ctree": ctree,
                "locs": self.get_locs(name=fit_name),
                "complete": False,
            }

    def remove_fit(self, fit_name):
        try:
            del self.fitted_models[fit_name]
        except KeyError:
            warnings.warn(f"Fit with name '{fit_name}' not in stored fits.")
        self.remove_locs(fit_name)

    def set_ctree(self, loc_arg, fit_name="", extend_w_bifurc=True, pprint=False):
        """
        Store an initial `neat.CompartmentTree`, providing a tree
        structure scaffold for the fit for a given set of locations. The
        locations are also stored on ``self`` under the name 'fit locs'

        Parameters
        ----------
        loc_arg: list of locations or string (see documentation of
                :func:`MorphTree.convert_loc_arg_to_locs` for details)
            The compartment locations
        fit_name: str (optional, default: '')
            The name of the fit. If provided, the resulting
            `neat.CompartmentTree` and list of fit locations will be
            stored. They can be accessed under the `fitted_models` attribute
            of `neat.CompartmentFitter`.
        extend_w_bifurc: bool (optional, default `True`)
            To extend the compartment locations with all intermediate
            bifurcations (see documentation of
            :func:`MorphTree.extend_with_bifurcation_locs`).
        pprint: bool
            whether to print additional info


        Returns
        -------
        `neat.CompartmentTree`
            The compartmenttree that is in the process of being fitted.
        list of <neat.MorphLoc>
            The corresponding list of fit locations.
        """
        locs = self.convert_loc_arg_to_locs(loc_arg)
        if extend_w_bifurc:
            locs = self.extend_with_bifurcation_locs(locs)
        else:
            warnings.warn(
                "Not adding bifurcations to `loc_arg`, this could "
                "lead to inaccurate fits. To add bifurcation, set"
                "kwarg `extend_w_bifurc` to ``True``"
            )
        # create the reduced compartment tree
        ctree = self.create_compartment_tree(locs)
        # store the fit
        self._store_fit(ctree, locs, fit_name=fit_name)

        # add currents to compartmental model
        for c_name, channel in self.channel_storage.items():
            e_revs = []
            for node in self:
                if c_name in node.currents:
                    e_revs.append(node.currents[c_name][1])
            # reversal potential is the same throughout the reduced model
            ctree.add_channel_current(copy.deepcopy(channel), np.mean(e_revs))

        for node in ctree:
            loc_idx = node.loc_idx
            concmechs = self[locs[loc_idx]["node"]].concmechs

            # try to set default parameters as the ones from the original tree
            # if the concmech is not present at the corresponding location,
            # use the default parameters
            for ion in self.ions:
                if ion in concmechs:
                    cparams = {pname: pval for pname, pval in concmechs[ion].items()}
                    node.add_conc_mech(ion, **cparams)
                else:
                    node.add_conc_mech(ion, **self.concmech_cfg.exp_conc_mech)

        return ctree, locs

    def create_tree_gf(
        self,
        channel_names=[],
        cache_name_suffix="",
        unmasked_nodes=None,
    ):
        """
        Create a `CachedGreensTree` copy of the original tree, but only with the
        channels in ``channel_names``. Leak 'L' is included in the tree by
        default.

        Parameters
        ----------
        channel_names: list of strings
            List of channel names of the channels that are to be included in the
            new tree.
        recompute_cache: bool
            Whether or not to force recompute the impedance caches
        unmasked_nodes: 'node_arg' (see documentation of `MorphTree.convert_node_arg_to_nodes`)
            The nodes where the channels in `channel_names` will be initialized
            to non-zero values

        Returns
        -------
        `CachedGreensTree()`
        """
        unmasked_node_indices = [
            node.index for node in self.convert_node_arg_to_nodes(unmasked_nodes)
        ]

        # create new tree and empty channel storage
        tree = CachedGreensTree(
            self,
            cache_path=self.cache_path,
            cache_name=self.cache_name + cache_name_suffix,
            save_cache=self.save_cache,
            recompute_cache=self.recompute_cache,
        )
        tree.channel_storage = {}
        # add the ion channel to the tree
        channel_names_newtree = set()
        for node, node_orig in zip(tree, self):
            node.currents = {}
            g_l, e_l = node_orig.currents["L"]
            # add the current to the tree
            node._add_current("L", g_l, e_l)

            if node.index not in unmasked_node_indices:
                continue

            for channel_name in channel_names:
                try:
                    g_max, e_rev = node_orig.currents[channel_name]
                    node._add_current(channel_name, g_max, e_rev)
                    channel_names_newtree.add(channel_name)
                except KeyError:
                    pass

        tree.channel_storage = {
            channel_name: self.channel_storage[channel_name]
            for channel_name in channel_names_newtree
        }
        tree.set_comp_tree(eps=self.fit_cfg.fit_comptree_eps)

        return tree

    def _eval_channel(self, fit_arg, channel_name, pprint=False):
        """
        Evaluate the impedance matrix for the model restricted to a single ion
        channel type.

        Parameters
        ----------
        fit_arg: see docstring of `CompartmentFitter.convert_fit_args()`
            Specifying the fit that is being performed.
        channel_name: string
            The name of the ion channel under consideration
        pprint:  bool (optional, defaults to ``False``)
            whether to print information

        Return
        ------
        fit_mats
            list of fit matrices
        """
        ctree, locs = self.convert_fit_arg(fit_arg)
        # find the expansion point parameters for the channel
        channel = self.channel_storage[channel_name]
        sv_h = get_expansion_points(self.fit_cfg.e_hs, channel)

        # create the trees with only a single channel and multiple expansion points
        fit_tree = self.create_tree_gf(
            [channel_name],
            cache_name_suffix=f"_{channel_name}_",
        )
        # set the impedances in the tree
        fit_tree.set_impedances_in_tree(
            freqs=self.fit_cfg.freqs, sv_h={channel_name: sv_h}, pprint=pprint
        )
        # compute the impedance matrix for this activation level
        z_mats = fit_tree.calc_impedance_matrix(locs)[None, :, :, :]

        # compute the fit matrices for all holding potentials
        fit_mats = []
        for ii, e_h in enumerate(sv_h["v"]):
            sv = SPDict(
                {
                    str(svar): sv_h[svar][ii]
                    for svar in channel.statevars
                    if str(svar) != "v"
                }
            )

            # compute the fit matrices
            m_f, v_t = ctree.compute_g_single_channel(
                channel_name,
                z_mats[:, ii, :, :],
                e_h,
                np.array([self.fit_cfg.freqs]),
                sv=sv,
                other_channel_names=["L"],
                all_channel_names=[channel_name],
                action="return",
            )

            # compute open probability to weigh fit matrices
            po_h = channel.compute_p_open(e_h, **sv)
            w_f = 1.0 / po_h

            fit_mats.append([m_f, v_t, w_f])

        # fit the model for this channel
        w_norm = 1.0 / np.sum([w_f for _, _, w_f in fit_mats])
        for _, _, w_f in fit_mats:
            w_f /= w_norm

        # store the fit matrices
        for m_f, v_t, w_f in fit_mats:
            if not (np.isnan(m_f).any() or np.isnan(v_t).any() or np.isnan(w_f).any()):
                ctree._fit_res_action(
                    "store", m_f, v_t, w_f, channel_names=[channel_name]
                )

        # run the fit
        ctree.run_fit()

        return fit_mats

    def fit_channels(self, fit_arg, pprint=False):
        """
        Fit the active ion channel parameters

        Parameters
        ----------
        fit_arg: see docstring of `CompartmentFitter.convert_fit_args()`
            Specifying the fit that is being performed.
        pprint:  bool (optional, defaults to ``False``)
            whether to print information

        Returns
        -------
        `neat.CompartmentTree`
            The compartmenttree that is in the process of being fitted.
        list of <neat.MorphLoc>
            The corresponding list of fit locations.
        """
        for channel_name in self.get_channels_in_tree():
            self._eval_channel(fit_arg, channel_name, pprint=pprint)

        return self.convert_fit_arg(fit_arg)

    def _calibrate_conc_mechs(self, ion, orig_node, comp_node):
        """
        Set the `gamma` factor of the concentration mechanism based on the ratio
        of fitted conducances over original conductances permeable to the
        associated ion

        Parameters
        ----------
        orig_node: `neat.PhysNode`
            the original node corresponding to the location of the compartment
        comp_node: `neat.CompartmentNode`
            the fitted compartment node
        """
        if ion not in orig_node.concmechs:
            return

        channel_storage = self.channel_storage
        currents_orig = copy.deepcopy(orig_node.currents)
        currents_comp = copy.deepcopy(comp_node.currents)

        # compute g_max for the ion
        g_ion_orig, g_ion_comp = 0.0, 0.0
        for cname in orig_node.currents:

            if cname in channel_storage and ion == channel_storage[cname].ion:
                g_ion_orig += currents_orig.pop(cname, [0.0, 0.0])[0]
                g_ion_comp += currents_comp.pop(cname, [0.0, 0.0])[0]

        try:
            comp_node.concmechs[ion].gamma = (
                orig_node.concmechs[ion].gamma * g_ion_orig / g_ion_comp
            )

        except ZeroDivisionError:
            # no Ca current so we rescale based on leak
            # maybe concmech can be removed at this node?
            g_l_orig = orig_node.currents["L"][0]
            g_l_comp = comp_node.currents["L"][0]
            comp_node.concmechs[ion].gamma = (
                orig_node.concmechs[ion].gamma * g_l_orig / g_l_comp
            )

    def fit_concentration(self, fit_arg, ion):
        """
        Fits the concentration mechanisms parameters associate with the `ion`
        ion type.

        Parameters
        ----------
        fit_arg: see docstring of `CompartmentFitter.convert_fit_args()`
            Specifying the fit that is being performed.
        ion: str
            The ion type that is to be fitted (e.g. 'ca').

        Returns
        -------
        `neat.CompartmentTree`
            The compartmenttree that is in the process of being fitted.
        list of <neat.MorphLoc>
            The corresponding list of fit locations.
        """
        ctree, locs = self.convert_fit_arg(fit_arg)

        has_concmech = False
        for node in self:
            if ion in node.concmechs:
                has_concmech = True
                break
        if not has_concmech:
            return 0

        orig_nodes = [self[loc["node"]] for loc in locs]
        comp_nodes = ctree.get_nodes_from_loc_idxs(list(range(len(locs))))

        for orig_node, comp_node in zip(orig_nodes, comp_nodes):
            self._calibrate_conc_mechs(ion, orig_node, comp_node)

        return ctree, locs

    def fit_passive(self, fit_arg, use_all_channels=True, pprint=False):
        """
        Fit the steady state passive model, consisting only of leak and coupling
        conductances, but ensure that the coupling conductances takes the passive
        opening of all channels into account

        Parameters
        ----------
        fit_arg: see docstring of `CompartmentFitter.convert_fit_args()`
            Specifying the fit that is being performed.
        use_all_channels: bool (optional)
            use leak at rest of all channels combined in the passive fit (passive
            leak has to be refit after capacitance fit)
        pprint:  bool (optional, defaults to ``False``)
            whether to print information

        Returns
        -------
        `neat.CompartmentTree`
            The compartmenttree that is in the process of being fitted.
        list of <neat.MorphLoc>
            The corresponding list of fit locations.
        """
        ctree, locs = self.convert_fit_arg(fit_arg)
        self.use_all_channels_for_passive = use_all_channels

        suffix = "_pas_"
        if use_all_channels:
            suffix = f"_passified_"

        if use_all_channels:
            fit_tree = EquilibriumTree(self)
            fit_tree.set_cache_params(
                cache_path=self.cache_path,
                cache_name=self.cache_name + "_eq" + suffix,
                save_cache=self.save_cache,
                recompute_cache=self.recompute_cache,
            )
            # set the channels to passive
            fit_tree.as_passive_membrane()
            # convert to a greens tree for further evaluation
            fit_tree = CachedGreensTree(
                fit_tree,
                cache_path=self.cache_path,
                cache_name=self.cache_name + "_gf" + suffix,
                save_cache=self.save_cache,
                recompute_cache=self.recompute_cache,
            )
            fit_tree.set_comp_tree(eps=self.fit_cfg.fit_comptree_eps)
        else:
            fit_tree = self.create_tree_gf(
                [],  # empty list of channel to include
                cache_name_suffix=suffix,
            )

        # set the impedances in the tree
        fit_tree.set_impedances_in_tree(freqs=0.0, pprint=pprint)
        # compute the steady state impedance matrix
        z_mat = fit_tree.calc_impedance_matrix(locs)
        # fit the coupling+leak conductances to steady state impedance matrix
        ctree.compute_gmc(z_mat, channel_names=["L"])

        # print passive impedance matrices
        if pprint:
            z_mat_fit = ctree.calc_impedance_matrix(channel_names=["L"])
            np.set_printoptions(precision=2, edgeitems=10, linewidth=500, suppress=True)
            print("\n----- Impedance matrix comparison -----")
            print("> Zmat orig =")
            print(z_mat)
            print("> Zmat fit  =")
            print(z_mat_fit)
            print("> Zmat diff =")
            print(z_mat - z_mat_fit)
            print("---------------------------------------\n")
            # restore defaults
            np.set_printoptions(precision=8, edgeitems=3, linewidth=75, suppress=False)

        return ctree, locs

    def fit_leak_only(self, fit_arg, pprint=True):
        """
        Fit leak only. Coupling conductances have to have been fit already.

        Parameters
        ----------
        fit_arg: see docstring of `CompartmentFitter.convert_fit_args()`
            Specifying the fit that is being performed.
        pprint:  bool (optional, defaults to ``False``)
            whether to print information

        Returns
        -------
        `neat.CompartmentTree`
            The compartmenttree that is in the process of being fitted.
        list of <neat.MorphLoc>
            The corresponding list of fit locations.
        """
        ctree, locs = self.convert_fit_arg(fit_arg)
        # compute the steady state impedance matrix
        fit_tree = self.create_tree_gf(
            [],
            cache_name_suffix="_only_leak_",
        )
        # set the impedances in the tree
        fit_tree.set_impedances_in_tree(self.fit_cfg.freqs, pprint=pprint)
        # compute the steady state impedance matrix
        z_mat = fit_tree.calc_impedance_matrix(locs)[None, :, :]
        # fit the conductances to steady state impedance matrix
        ctree.compute_g_single_channel(
            "L",
            z_mat,
            -75.0,
            np.array([self.fit_cfg.freqs]),
            other_channel_names=[],
            action="fit",
        )
        # print passive impedance matrices
        if pprint:
            z_mat_fit = ctree.calc_impedance_matrix(channel_names=["L"])
            np.set_printoptions(precision=2, edgeitems=10, linewidth=500, suppress=True)
            print("\n----- Impedance matrix comparison -----")
            print("> Zmat orig =")
            print(z_mat)
            print("> Zmat fit  =")
            print(z_mat_fit)
            print("> Zmat diff =")
            print(z_mat - z_mat_fit)
            print("---------------------------------------\n")
            # restore defaults
            np.set_printoptions(precision=8, edgeitems=3, linewidth=75, suppress=False)

        return ctree, locs

    def create_tree_sov(self):
        """
        Create a `SOVTree` copy of the old tree

        Parameters
        ----------
        channel_names: list of strings
            List of channel names of the channels that are to be included in the
            new tree

        Returns
        -------
        `neat.tools.fittools.compartmentfitter.CachedSOVTree`
        """
        if self.use_all_channels_for_passive:
            cache_name_suffix = "_SOV_allchans_"
        else:
            cache_name_suffix = "_SOV_only_leak_"

        # create new tree and empty channel storage
        tree = CachedSOVTree(
            self,
            cache_path=self.cache_path,
            cache_name=self.cache_name + cache_name_suffix,
            save_cache=self.save_cache,
            recompute_cache=self.recompute_cache,
        )
        if not self.use_all_channels_for_passive:
            tree.channel_storage = {}

            for node, node_orig in zip(tree, self):
                node.currents = {}
                g_l, e_l = node_orig.currents["L"]
                # add the current to the tree
                node._add_current("L", g_l, e_l)

        # set the computational tree
        tree.set_comp_tree(eps=self.fit_cfg.fit_comptree_eps)

        return tree

    def _calc_sov_mats(self, locs, pprint=False):
        """
        Use a `neat.SOVTree` to compute SOV matrices for fit
        """
        # create an SOV tree
        sov_tree = self.create_tree_sov()
        # compute the SOV expansion for this tree
        sov_tree.set_sov_in_tree(pprint=pprint)
        # get SOV constants
        alphas, phimat, importance = sov_tree.get_important_modes(
            loc_arg=locs, sort_type="importance", eps=1e-12, return_importance=True
        )
        alphas = alphas.real
        phimat = phimat.real

        return alphas, phimat, importance, sov_tree

    def fit_capacitance(
        self,
        fit_arg,
        inds=[0],
        check_fit=True,
        force_tau_m_fit=False,
        pprint=False,
        pplot=False,
    ):
        """
        Fit the capacitances of the model to the largest SOV time scale

        Parameters
        ----------
        fit_arg: see docstring of `CompartmentFitter.convert_fit_args()`
            Specifying the fit that is being performed.
        inds: list of int (optional, defaults to ``[0]``)
            indices of eigenmodes used in the fit. Default is [0], indicating
            the largest eigenmode
        check_fit: bool (optional, default ``True``)
            Check whether the largest eigenmode of the reduced model is within
            tolerance of the largest eigenmode of the full tree. If not,
            capacitances are set to mach membrane time scale
        force_tau_m_fit: bool (optional, default ``False``)
            force capacitance fit through membrance time scale matching
        pprint:  bool (optional, defaults to ``False``)
            whether to print information
        pplot: bool (optional, defaults to ``False``)
            whether to plot the eigenmode timescales

        Returns
        -------
        `neat.CompartmentTree`
            The compartmenttree that is in the process of being fitted.
        list of <neat.MorphLoc>
            The corresponding list of fit locations.
        """
        ctree, locs = self.convert_fit_arg(fit_arg)

        # compute SOV matrices for fit
        try:
            alphas, phimat, importance, sov_tree = self._calc_sov_mats(locs, pprint=pprint)
            # fit the capacitances from SOV time-scales
            ctree.compute_c(-alphas[inds] * 1e3, phimat[inds, :], weights=importance[inds])

            def calcTau():
                nm = len(locs)
                # original timescales
                taus_orig = np.sort(np.abs(1.0 / alphas))[::-1][:nm]
                # fitted timescales
                lambdas, _, _ = ctree.calc_eigenvalues()
                taus_fit = np.sort(np.abs(1.0 / lambdas))[::-1]

                return taus_orig, taus_fit

            taus_orig, taus_fit = calcTau()

            if check_fit:
                fit_not_sane = np.abs(taus_fit[0] - taus_orig[0]) > 0.8 * taus_orig[0]
            else:
                fit_not_sane = False

        except Exception as e:
            if pprint:
                print(f"Issue in SOV calculations:\n{e}\n> reverting to membrane timescale matching")
            sov_tree = self.create_tree_sov()
            fit_not_sane = True

        def calcTauM():
            clocs = [locs[n.loc_idx] for n in ctree]
            # original membrane time scales
            taus_m = []
            for l in clocs:
                g_m = sov_tree[l[0]].calc_g_tot(
                    channel_storage=sov_tree.channel_storage
                )
                taus_m.append(self[l[0]].c_m / g_m * 1e3)
            taus_m_orig = np.array(taus_m)
            # fitted membrance time scales
            taus_m_fit = (
                np.array([node.ca / node.currents["L"][0] for node in ctree]) * 1e3
            )

            return taus_m_orig, taus_m_fit

        if fit_not_sane or force_tau_m_fit:

            taus_m_orig, taus_m_fit = calcTauM()
            # if fit was not sane, revert to more basic membrane timescale match
            for ii, node in enumerate(ctree):
                node.ca = node.currents["L"][0] * taus_m_orig[ii] * 1e-3

            warnings.warn(
                "No sane capacitance fit achieved for this configuragion,"
                + "reverted to more basic membrane time scale matching."
            )

        if pprint:
            # mode time scales
            taus_orig, taus_fit = calcTau()
            # membrane time scales
            taus_m_orig, taus_m_fit = calcTauM()

            np.set_printoptions(
                precision=2, edgeitems=10, linewidth=500, suppress=False
            )
            print("\n----- capacitances -----")
            print(("Ca (uF) =\n" + str([nn.ca for nn in ctree])))
            print("\n----- Eigenmode time scales -----")
            print(("> Taus original (ms) =\n" + str(taus_orig)))
            print(("> Taus fitted (ms) =\n" + str(taus_fit)))
            print("\n----- Membrane time scales -----")
            print(("> Tau membrane original (ms) =\n" + str(taus_m_orig)))
            print(("> Tau membrane fitted (ms) =\n" + str(taus_m_fit)))
            print("---------------------------------\n")
            # restore default print options
            np.set_printoptions(precision=8, edgeitems=3, linewidth=75, suppress=False)

        if pplot:
            self.plot_kernels(alphas, phimat)

        return ctree, locs

    def plot_sov(
        self, fit_arg, alphas=None, phimat=None, importance=None, n_mode=8, alphas2=None
    ):
        ctree, fit_locs = self.convert_fit_arg(fit_arg)

        if alphas is None or phimat is None or importance is None:
            alphas, phimat, importance, _ = self._calc_sov_mats(fit_locs, pprint=False)
        if alphas2 is None:
            alphas2, _, _ = ctree.calc_eigenvalues()

        colours = list(pl.rcParams["axes.prop_cycle"].by_key()["color"])
        loc_colours = np.array(
            [colours[ii % len(colours)] for ii in range(len(fit_locs))]
        )
        markers = Line2D.filled_markers

        pl.figure("SOV", figsize=(10, 10))
        gs = GridSpec(2, 2)
        ax1, ax2, ax3 = pl.subplot(gs[0, 0]), pl.subplot(gs[0, 1]), pl.subplot(gs[1, :])
        # x axis modes
        x_arr = np.arange(n_mode)
        x_loc = np.arange(len(fit_locs))
        # time scales
        ax1.semilogy(x_arr, np.abs(1.0 / alphas[:n_mode]), "rD--")
        if alphas2 is not None:
            ax1.semilogy(
                x_arr[: len(alphas2)], np.sort(np.abs(1.0 / alphas2))[::-1], "bo--"
            )
        ax1.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$\tau_k$ (ms)")
        # importance
        ax2.semilogy(x_arr, importance[:n_mode], "rD--")
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$I_k$")
        # spatial modes
        for kk in range(n_mode):
            ax3.plot(x_loc, phimat[kk, :], ls="--", c="DarkGrey")
            ax3.scatter(
                x_loc,
                phimat[kk, :],
                c=loc_colours,
                marker=markers[kk % len(markers)],
                label=r"" + str(kk),
            )
        ax3.set_xlabel(r"$x_i$")
        ax3.set_ylabel(r"$\phi_k(x_i)$")
        ax3.legend(loc=0)

    def _construct_kernels(self, nn, a, c):
        return [[Kernel((a, c[:, ii, jj])) for ii in range(nn)] for jj in range(nn)]

    def get_kernels(
        self,
        fit_arg,
        alphas=None,
        phimat=None,
        pprint=False,
    ):
        """
        Returns the impedance kernels as a double nested list of "neat.Kernel".
        The element at the position i,j represents the transfer impedance kernel
        between compartments i and j.

        If one of the `alphas` and or `phimat` are not provided, these SOV matrices
        are recomputed.

        Parameters
        ----------
        fit_arg: see docstring of `CompartmentFitter.convert_fit_args()`
            Specifying the compartmentree for which the kernels have to be computed.
        alphas: `np.array`
            The exponential coefficients, as follows from the SOV expansion
        phimat: `np.ndarray` (dim=2)
            The matrix to compute the exponential prefactors, as follows from
            the SOV expansion
        pprint: `bool`
            Is verbose if ``True``

        Returns
        -------
        k_orig: list of list of `neat.Kernel`
            The kernels of the full model
        k_comp: list of list of `neat.Kernel`
            The kernels of the reduced model (i.e. of the compartment tree)
        """
        ctree, locs = self.convert_fit_arg(fit_arg)
        if alphas is None or phimat is None:
            alphas, phimat, _, _ = self._calc_sov_mats(locs, pprint=pprint)
        nn = len(locs)
        # compute eigenvalues
        alphas_comp, phimat_comp, phimat_inv_comp = ctree.calc_eigenvalues(
            indexing="locs"
        )

        # get the kernels
        k_orig = self._construct_kernels(
            nn, alphas, np.einsum("ik,kj->kij", phimat.T, phimat)
        )
        k_comp = self._construct_kernels(
            nn, -alphas_comp, np.einsum("ik,kj->kij", phimat_comp, phimat_inv_comp)
        )

        return k_orig, k_comp

    def plot_kernels(
        self,
        fit_arg,
        alphas=None,
        phimat=None,
        t_arr=None,
    ):
        """
        Plots the impedance kernels.
        The kernel at the position i,j represents the transfer impedance kernel
        between compartments i and j.

        Parameters
        ----------
        fit_arg: see docstring of `CompartmentFitter.convert_fit_args()`
            Specifying the compartmentree for which the kernels have to be plotted.
        alphas: `np.array`
            The exponential coefficients, as follows from the SOV expansion
        phimat: `np.ndarray` (dim=2)
            The matrix to compute the exponential prefactors, as follows from
            the SOV expansion
        t_arr: `np.array`
            The time-points at which the to be plotted kernels are evaluated.
            Default is ``np.linspace(0.,200.,int(1e3))``
        """
        ctree, fit_locs = self.convert_fit_arg(fit_arg)
        nn = len(fit_locs)

        if alphas is None or phimat is None:
            alphas, phimat, _, _ = self._calc_sov_mats(fit_locs, pprint=False)

        k_orig, k_comp = self.get_kernels(ctree, alphas=alphas, phimat=phimat)

        if t_arr is None:
            t_arr = np.linspace(0.0, 200.0, int(1e3))

        pl.figure("Kernels", figsize=(2.0 * nn, 1.5 * nn))
        gs = GridSpec(nn, nn)
        gs.update(top=0.98, bottom=0.04, left=0.04, right=0.98)
        colours = list(pl.rcParams["axes.prop_cycle"].by_key()["color"])
        loc_colours = np.array(
            [colours[ii % len(colours)] for ii in range(len(fit_locs))]
        )

        for ii in range(nn):
            for jj in range(ii, nn):
                ko, kc = k_orig[ii][jj], k_comp[ii][jj]
                ax = pl.subplot(gs[ii, jj])
                ax.plot(t_arr, ko(t_arr), c="DarkGrey")
                ax.plot(t_arr, kc(t_arr), ls="--", c=loc_colours[jj])
                # limits
                ax.set_ylim((-0.5, 20.0))
                # kernel label
                pstring = r"%d $\leftrightarrow$ %d" % (ii, jj)
                ax.set_title(pstring, pad=-10)

    def _store_sov_mats(self):
        fit_locs = self.get_locs("fit locs")
        self.alphas, self.phimat, _, _ = self._calc_sov_mats(fit_locs, pprint=False)

    def kernel_objective(self, t_arr=None):
        fit_locs = self.get_locs("fit locs")
        nn = len(fit_locs)

        if t_arr is None:
            t_arr = np.concatenate(
                (np.logspace(-2, 0, 200), np.linspace(1.0, 200.0, 400)[1:])
            )

        k_orig, k_comp = self.get_kernels(alphas=self.alphas, phimat=self.phimat)

        res = 0.0
        for ii in range(nn):
            for jj in range(ii, nn):
                ko, kc = k_orig[ii][jj], k_comp[ii][jj]
                res += np.sum((ko(t_arr) - kc(t_arr)) ** 2)

        return res

    def check_passive(
        self,
        loc_arg,
        alpha_inds=[0],
        use_all_channels_for_passive=True,
        force_tau_m_fit=False,
        pprint=False,
    ):
        """
        Checks the impedance kernels of the passive model.

        Parameters
        ----------
        loc_arg: list of locations or string (see documentation of
                :func:`MorphTree.convert_loc_arg_to_locs` for details)
            The compartment locations
        alpha_inds: list of ints
            Indices of all mode time-scales to be included in the fit
        n_modes: int
            The number of eigen modes that are shown
        use_all_channels_for_passive: bool
            Uses all channels in the tree to compute coupling conductances
        force_tau_m_fit: bool
            Force using the local membrane time-scale for capacitance fit
        pprint: bool
            is verbose if ``True``
        """
        fit_arg = self.set_ctree(loc_arg)
        # fit the passive steady state model
        fit_arg = self.fit_passive(
            fit_arg, use_all_channels=use_all_channels_for_passive, pprint=pprint
        )
        # fit the capacitances
        fit_arg = self.fit_capacitance(
            fit_arg,
            inds=alpha_inds,
            force_tau_m_fit=force_tau_m_fit,
            pprint=pprint,
            pplot=False,
        )

        _, fit_locs = self.convert_fit_arg(fit_arg)
        colours = list(pl.rcParams["axes.prop_cycle"].by_key()["color"])
        loc_colours = np.array(
            [colours[ii % len(colours)] for ii in range(len(fit_locs))]
        )

        pl.figure("tree")
        ax = pl.gca()
        loc_args = [
            dict(marker="o", mec="k", mfc=lc, markersize=6.0) for lc in loc_colours
        ]
        self.plot_2d_morphology(
            ax, marklocs=fit_locs, loc_args=loc_args, use_radius=False
        )

        pl.tight_layout()
        pl.show()

    def get_net(self, c_loc, locs, channel_names=[], pprint=False):
        greens_tree = self.create_tree_gf(
            channel_names=channel_names,
            cache_name_suffix="_for_NET_",
        )
        greens_tree.set_impedances_in_tree(self.fit_cfg.freqs, pprint=False)
        # create the NET
        net, z_mat = greens_tree.calc_net_steadystate(c_loc)
        net.improve_input_resistance(z_mat)

        # prune the NET to only retain ``locs``
        loc_idxs = greens_tree.get_nearest_loc_idxs([c_loc] + locs, "net eval")
        net_reduced = net.get_reduced_tree(loc_idxs, indexing="locs")

        return net_reduced

    def fit_e_eq(self, fit_arg):
        """
        Fits the leak potentials of the reduced model to yield the same
        equilibrium potentials as the full model

        Parameters
        ----------
        fit_arg: see docstring of `CompartmentFitter.convert_fit_args()`
            Specifying the fit that is being performed.
        loc_arg: `list` of locations or string
            if `list` of locations, specifies the locations at which to compute
            the equilibrium potentials, if ``string``, specifies the
            name under which a set of location is stored

        Returns
        -------
        `neat.CompartmentTree`
            The compartmenttree that is in the process of being fitted.
        list of <neat.MorphLoc>
            The corresponding list of fit locations.
        """
        ctree, locs = self.convert_fit_arg(fit_arg)

        # compute the equilibirum potentials at fit locations
        v_eqs_fit, conc_eqs_fit = self.calc_e_eq(locs)

        # set the equilibria
        ctree.set_e_eq(v_eqs_fit)
        for ion in self.ions:
            ctree.set_conc_eq(ion, conc_eqs_fit[ion])

        # fit the leak
        ctree.fit_e_leak()

        return ctree, locs

    def fit_model(
        self,
        loc_arg,
        fit_name="",
        alpha_inds=[0],
        use_all_channels_for_passive=True,
        pprint=False,
    ):
        """
        Runs the full fit for a set of locations (the location are automatically
        extended with the bifurcation locs)

        Parameters
        ----------
        loc_arg: list of locations or string (see documentation of
                :func:`MorphTree.convert_loc_arg_to_locs` for details)
            The compartment locations
        fit_name: string
            The name under which the fit will be stored. By default, the fit
            will not be stored.
        alpha_inds: list of ints
            Indices of all mode time-scales to be included in the fit
        use_all_channels_for_passive: bool (optional, default ``True``)
            Uses all channels in the tree to compute coupling conductances
        pprint:  bool
            whether to print information

        Returns
        -------
        `neat.CompartmentTree`
            The compartmenttree that is fitted.
        list of <neat.MorphLoc>
            The corresponding list of fit locations.
        """
        if fit_name == "":
            fit_name = "temp"

        fit_arg = self.set_ctree(loc_arg, fit_name=fit_name, pprint=pprint)

        # fit the passive steady state model
        fit_arg = self.fit_passive(
            fit_arg,
            pprint=pprint,
            use_all_channels=use_all_channels_for_passive,
        )
        # fit the capacitances
        fit_arg = self.fit_capacitance(
            fit_arg, inds=alpha_inds, pprint=pprint, pplot=False
        )
        # refit with only leak
        if use_all_channels_for_passive:
            fit_arg = self.fit_leak_only(fit_arg, pprint=pprint)

        # fit the ion channels
        fit_arg = self.fit_channels(fit_arg, pprint=pprint)

        # fit the concentration mechansims
        for ion in self.ions:
            fit_arg = self.fit_concentration(fit_arg, ion)

        # fit the resting potentials
        fit_arg = self.fit_e_eq(fit_arg)

        if fit_name == "temp":
            self.remove_fit(fit_name)
        else:
            self.fitted_models[fit_name]["complete"] = True

        return fit_arg

    def recalc_impedance_matrix(self, loc_arg, g_syns, channel_names=None):
        # process input
        locs = self.convert_loc_arg_to_locs(loc_arg)
        n_syn = len(locs)
        assert n_syn == len(g_syns)
        if n_syn == 0:
            return np.array([[]])
        if channel_names is None:
            channel_names = list(self.channel_storage.keys())
        suffix = "_".join(channel_names)

        # create a greenstree with equilibrium potentials at rest
        greens_tree = self.create_tree_gf(
            channel_names=channel_names,
            cache_name_suffix=f"_{'_'.join(channel_names)}_",
        )
        greens_tree.set_impedances_in_tree(self.fit_cfg.freqs, pprint=False)
        # compute the impedance matrix of the synapse locations
        z_mat = greens_tree.calc_impedance_matrix(locs, explicit_method=False)

        # compute the ZG matrix
        gd_mat = np.diag(g_syns)
        zg_mat = np.dot(z_mat, gd_mat)
        z_mat_ = np.linalg.solve(np.eye(n_syn) + zg_mat, z_mat)

        return z_mat_

    def fit_syn_rescale(
        self,
        c_loc_arg,
        s_loc_arg,
        comp_inds,
        g_syns,
        e_revs,
        fit_impedance=False,
        channel_names=None,
    ):
        """
        Computes the rescaled conductances when synapses are moved to compartment
        locations, assuming a given average conductance for each synapse.

        Parameters
        ----------
        c_loc_arg: list of locations or string (see documentation of
                  :func:`MorphTree.convert_loc_arg_to_locs` for details)
            The compartment locations
        s_loc_arg: list of locations or string (see documentation of
                  :func:`MorphTree.convert_loc_arg_to_locs` for details)
            The synapse locations
        comp_inds: list or numpy.array of ints
            for each location in [s_loc_arg], gives the index of the compartment
            location in [c_loc_arg] to which the synapse is assigned
        g_syns: list or numpy.array of floats
            The average conductances for each synapse
        e_revs: list or numpy.array of floats
            The reversal potential of each synapse
        fit_impdedance: bool (optional, default `False`)
            Whether to also use the reproduction of the rescaled impedance matrix
            as target.
        channel_names: list of str or `None` (default)
            List of ion channels to be included in impedance matrix calculation.
            `None` includes all ion channels

        Returns
        -------
        g_resc: numpy.array of floats
            The rescale values for the synaptic weights
        """
        # process input
        c_locs = self.convert_loc_arg_to_locs(c_loc_arg)
        s_locs = self.convert_loc_arg_to_locs(s_loc_arg)
        n_comp, n_syn = len(c_locs), len(s_locs)
        assert n_syn == len(g_syns) and n_syn == len(e_revs)
        assert len(c_locs) > 0
        if n_syn == 0:
            return np.array([])
        if channel_names is None:
            channel_names = list(self.channel_storage.keys())
        cs_locs = c_locs + s_locs
        cg_syns = np.concatenate((np.zeros(n_comp), np.array(g_syns)))
        comp_inds, g_syns, e_revs = (
            np.array(comp_inds),
            np.array(g_syns),
            np.array(e_revs),
        )

        # create a greenstree with equilibrium potentials at rest
        greens_tree = self.create_tree_gf(
            channel_names=channel_names,
            cache_name_suffix=f"_{'_'.join(channel_names)}_",
        )
        greens_tree.set_impedances_in_tree(self.fit_cfg.freqs, pprint=False)
        # compute the impedance matrix of the synapse locations
        z_mat = greens_tree.calc_impedance_matrix(cs_locs, explicit_method=False)
        zc_mat = z_mat[:n_comp, :n_comp]

        # get the reversal potentials of the synapse locations
        e_eqs = self.calc_e_eq(cs_locs)[0]
        e_cs = e_eqs[:n_comp]
        e_ss = e_eqs[-n_syn:]

        # compute the ZG matrix
        gd_mat = np.diag(cg_syns)
        zg_mat_ = np.dot(z_mat, gd_mat)
        zg_mat = np.linalg.solve(np.eye(n_comp + n_syn) + zg_mat_, zg_mat_)
        zg_mat = zg_mat[:n_comp, n_comp:]

        # create the compartment assignment matrix & syn index vector
        c_mat = np.array([comp_inds == cc for cc in range(n_comp)]).astype(int)
        s_inds = np.array([np.where(cc > 0)[0][0] for cc in c_mat.T])

        # compute the driving potential vectors
        es_vec = e_revs - e_ss
        ec_vec = e_revs - e_cs[s_inds]

        zc_mat = np.dot(zc_mat, c_mat)
        czg_mat = np.dot(c_mat.T, zg_mat)

        # create matrices for inverse fit
        a1_mat = np.einsum("ck,kn->cnk", zc_mat, np.diag(ec_vec))
        a2_mat = np.einsum("ck,kn->cnk", zc_mat, czg_mat * es_vec[None, :])
        b_mat = zg_mat * es_vec[None, :]

        # unravel first two indices
        a_mat = np.reshape(a1_mat - a2_mat, (n_syn * n_comp, -1))
        b_vec = np.reshape(b_mat, (n_syn * n_comp,))

        if fit_impedance:
            # fit based on impedance matrix
            zr_mat = np.linalg.solve(np.eye(n_comp + n_syn) + zg_mat_, z_mat)

            zr_mat = zr_mat[:n_comp, :n_comp]
            zc_mat = z_mat[:n_comp, :n_comp]

            # b matrix for fit
            b_mat = zc_mat - zr_mat
            # a tensor for fit
            zcc = np.dot(zc_mat, c_mat)
            czr = np.dot(c_mat.T, zr_mat)
            aa_mat = np.einsum("ik,kn->ink", zcc, czr)

            # unravel first two indices
            a_mat_ = np.reshape(aa_mat, (n_comp * n_comp, -1))
            b_vec_ = np.reshape(b_mat, (n_comp * n_comp,))

            # perfor mfit
            a_mat = np.concatenate((a_mat, a_mat_), axis=0)
            b_vec = np.concatenate((b_vec, b_vec_), axis=0)

        # compute rescaled synaptic conductances
        g_resc = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]

        b_arr = g_syns > 1e-9
        g_resc[np.logical_not(b_arr)] = 1.0
        g_resc[b_arr] = g_resc[b_arr] / g_syns[b_arr]

        return g_resc

    def assign_locs_to_comps(self, c_loc_arg, s_loc_arg, fz=0.8, channel_names=None):
        """
        assumes the root node is in `c_loc_arg`
        """
        if channel_names is None:
            channel_names = list(self.channel_storage.keys())

        # create a greenstree with equilibrium potentials at rest
        greens_tree = self.create_tree_gf(
            channel_names=channel_names,
            cache_name_suffix=f"_{'_'.join(channel_names)}_at_rest_",
        )
        greens_tree.set_impedances_in_tree(self.fit_cfg.freqs, pprint=False)

        # process input
        c_locs = self.convert_loc_arg_to_locs(c_loc_arg)
        s_locs = self.convert_loc_arg_to_locs(s_loc_arg)
        # find nodes corresponding to locs
        c_nodes = [self[loc["node"]] for loc in c_locs]
        s_nodes = [self[loc["node"]] for loc in s_locs]
        # compute input impedances
        c_zins = [greens_tree.calc_zf(c_loc, c_loc)[0] for c_loc in c_locs]
        s_zins = [greens_tree.calc_zf(s_loc, s_loc)[0] for s_loc in s_locs]
        # paths to root
        c_ptrs = [self.path_to_root(node) for node in c_nodes]
        s_ptrs = [self.path_to_root(node) for node in s_nodes]

        c_inds = []
        for s_node, s_path, s_loc, s_zin in zip(s_nodes, s_ptrs, s_locs, s_zins):
            z_diffs = []
            # check if there are compartment nodes before bifurcation nodes in up direction
            nn_inds = greens_tree.get_nearest_neighbour_loc_idxs(s_loc, c_locs)
            # print c_before_b
            c_ns = [c_nodes[ii] for ii in nn_inds]
            c_ps = [c_ptrs[ii] for ii in nn_inds]
            c_ls = [c_locs[ii] for ii in nn_inds]
            c_zs = [c_zins[ii] for ii in nn_inds]
            for c_node, c_path, c_loc, c_zin in zip(c_ns, c_ps, c_ls, c_zs):
                # find the common node as far from the root as possible
                s_p, c_p = s_path[::-1], c_path[::-1]
                kk = 0
                while kk < min(len(s_p), len(c_p)) and s_p[kk] == c_p[kk]:
                    p_node = s_p[kk]
                    kk += 1
                # distinguish cases for computing impedance different
                if p_node == s_node and p_node != c_node:
                    z_diffs.append(fz * np.abs(c_zin - s_zin))
                elif p_node == c_node and p_node != s_node:
                    z_diffs.append((1.0 - fz) * np.abs(s_zin - c_zin))
                elif p_node == c_node and p_node == s_node:
                    fz_ = fz if c_loc["x"] > s_loc["x"] else (1.0 - fz)
                    z_diffs.append(fz_ * np.abs(s_zin - c_zin))
                else:
                    b_loc = (p_node.index, 1.0)
                    b_z = greens_tree.calc_zf(b_loc, b_loc)[0]
                    z_diffs.append((1.0 - fz) * (c_zin - b_z) + fz * (s_zin - b_z))
            # compartment node with minimal impedance difference
            ind_aux = np.argmin(z_diffs)
            c_inds.append(nn_inds[ind_aux])

        return c_inds
