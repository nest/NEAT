# -*- coding: utf-8 -*-
#
# test_compartmentfitter.py
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
import os

import pytest
import pickle

from neat import (
    MorphTree,
    PhysTree,
    GreensTree,
    SOVTree,
    NeuronCompartmentTree,
    check_for_coreneuron,
)
from neat import CompartmentFitter, CachedGreensTree
import neat.modelreduction.compartmentfitter as compartmentfitter

import channelcollection_for_tests as channelcollection
import channel_installer

channel_installer.load_or_install_neuron_test_channels()


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_morphologies")
)
SKIP_ACTIVE_MECHS_WITH_CORENEURON = pytest.mark.skipif(
    check_for_coreneuron(),
    reason="Active mechanism tests are skipped when running with CoreNEURON",
)


class TestCompartmentFitter:
    def load_T_tree(self):
        """
        Load the T-tree model

          6--5--4--7--8
                |
                |
                1
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "Tsovtree.swc")
        self.tree = PhysTree(fname, types=[1, 3, 4])
        self.tree.set_physiology(0.8, 100.0 / 1e6)
        self.tree.fit_leak_current(-75.0, 10.0)
        self.tree.set_comp_tree()

    def load_ball_and_stick(self):
        """
        Load the ball and stick model

        1--4
        """
        self.tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball_and_stick.swc")
        )
        self.tree.set_physiology(0.8, 100.0 / 1e6)
        self.tree.set_leak_current(100.0, -75.0)
        self.tree.set_comp_tree()

    def load_ball(self):
        """
        Load point neuron model
        """
        self.tree = PhysTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc"))
        # capacitance and axial resistance
        self.tree.set_physiology(0.8, 100.0 / 1e6)
        # ion channels
        k_chan = channelcollection.Kv3_1()
        self.tree.add_channel_current(k_chan, 0.766 * 1e6, -85.0)
        na_chan = channelcollection.Na_Ta()
        self.tree.add_channel_current(na_chan, 1.71 * 1e6, 50.0)
        # fit leak current
        self.tree.fit_leak_current(-75.0, 10.0)
        # set equilibirum potententials
        self.tree.set_v_ep(-75.0)
        # set computational tree
        self.tree.set_comp_tree()

    def load_T_segment_tree(self, fit_e_l=True):
        """
        Load T tree model
        """
        self.tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "Ttree_segments.swc")
        )
        # capacitance and axial resistance
        self.tree.set_physiology(0.8, 100.0 / 1e6)
        # ion channels
        k_chan = channelcollection.Kv3_1()

        g_k = {1: 0.766 * 1e6}
        g_k.update(
            {
                n.index: 0.034 * 1e6 / self.tree.path_length((1, 0.5), (n.index, 0.5))
                for n in self.tree
                if n.index != 1
            }
        )

        self.tree.add_channel_current(k_chan, g_k, -85.0)
        na_chan = channelcollection.Na_Ta()
        self.tree.add_channel_current(
            na_chan, 1.71 * 1e6, 50.0, node_arg=[self.tree[1]]
        )
        # fit leak current
        if fit_e_l:
            self.tree.fit_leak_current(-75.0, 10.0)
        else:
            self.tree.set_leak_current(0.0001 * 1e6, lambda x: -60.0 - 0.05 * x)
        for node in self.tree:
            print(node.currents)
        # set equilibirum potententials
        self.tree.set_v_ep(-75.0)
        # set computational tree
        self.tree.set_comp_tree()

    def test_tree_structure(self):
        self.load_T_tree()
        cm = CompartmentFitter(self.tree, cache_path="neatcache/")
        # set of locations
        fit_locs1 = [(1, 0.5), (4, 0.5), (5, 0.5)]  # no bifurcations
        fit_locs2 = [
            (1, 0.5),
            (4, 0.5),
            (5, 0.5),
            (8, 0.5),
        ]  # w bifurcation, should be added
        fit_locs3 = [
            (1, 0.5),
            (4, 1.0),
            (5, 0.5),
            (8, 0.5),
        ]  # w bifurcation, already added

        # test fit_locs1, no bifurcation are added
        # input paradigm 1
        cm.set_ctree(fit_locs1, extend_w_bifurc=True)

    def _check_channels(self, tree, channel_names):
        assert isinstance(tree, CachedGreensTree)
        assert set(tree.channel_storage.keys()) == set(channel_names)
        for node in tree:
            assert set(node.currents.keys()) == set(channel_names + ["L"])

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_create_tree_gf(self):
        self.load_ball()
        cm = CompartmentFitter(self.tree, cache_path="neatcache/")

        # create tree with only 'L'
        tree_pas = cm.create_tree_gf()
        self._check_channels(tree_pas, [])
        # create tree with only 'Na_Ta'
        tree_na = cm.create_tree_gf(["Na_Ta"])
        self._check_channels(tree_na, ["Na_Ta"])
        # create tree with only 'Kv3_1'
        tree_k = cm.create_tree_gf(["Kv3_1"])
        self._check_channels(tree_k, ["Kv3_1"])
        # create tree with all channels
        tree_all = cm.create_tree_gf(["Na_Ta", "Kv3_1"])
        self._check_channels(tree_all, ["Na_Ta", "Kv3_1"])

    def reduce_explicit(self):
        self.load_ball()

        freqs = np.array([0.0])
        locs = [(1, 0.5)]
        e_eqs = [-75.0, -55.0, -35.0, -15.0]
        # create compartment tree
        ctree = self.tree.create_compartment_tree(locs)
        ctree.add_channel_current(channelcollection.Na_Ta(), 50.0)
        ctree.add_channel_current(channelcollection.Kv3_1(), -85.0)

        # create tree with only leak
        greens_tree_pas = GreensTree(self.tree)
        greens_tree_pas[1].currents = {"L": greens_tree_pas[1].currents["L"]}
        greens_tree_pas.set_comp_tree()
        greens_tree_pas.set_impedance(freqs)
        # compute the passive impedance matrix
        z_mat_pas = greens_tree_pas.calc_impedance_matrix(locs)[0]

        # create tree with only potassium
        greens_tree_k = GreensTree(self.tree)
        greens_tree_k[1].currents = {
            key: val for key, val in greens_tree_k[1].currents.items() if key != "Na_Ta"
        }
        # compute potassium impedance matrices
        z_mats_k = []
        for e_eq in e_eqs:
            greens_tree_k.set_v_ep(e_eq)
            greens_tree_k.set_comp_tree()
            greens_tree_k.set_impedance(freqs)
            z_mats_k.append(greens_tree_k.calc_impedance_matrix(locs))

        # create tree with only sodium
        greens_tree_na = GreensTree(self.tree)
        greens_tree_na[1].currents = {
            key: val
            for key, val in greens_tree_na[1].currents.items()
            if key != "Kv3_1"
        }
        # create state variable expansion points
        svs = []
        e_eqs_ = []
        na_chan = greens_tree_na.channel_storage["Na_Ta"]
        for e_eq1 in e_eqs:
            sv1 = na_chan.compute_varinf(e_eq1)
            for e_eq2 in e_eqs:
                e_eqs_.append(e_eq1)
                sv2 = na_chan.compute_varinf(e_eq2)
                svs.append({"m": sv1["m"], "h": sv2["h"]})

        # compute sodium impedance matrices
        z_mats_na = []
        for sv, eh in zip(svs, e_eqs_):
            greens_tree_na.set_v_ep(eh)
            greens_tree_na[1].set_expansion_point("Na_Ta", sv)
            greens_tree_na.set_comp_tree()
            greens_tree_na.set_impedance(freqs)
            z_mats_na.append(greens_tree_na.calc_impedance_matrix(locs))

        # passive fit
        ctree.compute_gmc(z_mat_pas)

        # potassium channel fit matrices
        fit_mats_k = []
        print(z_mats_k, e_eqs)
        for z_mat_k, e_eq in zip(z_mats_k, e_eqs):
            mf, vt = ctree.compute_g_single_channel(
                "Kv3_1",
                z_mat_k,
                e_eq,
                freqs,
                all_channel_names=["Kv3_1"],
                other_channel_names=["L"],
                action="return",
            )
            fit_mats_k.append([mf, vt])

        # sodium channel fit matrices
        fit_mats_na = []
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            mf, vt = ctree.compute_g_single_channel(
                "Na_Ta",
                z_mat_na,
                e_eq,
                freqs,
                sv=sv,
                all_channel_names=["Na_Ta"],
                other_channel_names=["L"],
                action="return",
            )
            fit_mats_na.append([mf, vt])

        return fit_mats_na, fit_mats_k

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_channel_fit_mats(self):
        self.load_ball()
        cm = CompartmentFitter(
            self.tree, cache_name="channelfitmats", cache_path="neatcache/"
        )
        cm.set_ctree([(1, 0.5)], fit_name="test fit mats")
        ctree_cm = cm.fitted_models["test fit mats"]["ctree"]
        # check if reversals are correct
        for key in set(ctree_cm[0].currents) - {"L"}:
            assert (
                np.abs(ctree_cm[0].currents[key][1] - self.tree[1].currents[key][1])
                < 1e-10
            )

        # fit the passive model
        cm.fit_passive("test fit mats", use_all_channels=False)

        fit_mats_cm_na = cm._eval_channel("test fit mats", "Na_Ta")
        fit_mats_cm_k = cm._eval_channel("test fit mats", "Kv3_1")
        fit_mats_control_na, fit_mats_control_k = self.reduce_explicit()

        # test whether potassium fit matrices agree
        for fm_cm, fm_control in zip(fit_mats_cm_k, fit_mats_control_k):
            assert np.allclose(fm_cm[1] / fm_control[1], fm_cm[0] / fm_control[0])
        # test whether sodium fit matrices agree
        for fm_cm, fm_control in zip(fit_mats_cm_na[4:], fit_mats_control_na):
            assert np.allclose(fm_cm[1] / fm_control[1], fm_cm[0] / fm_control[0])

    def _check_pas_cond_props(self, ctree1, ctree2):
        assert len(ctree1) == len(ctree2)
        for n1, n2 in zip(ctree1, ctree2):
            assert np.allclose(n1.currents["L"][0], n2.currents["L"][0])
            assert np.allclose(n1.g_c, n2.g_c)

    def _check_pas_ca_props(self, ctree1, ctree2):
        assert len(ctree1) == len(ctree2)
        for n1, n2 in zip(ctree1, ctree2):
            assert np.allclose(n1.ca, n2.ca)

    def _check_all_curr_props(self, ctree1, ctree2):
        assert len(ctree1) == len(ctree2)
        assert ctree1.channel_storage.keys() == ctree2.channel_storage.keys()
        for n1, n2 in zip(ctree1, ctree2):
            assert np.allclose(n1.g_c, n2.g_c)
            for key in n1.currents:
                assert np.allclose(n1.currents[key][0], n2.currents[key][0])
                assert np.allclose(n1.currents[key][1], n2.currents[key][1])

    def _check_phys_trees(self, tree1, tree2):
        assert len(tree1) == len(tree2)
        assert tree1.channel_storage.keys() == tree2.channel_storage.keys()
        for n1, n2 in zip(tree1, tree2):
            assert np.allclose(n1.r_a, n2.r_a)
            assert np.allclose(n1.c_m, n2.c_m)
            for key in n1.currents:
                assert np.allclose(n1.currents[key][0], n2.currents[key][0])
                assert np.allclose(n1.currents[key][1], n2.currents[key][1])

    def _check_e_leak(self, ctree, e_l):
        for n in ctree:
            assert np.allclose(n.currents["L"][1], e_l)

    def test_construction(self):
        swc_path = os.path.join(MORPHOLOGIES_PATH_PREFIX, "Tsovtree.swc")
        with pytest.warns(
            UserWarning,
            match="Initialization of a CompartmentFitter-instance as a tree",
        ):
            cfit = CompartmentFitter(save_cache=False)
        with pytest.warns(
            UserWarning,
            match="Initialization of a CompartmentFitter-instance as a tree",
        ):
            cfit = CompartmentFitter(swc_path, save_cache=False)
        with pytest.warns(
            UserWarning,
            match="Initialization of a CompartmentFitter-instance as a tree",
        ):
            cfit = CompartmentFitter(MorphTree(swc_path), save_cache=False)

        cfit = CompartmentFitter(
            PhysTree(swc_path), cache_name="test_name", save_cache=False
        )
        assert cfit.cache_name == "test_name"
        cfit_ = CompartmentFitter(cfit, cache_name="test_name_new", save_cache=False)
        assert cfit_.cache_name == "test_name_new"
        cfit__ = CompartmentFitter(cfit_, save_cache=False)
        assert cfit__.cache_name == "test_name_new"

        # TODO: write similar test for fit_cfg and concmech_cfg

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_passive_fit(self):
        self.load_T_tree()
        fit_locs = [(1, 0.5), (4, 1.0), (5, 0.5), (8, 0.5)]

        # fit a tree directly from CompartmentTree
        greens_tree = GreensTree(self.tree)
        greens_tree.set_comp_tree()
        freqs = np.array([0.0])
        greens_tree.set_impedance(freqs)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs)[0].real
        ctree = greens_tree.create_compartment_tree(fit_locs)
        ctree.compute_gmc(z_mat)
        sov_tree = SOVTree(self.tree)
        sov_tree.calc_sov_equations()
        alphas, phimat = sov_tree.get_important_modes(loc_arg=fit_locs)
        ctree.compute_c(-alphas[0:1].real * 1e3, phimat[0:1, :].real)

        # fit a tree with compartment fitter
        cm = CompartmentFitter(
            self.tree, cache_name="passivefit1", cache_path="neatcache/"
        )
        cm.set_ctree(fit_locs, fit_name="test passive fit")
        cm.fit_passive("test passive fit")
        cm.fit_capacitance("test passive fit")
        cm.fit_e_eq("test passive fit")
        ctree_ = cm.fitted_models["test passive fit"]["ctree"]

        # check whether both trees are the same
        self._check_pas_cond_props(ctree_, ctree)
        self._check_pas_ca_props(ctree_, ctree)
        self._check_e_leak(ctree_, -75.0)

        # test whether all channels are used correctly for passive fit
        self.load_ball()
        fit_locs = [(1, 0.5)]
        # fit ball model with only leak
        greens_tree = GreensTree(self.tree)
        greens_tree.channel_storage = {}
        for n in greens_tree:
            n.currents = {"L": n.currents["L"]}
        greens_tree.set_comp_tree()
        freqs = np.array([0.0])
        greens_tree.set_impedance(freqs)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs)[0].real
        ctree_leak = greens_tree.create_compartment_tree(fit_locs)
        ctree_leak.compute_gmc(z_mat)
        sov_tree = SOVTree(greens_tree)
        sov_tree.calc_sov_equations()
        alphas, phimat = sov_tree.get_important_modes(loc_arg=fit_locs)
        ctree_leak.compute_c(-alphas[0:1].real * 1e3, phimat[0:1, :].real)
        # make ball model with leak based on all channels
        tree = PhysTree(self.tree)
        tree.as_passive_membrane()
        tree.set_comp_tree()
        greens_tree = GreensTree(tree)
        greens_tree.set_comp_tree()
        freqs = np.array([0.0])
        greens_tree.set_impedance(freqs)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs)[0].real
        ctree_all = greens_tree.create_compartment_tree(fit_locs)
        ctree_all.compute_gmc(z_mat)
        sov_tree = SOVTree(tree)
        sov_tree.set_comp_tree()
        sov_tree.calc_sov_equations()
        alphas, phimat = sov_tree.get_important_modes(loc_arg=fit_locs)
        ctree_all.compute_c(-alphas[0:1].real * 1e3, phimat[0:1, :].real)

        # new compartment fitter
        cm = CompartmentFitter(
            self.tree, cache_name="passivefit2", cache_path="neatcache/"
        )
        cm.set_ctree(fit_locs, fit_name="test passive fit 2")
        # test fitting
        cm.fit_passive("test passive fit 2", use_all_channels=False)
        cm.fit_capacitance("test passive fit 2")
        cm.fit_e_eq("test passive fit 2")
        ctree_cm = cm.fitted_models["test passive fit 2"]["ctree"]

        self._check_pas_cond_props(ctree_leak, ctree_cm)
        self._check_pas_ca_props(ctree_leak, ctree_cm)
        with pytest.raises(AssertionError):
            self._check_e_leak(ctree_cm, self.tree[1].currents["L"][1])
        cm.fit_passive("test passive fit 2", use_all_channels=True)
        cm.fit_capacitance("test passive fit 2")
        cm.fit_e_eq("test passive fit 2")
        self._check_pas_cond_props(ctree_all, ctree_cm)
        self._check_pas_ca_props(ctree_all, ctree_cm)
        self._check_e_leak(ctree_cm, greens_tree[1].currents["L"][1])
        with pytest.raises(AssertionError):
            self._check_e_leak(ctree_cm, self.tree[1].currents["L"][1])
        with pytest.raises(AssertionError):
            self._check_pas_cond_props(ctree_leak, ctree_all)

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_recalc_impedance_matrix(self, g_inp=np.linspace(0.0, 0.01, 20)):
        self.load_ball()
        fit_locs = [(1, 0.5)]
        cm = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        cm.set_ctree(fit_locs, fit_name="impedance matrix test")

        # test only leak
        # compute impedances explicitly
        greens_tree = cm.create_tree_gf(
            channel_names=[],
        )
        greens_tree.set_v_ep(-75.0)
        greens_tree.set_impedances_in_tree(freqs=0.0, pprint=True)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs, explicit_method=False)
        z_test = z_mat[:, :, None] / (1.0 + z_mat[:, :, None] * g_inp[None, None, :])
        # compute impedances with compartmentfitter function
        z_calc = np.array(
            [
                cm.recalc_impedance_matrix(
                    "impedance matrix test", [g_i], channel_names=[]
                )
                for g_i in g_inp
            ]
        )
        z_calc = np.swapaxes(z_calc, 0, 2)
        assert np.allclose(z_calc, z_test)

        # test with z based on all channels (passive)
        # compute impedances explicitly
        greens_tree = cm.create_tree_gf(
            channel_names=list(cm.channel_storage.keys()),
        )
        greens_tree.set_v_ep(-75.0)
        greens_tree.set_impedances_in_tree(freqs=0.0, pprint=True)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs, explicit_method=False)
        z_test = z_mat[:, :, None] / (1.0 + z_mat[:, :, None] * g_inp[None, None, :])
        # compute impedances with compartmentfitter function
        z_calc = np.array(
            [
                cm.recalc_impedance_matrix(
                    "impedance matrix test",
                    [g_i],
                    channel_names=list(cm.channel_storage.keys()),
                )
                for g_i in g_inp
            ]
        )
        z_calc = np.swapaxes(z_calc, 0, 2)
        assert np.allclose(z_calc, z_test)

    def test_syn_rescale(self, g_inp=np.linspace(0.0, 0.01, 20)):
        e_rev, v_eq = 0.0, -75.0
        self.load_ball_and_stick()
        fit_locs = [(4, 0.7)]
        syn_locs = [(4, 1.0)]
        cm = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        cm.set_ctree(fit_locs)
        # compute impedance matrix
        greens_tree = cm.create_tree_gf(
            channel_names=[],
        )
        greens_tree.set_v_ep(-75.0)
        greens_tree.set_impedances_in_tree(freqs=0.0)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs + syn_locs)
        # analytical synapse scale factors
        beta_calc = 1.0 / (1.0 + (z_mat[1, 1] - z_mat[0, 0]) * g_inp)
        beta_full = (
            z_mat[0, 1]
            / z_mat[0, 0]
            * (e_rev - v_eq)
            / ((1.0 + (z_mat[1, 1] - z_mat[0, 0]) * g_inp) * (e_rev - v_eq))
        )
        # synapse scale factors from compartment fitter
        beta_cm = np.array(
            [
                cm.fit_syn_rescale(fit_locs, syn_locs, [0], [g_i], e_revs=[0.0])[0]
                for g_i in g_inp
            ]
        )
        assert np.allclose(beta_calc, beta_cm, atol=0.020)
        assert np.allclose(beta_full, beta_cm, atol=0.015)

    def fit_ball(self):
        self.load_ball()
        freqs = np.array([0.0])
        locs = [(1, 0.5)]
        e_eqs = [-75.0, -55.0, -35.0, -15.0]
        # create compartment tree
        ctree = self.tree.create_compartment_tree(locs)
        ctree.add_channel_current(channelcollection.Na_Ta(), 50.0)
        ctree.add_channel_current(channelcollection.Kv3_1(), -85.0)

        # create tree with only leak
        greens_tree_pas = GreensTree(self.tree)
        greens_tree_pas[1].currents = {"L": greens_tree_pas[1].currents["L"]}
        greens_tree_pas.set_comp_tree()
        greens_tree_pas.set_impedance(freqs)
        # compute the passive impedance matrix
        z_mat_pas = greens_tree_pas.calc_impedance_matrix(locs)[0]

        # create tree with only potassium
        greens_tree_k = GreensTree(self.tree)
        greens_tree_k[1].currents = {
            key: val for key, val in greens_tree_k[1].currents.items() if key != "Na_Ta"
        }
        # compute potassium impedance matrices
        z_mats_k = []
        for e_eq in e_eqs:
            greens_tree_k.set_v_ep(e_eq)
            greens_tree_k.set_comp_tree()
            greens_tree_k.set_impedance(freqs)
            z_mats_k.append(greens_tree_k.calc_impedance_matrix(locs))

        # create tree with only sodium
        greens_tree_na = GreensTree(self.tree)
        greens_tree_na[1].currents = {
            key: val
            for key, val in greens_tree_na[1].currents.items()
            if key != "Kv3_1"
        }
        # create state variable expansion points
        svs = []
        e_eqs_ = []
        na_chan = greens_tree_na.channel_storage["Na_Ta"]
        for e_eq1 in e_eqs:
            sv1 = na_chan.compute_varinf(e_eq1)
            for e_eq2 in e_eqs:
                e_eqs_.append(e_eq2)
                sv2 = na_chan.compute_varinf(e_eq2)
                svs.append({"m": sv2["m"], "h": sv1["h"]})

        # compute sodium impedance matrices
        z_mats_na = []
        for ii, sv in enumerate(svs):
            greens_tree_na.set_v_ep(e_eqs[ii % len(e_eqs)])
            greens_tree_na[1].set_expansion_point("Na_Ta", sv)
            greens_tree_na.set_comp_tree()
            greens_tree_na.set_impedance(freqs)
            z_mats_na.append(greens_tree_na.calc_impedance_matrix(locs))

        # passive fit
        ctree.compute_gmc(z_mat_pas)
        # get SOV constants for capacitance fit
        sov_tree = SOVTree(greens_tree_pas)
        sov_tree.set_comp_tree()
        sov_tree.calc_sov_equations()
        alphas, phimat, importance = sov_tree.get_important_modes(
            loc_arg=locs, sort_type="importance", eps=1e-12, return_importance=True
        )
        # fit the capacitances from SOV time-scales
        ctree.compute_c(
            -alphas[0:1].real * 1e3, phimat[0:1, :].real, weights=importance[0:1]
        )

        # potassium channel fit
        for z_mat_k, e_eq in zip(z_mats_k, e_eqs):
            ctree.compute_g_single_channel(
                "Kv3_1", z_mat_k, e_eq, freqs, other_channel_names=["L"]
            )
        ctree.run_fit()
        # sodium channel fit
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            ctree.compute_g_single_channel(
                "Na_Ta", z_mat_na, e_eq, freqs, sv=sv, other_channel_names=["L"]
            )
        ctree.run_fit()

        ctree.set_e_eq(-75.0)
        ctree.remove_expansion_points()
        ctree.fit_e_leak()

        self.ctree = ctree

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_fit_model(self):
        self.load_T_tree()
        fit_locs = [(1, 0.5), (4, 1.0), (5, 0.5), (8, 0.5)]

        # fit a tree directly from CompartmentTree
        greens_tree = GreensTree(self.tree)
        greens_tree.set_comp_tree()
        freqs = np.array([0.0])
        greens_tree.set_impedance(freqs)
        z_mat = greens_tree.calc_impedance_matrix(fit_locs)[0]
        ctree = greens_tree.create_compartment_tree(fit_locs)
        ctree.compute_gmc(z_mat)
        sov_tree = SOVTree(self.tree)
        sov_tree.calc_sov_equations()
        alphas, phimat = sov_tree.get_important_modes(loc_arg=fit_locs)
        ctree.compute_c(-alphas[0:1].real * 1e3, phimat[0:1, :].real)

        # fit a tree with compartmentfitter
        cm = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        ctree_cm, _ = cm.fit_model(fit_locs)

        # compare the two trees
        self._check_pas_cond_props(ctree_cm, ctree)
        self._check_pas_ca_props(ctree_cm, ctree)
        self._check_e_leak(ctree_cm, -75.0)

        # check active channel
        self.fit_ball()
        locs = [(1, 0.5)]
        cm = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        print("(i)")
        ctree_cm_1, _ = cm.fit_model(locs, use_all_channels_for_passive=False)
        print("(ii)")
        ctree_cm_2, _ = cm.fit_model(locs, use_all_channels_for_passive=True)
        print("(iii)")

        self._check_all_curr_props(self.ctree, ctree_cm_1)
        self._check_all_curr_props(self.ctree, ctree_cm_2)

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_pickling(self):
        self.load_ball()

        # of PhysTree
        ss = pickle.dumps(self.tree)
        pt_ = pickle.loads(ss)
        self._check_phys_trees(self.tree, pt_)

        # of GreensTree
        greens_tree = GreensTree(self.tree)
        greens_tree.set_comp_tree()
        freqs = np.array([0.0])
        greens_tree.set_impedance(freqs)

        ss = pickle.dumps(greens_tree)
        gt_ = pickle.loads(ss)
        self._check_phys_trees(greens_tree, gt_)

        # of SOVTree
        sov_tree = SOVTree(self.tree)
        sov_tree.calc_sov_equations()

        # works with pickle
        ss = pickle.dumps(sov_tree)
        st_ = pickle.loads(ss)
        self._check_phys_trees(sov_tree, st_)

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_cacheing(self):
        self.load_T_segment_tree()
        locs = [(1, 0.5), (4, 0.9)]

        cm1 = CompartmentFitter(
            self.tree,
            cache_name="cacheingtest",
            cache_path="neatcache/",
            recompute_cache=True,
        )
        ctree_cm_1a, _ = cm1.fit_model(locs, use_all_channels_for_passive=False)
        ctree_cm_1b, _ = cm1.fit_model(locs, use_all_channels_for_passive=True)

        sv_h = compartmentfitter.get_expansion_points(
            cm1.fit_cfg.e_hs, channelcollection.Na_Ta()
        )
        tree_na_1 = cm1.create_tree_gf(["Na_Ta"])
        tree_na_1.set_impedances_in_tree(freqs=cm1.fit_cfg.freqs, sv_h={"Na_Ta": sv_h})
        del cm1

        cm2 = CompartmentFitter(
            self.tree,
            cache_name="cacheingtest",
            cache_path="neatcache/",
            recompute_cache=False,
        )
        ctree_cm_2a, _ = cm2.fit_model(locs, use_all_channels_for_passive=False)
        ctree_cm_2b, _ = cm2.fit_model(locs, use_all_channels_for_passive=True)

        sv_h = compartmentfitter.get_expansion_points(
            cm2.fit_cfg.e_hs, channelcollection.Na_Ta()
        )
        tree_na_2 = cm2.create_tree_gf(["Na_Ta"])
        tree_na_2.set_impedances_in_tree(freqs=cm2.fit_cfg.freqs, sv_h={"Na_Ta": sv_h})
        del cm2

        self._check_all_curr_props(ctree_cm_1a, ctree_cm_2a)
        self._check_all_curr_props(ctree_cm_1b, ctree_cm_2b)

        with pytest.raises(AssertionError):
            self._check_all_curr_props(ctree_cm_1a, ctree_cm_1b)

        print(tree_na_1.unique_hash())

        assert (
            tree_na_1.unique_hash()
            == "41061ba7674c6a237bd7e15428904ba9524629cd5ea83a525d6fc7da305d3f34"
        )
        assert repr(tree_na_1) == repr(tree_na_2)

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_fit_storage(self):
        self.load_T_segment_tree()
        locs = [(1, 0.5), (4, 0.9)]

        cm = CompartmentFitter(self.tree, save_cache=False)

        # check that nothing is stored without an argument
        ctree0, locs0 = cm.fit_model(locs)
        assert len(cm.fitted_models) == 0

        # check that the trees are stored correctly with an argument
        ctree1, locs1 = cm.fit_model(locs, "test fit")
        assert cm.fitted_models["test fit"]["complete"]
        assert cm.fitted_models["test fit"]["ctree"] == ctree1
        for loc_a, loc_b in zip(locs1, cm.fitted_models["test fit"]["locs"]):
            assert loc_a == loc_b

        # check convert fit arg functionality
        ctree2, locs2 = cm.convert_fit_arg("test fit")
        assert ctree2 == ctree1
        for loc_a, loc_b in zip(locs1, locs2):
            assert loc_a == loc_b
        ctree3, locs3 = cm.convert_fit_arg((ctree2, locs2))
        assert ctree3 == ctree1
        for loc_a, loc_b in zip(locs1, locs3):
            assert loc_a == loc_b
            assert loc_a == loc_b
        ctree4, locs4 = cm.convert_fit_arg(cm.fitted_models["test fit"])
        assert ctree4 == ctree1
        for loc_a, loc_b in zip(locs1, locs4):
            assert loc_a == loc_b

        # test fit deletion
        cm.remove_fit("test fit")
        assert len(cm.fitted_models) == 0

        # test errors
        with pytest.raises(TypeError):
            cm.convert_fit_arg(13)
        with pytest.warns(
            UserWarning, match=f"Fit with name 'not defined' not in stored fits."
        ):
            cm.remove_fit("not defined")

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_e_eq_fit(self):
        self.load_ball()
        cm = CompartmentFitter(self.tree, save_cache=False)
        v_eq_target = cm[1].v_ep
        # create simplified model
        ctree, _ = cm.fit_model([(1, 0.5)])
        # simulate for effective equilibrium potential
        nct = NeuronCompartmentTree(ctree)
        nct.init_model(t_calibrate=1000.0)
        nct.store_locs([(0, 0.5)], name="rec locs")
        res = nct.run(100.0)
        v_eq_sim = res["v_m"][0, -1]

        assert ctree[0].currents["L"][1] == pytest.approx(self.tree[1].currents["L"][1])
        assert v_eq_sim == pytest.approx(v_eq_target)

        self.load_T_segment_tree(fit_e_l=False)
        cm = CompartmentFitter(self.tree, save_cache=False)
        v_eq_target = cm[1].v_ep
        ctree, _ = cm.fit_model([(1, 0.5), (10, 0.5)])

        nct = NeuronCompartmentTree(ctree)
        nct.init_model(t_calibrate=1000.0)
        nct.store_locs([(0, 0.5)], name="rec locs")
        res = nct.run(100.0)
        v_eq_sim = res["v_m"][0, -1]

        assert v_eq_sim == pytest.approx(v_eq_target)


@SKIP_ACTIVE_MECHS_WITH_CORENEURON
def test_expansion_points():
    kv3_1 = channelcollection.Kv3_1()
    na_ta = channelcollection.Na_Ta()

    e_hs = np.array([-75.0, -15.0])

    # test expansion point for channel with 1 state variable
    sv_hs = compartmentfitter.get_expansion_points(e_hs, kv3_1)
    for svar, f_inf in kv3_1.f_varinf.items():
        assert np.allclose(f_inf(e_hs), sv_hs[svar])
    assert np.allclose(sv_hs["v"], e_hs)

    # test expansion point for channel with 2 state variables
    sv_hs = compartmentfitter.get_expansion_points(e_hs, na_ta)
    v_act = np.array([-75.0, -15.0, -75.0, -75.0, -15.0, -15.0])
    v_inact = np.array([-75.0, -15.0, -75.0, -15.0, -75.0, -15.0])

    assert np.allclose(na_ta.f_varinf["m"](v_act), sv_hs["m"])
    assert np.allclose(na_ta.f_varinf["h"](v_inact), sv_hs["h"])
    assert not np.allclose(na_ta.f_varinf["h"](v_act), sv_hs["h"])
    assert np.allclose(v_act, sv_hs["v"])


if __name__ == "__main__":
    tcf = TestCompartmentFitter()
    tcf.test_construction()
    tcf.test_tree_structure()
    tcf.test_create_tree_gf()
    tcf.test_channel_fit_mats()
    tcf.test_passive_fit()
    tcf.test_recalc_impedance_matrix()
    tcf.test_syn_rescale()
    tcf.test_fit_model()
    tcf.test_pickling()
    tcf.test_parallel(w_benchmark=True)
    tcf.test_cacheing()
    tcf.test_fit_storage()
    tcf.test_e_eq_fit()

    test_expansion_points()
