# -*- coding: utf-8 -*-
#
# test_compartmenttree.py
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
import os

import pytest
import random
import copy

from neat import (
    SOVTree,
    SOVNode,
    Kernel,
    GreensTree,
    CompartmentTree,
    CompartmentNode,
    check_for_coreneuron,
)
import neat.tools.kernelextraction as ke

# from neat.channels.channelcollection import channelcollection

import channelcollection_for_tests as channelcollection

MORPHOLOGIES_PATH_PREFIX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_morphologies")
)
SKIP_ACTIVE_MECHS_WITH_CORENEURON = pytest.mark.skipif(
    check_for_coreneuron(),
    reason="Active mechanism tests are skipped when running with CoreNEURON",
)


class TestCompartmentTree:
    def load_T_tree(self):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "Tsovtree.swc")
        self.tree = SOVTree(fname, types=[1, 3, 4])
        self.tree.fit_leak_current(-75.0, 10.0)
        self.tree.set_comp_tree()
        # do SOV calculation
        self.tree.calc_sov_equations()

    def test_string_representation(self):
        # create simple compartment tree
        self.load_T_tree()
        locs = [(1, 0.5), (4, 0.5)]
        ctree = self.tree.create_compartment_tree(locs)

        assert (
            str(ctree) == ">>> CompartmentTree\n"
            "    CompartmentNode 0, Parent: None --- loc_idx = 0, g_c = 0.0 uS, ca = 1.0 uF, e_eq = -75.0 mV, (g_L = 0.01 uS, e_L = -75.0 mV)\n"
            "    CompartmentNode 1, Parent: 0 --- loc_idx = 1, g_c = 0.0 uS, ca = 1.0 uF, e_eq = -75.0 mV, (g_L = 0.01 uS, e_L = -75.0 mV)"
        )

        assert (
            repr(ctree) == "['CompartmentTree', "
            "\"{'node index': 0, 'parent index': -1, 'content': '{}', 'loc_idx': 0, 'ca': '1', 'g_c': '0', 'e_eq': '-75', 'conc_eqs': {}, 'currents': {'L': '0.01, -75'}, 'concmechs': {}, 'expansion_points': {}}\", "
            "\"{'node index': 1, 'parent index': 0, 'content': '{}', 'loc_idx': 1, 'ca': '1', 'g_c': '0', 'e_eq': '-75', 'conc_eqs': {}, 'currents': {'L': '0.01, -75'}, 'concmechs': {}, 'expansion_points': {}}\""
            "]{'channel_storage': []}"
        )

    def test_tree_derivation(self):
        self.load_T_tree()
        # locations
        locs_soma = [(1, 0.5)]
        locs_prox = [(4, 0.2)]
        locs_bifur = [(4, 1.0)]
        locs_dist_nobifur = [(6.0, 0.5), (8.0, 0.5)]
        locs_dist_bifur = [(4, 1.0), (6.0, 0.5), (8.0, 0.5)]
        locs_dist_nroot = [(4, 1.0), (4, 0.5), (6.0, 0.5), (8.0, 0.5)]
        # test structures
        with pytest.raises(KeyError):
            self.tree.create_compartment_tree("set0")
        # test root (is soma) in set
        self.tree.store_locs(locs_dist_bifur + locs_soma, "set0")
        ctree = self.tree.create_compartment_tree("set0")
        assert ctree[0].loc_idx == 3
        assert ctree[1].loc_idx == 0
        cloc_idxs = [cn.loc_idx for cn in ctree[1].child_nodes]
        assert 1 in cloc_idxs and 2 in cloc_idxs
        # test soma not in set (but common root)
        self.tree.store_locs(locs_dist_bifur, "set1")
        ctree = self.tree.create_compartment_tree("set1")
        assert ctree[0].loc_idx == 0
        cloc_idxs = [cn.loc_idx for cn in ctree[0].child_nodes]
        assert 1 in cloc_idxs and 2 in cloc_idxs
        # test soma not in set and no common root
        self.tree.store_locs(locs_dist_nobifur, "set2")
        with pytest.warns(UserWarning):
            ctree = self.tree.create_compartment_tree("set2")
        assert self.tree.get_locs("set2")[0] == (4, 1.0)
        cloc_idxs = [cn.loc_idx for cn in ctree[0].child_nodes]
        assert 1 in cloc_idxs and 2 in cloc_idxs
        # test 2 locs on common root
        self.tree.store_locs(locs_dist_nroot, "set3")
        ctree = self.tree.create_compartment_tree("set3")
        assert ctree[0].loc_idx == 1
        assert ctree[1].loc_idx == 0

    def testFitting(self):
        self.load_T_tree()
        # locations
        locs_soma = [(1, 0.5)]
        locs_prox = [(4, 0.2)]
        locs_bifur = [(4, 1.0)]
        locs_dist_nobifur = [(6.0, 0.5), (8.0, 0.5)]
        locs_dist_bifur = [(4, 1.0), (6.0, 0.5), (8.0, 0.5)]
        # store the locations
        self.tree.store_locs(locs_soma + locs_prox, "prox")
        self.tree.store_locs(locs_soma + locs_bifur, "bifur")
        self.tree.store_locs(locs_soma + locs_dist_nobifur, "dist_nobifur")
        self.tree.store_locs(locs_soma + locs_dist_bifur, "dist_bifur")
        # derive steady state impedance matrices
        z_mat_prox = self.tree.calc_impedance_matrix(loc_arg="prox")
        z_mat_bifur = self.tree.calc_impedance_matrix(loc_arg="bifur")
        z_mat_dist_nobifur = self.tree.calc_impedance_matrix(loc_arg="dist_nobifur")
        z_mat_dist_bifur = self.tree.calc_impedance_matrix(loc_arg="dist_bifur")
        # create the tree structures
        ctree_prox = self.tree.create_compartment_tree("prox")
        ctree_bifur = self.tree.create_compartment_tree("bifur")
        ctree_dist_nobifur = self.tree.create_compartment_tree("dist_nobifur")
        ctree_dist_bifur = self.tree.create_compartment_tree("dist_bifur")
        # test the tree structures
        assert len(ctree_prox) == len(locs_prox) + 1
        assert len(ctree_bifur) == len(locs_bifur) + 1
        assert len(ctree_dist_nobifur) == len(locs_dist_nobifur) + 1
        assert len(ctree_dist_bifur) == len(locs_dist_bifur) + 1
        # fit the steady state models
        ctree_prox.compute_gmc(z_mat_prox)
        ctree_bifur.compute_gmc(z_mat_bifur)
        ctree_dist_nobifur.compute_gmc(z_mat_dist_nobifur)
        ctree_dist_bifur.compute_gmc(z_mat_dist_bifur)
        # compute the fitted impedance matrices
        z_fit_prox = ctree_prox.calc_impedance_matrix()
        z_fit_bifur = ctree_bifur.calc_impedance_matrix()
        z_fit_dist_nobifur = ctree_dist_nobifur.calc_impedance_matrix()
        z_fit_dist_bifur = ctree_dist_bifur.calc_impedance_matrix()
        # test correctness
        assert np.allclose(z_fit_prox, z_mat_prox, atol=0.5)
        assert np.allclose(z_fit_bifur, z_mat_bifur, atol=0.5)
        assert not np.allclose(z_fit_dist_nobifur, z_mat_dist_nobifur, atol=0.5)
        assert np.allclose(z_fit_dist_bifur, z_mat_dist_bifur, atol=0.5)

    def test_reordering(self):
        self.load_T_tree()
        # test reordering
        locs_dist_badorder = [(1.0, 0.5), (8.0, 0.5), (4, 1.0)]
        self.tree.store_locs(locs_dist_badorder, "badorder")
        z_mat_badorder = self.tree.calc_impedance_matrix(loc_arg="badorder")
        ctree_badorder = self.tree.create_compartment_tree("badorder")
        # check if location indices are assigned correctly
        assert [node.loc_idx for node in ctree_badorder] == [0, 2, 1]
        # check if reordering works
        z_mat_reordered = ctree_badorder._preprocess_z_mat_arg(z_mat_badorder)
        assert np.allclose(z_mat_reordered, z_mat_badorder[:, [0, 2, 1]][[0, 2, 1], :])
        # check if fitting is correct
        ctree_badorder.compute_gmc(z_mat_badorder)
        z_fit_badorder = ctree_badorder.calc_impedance_matrix()
        assert np.allclose(z_mat_badorder, z_fit_badorder, atol=0.5)
        assert not np.allclose(z_mat_reordered, z_fit_badorder)
        # test if equivalent locs are returned correctly
        locs_equiv = ctree_badorder.get_equivalent_locs()
        assert all(
            [
                loc == loc_
                for loc, loc_ in zip(locs_equiv, [(0, 0.5), (2, 0.5), (1, 0.5)])
            ]
        )

    def load_ball_and_stick(self):
        self.greens_tree = GreensTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball_and_stick.swc")
        )
        self.greens_tree.set_physiology(0.8, 100.0 / 1e6)
        self.greens_tree.set_leak_current(100.0, -75.0)
        self.greens_tree.set_comp_tree()
        # set the impedances
        self.freqs = np.array([0.0]) * 1j
        self.greens_tree.set_impedance(self.freqs)
        # create sov tree
        self.sov_tree = SOVTree(self.greens_tree)
        self.sov_tree.calc_sov_equations(maxspace_freq=50.0)

    def test_location_mapping(self, n_loc=20):
        self.load_ball_and_stick()
        # define locations
        xvals = np.linspace(0.0, 1.0, n_loc + 1)[1:]
        locs_1 = [(1, 0.5)] + [(4, x) for x in xvals]
        locs_2 = [(1, 0.5)] + [(4, x) for x in xvals][::-1]
        locs_3 = [(4, x) for x in xvals] + [(1, 0.5)]
        # create compartment trees
        ctree_1 = self.greens_tree.create_compartment_tree(locs_1)
        ctree_2 = self.greens_tree.create_compartment_tree(locs_2)
        ctree_3 = self.greens_tree.create_compartment_tree(locs_3)
        # test location indices
        locinds_1 = np.array([node.loc_idx for node in ctree_1])
        locinds_2 = np.array([node.loc_idx for node in ctree_2])
        locinds_3 = np.array([node.loc_idx for node in ctree_3])
        # check consecutive
        assert np.allclose(locinds_1[:-1], locinds_1[1:] - 1)
        # check permutation
        assert np.allclose(locinds_1[1:], locinds_2[1:][::-1])
        assert np.allclose(locinds_1[:-1], locinds_3[1:])

    def test_gss_fit(self, n_loc=20):
        self.load_ball_and_stick()
        # define locations
        xvals = np.linspace(0.0, 1.0, n_loc + 1)[1:]
        locs_1 = [(1, 0.5)] + [(4, x) for x in xvals]
        locs_2 = [(1, 0.5)] + [(4, x) for x in xvals][::-1]
        locs_3 = [(4, x) for x in xvals] + [(1, 0.5)]
        locs_4 = random.sample(locs_1, k=len(locs_1))
        # calculate impedance matrices
        z_mat_1 = self.greens_tree.calc_impedance_matrix(locs_1)[0].real
        z_mat_2 = self.greens_tree.calc_impedance_matrix(locs_2)[0].real
        z_mat_3 = self.greens_tree.calc_impedance_matrix(locs_3)[0].real
        z_mat_4 = self.greens_tree.calc_impedance_matrix(locs_4)[0].real
        # create compartment trees
        ctree_1 = self.greens_tree.create_compartment_tree(locs_1)
        ctree_2 = self.greens_tree.create_compartment_tree(locs_2)
        ctree_3 = self.greens_tree.create_compartment_tree(locs_3)
        ctree_4 = self.greens_tree.create_compartment_tree(locs_4)
        # fit g_m and g_c
        ctree_1.compute_gmc(z_mat_1, channel_names=["L"])
        ctree_2.compute_gmc(z_mat_2, channel_names=["L"])
        ctree_3.compute_gmc(z_mat_3, channel_names=["L"])
        ctree_4.compute_gmc(z_mat_4, channel_names=["L"])
        # compare impedance matrices
        z_fit_1 = ctree_1.calc_impedance_matrix(self.freqs)
        z_fit_2 = ctree_2.calc_impedance_matrix(self.freqs)
        z_fit_3 = ctree_3.calc_impedance_matrix(self.freqs)
        z_fit_4 = ctree_4.calc_impedance_matrix(self.freqs)
        assert np.allclose(z_fit_1, z_mat_1, atol=1e-8)
        assert np.allclose(z_fit_2, z_mat_2, atol=1e-8)
        assert np.allclose(z_fit_3, z_mat_3, atol=1e-8)
        assert np.allclose(z_fit_4, z_mat_4, atol=1e-8)
        assert np.allclose(z_fit_1, ctree_2.calc_impedance_matrix(indexing="tree"))
        assert np.allclose(z_fit_1, ctree_3.calc_impedance_matrix(indexing="tree"))
        assert np.allclose(z_fit_1, ctree_4.calc_impedance_matrix(indexing="tree"))

    def test_c_fit(self, n_loc=20):
        self.load_ball_and_stick()
        # define locations
        xvals = np.linspace(0.0, 1.0, n_loc + 1)[1:]
        locs = [(1, 0.5)] + [(4, x) for x in xvals]
        # create compartment tree
        ctree = self.greens_tree.create_compartment_tree(locs)
        # steady state fit
        z_mat = self.greens_tree.calc_impedance_matrix(locs)[0].real
        ctree.compute_gmc(z_mat)
        # get SOV constants for capacitance fit
        alphas, phimat, importance = self.sov_tree.get_important_modes(
            loc_arg=locs, sort_type="importance", eps=1e-12, return_importance=True
        )
        # fit the capacitances from SOV time-scales
        ctree.compute_c(
            -alphas[0:1].real * 1e3, phimat[0:1, :].real, weights=importance[0:1]
        )
        # check if equal to membrane time scale
        nds = [self.greens_tree[loc[0]] for loc in locs]
        taus_orig = np.array([n.c_m / n.currents["L"][0] for n in nds])
        taus_fit = np.array([n.ca / n.currents["L"][0] for n in ctree])
        assert np.allclose(taus_orig, taus_fit)

    def fit_ball_and_stick(self, n_loc=20):
        self.load_ball_and_stick()
        # define locations
        xvals = np.linspace(0.0, 1.0, n_loc + 1)[1:]
        np.random.shuffle(xvals)
        locs = [(1, 0.5)] + [(4, x) for x in xvals]
        # create compartment tree
        ctree = self.greens_tree.create_compartment_tree(locs)
        # steady state fit
        z_mat = self.greens_tree.calc_impedance_matrix(locs)[0].real
        ctree.compute_gmc(z_mat)
        # get SOV constants for capacitance fit
        alphas, phimat, importance = self.sov_tree.get_important_modes(
            loc_arg=locs, sort_type="importance", eps=1e-12, return_importance=True
        )
        # fit the capacitances from SOV time-scales
        ctree.compute_c(
            -alphas[0:1].real * 1e3, phimat[0:1, :].real, weights=importance[0:1]
        )
        self.ctree = ctree

    def test_pas_functionality(self, n_loc=10):
        self.fit_ball_and_stick(n_loc=n_loc)

        # test equilibrium potential setting
        e_eq = -75.0 + np.random.randint(10, size=n_loc + 1)
        # with tree indexing
        self.ctree.set_e_eq(e_eq, indexing="tree")
        assert np.allclose(e_eq, np.array([n.e_eq for n in self.ctree]))
        assert np.allclose(e_eq, self.ctree.get_e_eq(indexing="tree"))
        assert not np.allclose(e_eq, self.ctree.get_e_eq(indexing="locs"))
        # with loc indexing
        self.ctree.set_e_eq(e_eq, indexing="locs")
        assert not np.allclose(e_eq, np.array([n.e_eq for n in self.ctree]))
        assert not np.allclose(e_eq, self.ctree.get_e_eq(indexing="tree"))
        assert np.allclose(e_eq, self.ctree.get_e_eq(indexing="locs"))

        # conductance matrices
        gm1 = self.ctree.calc_conductance_matrix(indexing="locs")
        gm2 = self.ctree.calc_system_matrix(
            indexing="locs", channel_names=["L"], with_ca=True, use_conc=False
        )
        gm3 = self.ctree.calc_system_matrix(
            indexing="locs", channel_names=["L"], with_ca=False, use_conc=False
        )
        gm4 = self.ctree.calc_system_matrix(
            indexing="locs", channel_names=["L"], with_ca=False, use_conc=True
        )
        gm5 = self.ctree.calc_system_matrix(
            indexing="locs", with_ca=False, use_conc=True
        )
        gm6 = self.ctree.calc_system_matrix(
            indexing="tree", with_ca=False, use_conc=True
        )
        assert np.allclose(gm1, gm2)
        assert np.allclose(gm1, gm3)
        assert np.allclose(gm1, gm4)
        assert np.allclose(gm1, gm5)
        assert not np.allclose(gm1, gm6)

        # eigenvalues
        alphas, phimat, phimat_inv = self.ctree.calc_eigenvalues()
        ca_vec = np.array([1.0 / node.ca for node in self.ctree]) * 1e-3
        assert np.allclose(np.dot(phimat, phimat_inv), np.diag(ca_vec))
        assert np.allclose(
            np.array([n.ca / n.currents["L"][0] for n in self.ctree]),
            np.ones(len(self.ctree)) * np.max(1e-3 / np.abs(alphas)),
        )

    def load_ball(self):
        self.greens_tree = GreensTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc")
        )
        # capacitance and axial resistance
        self.greens_tree.set_physiology(0.8, 100.0 / 1e6)
        # ion channels
        k_chan = channelcollection.Kv3_1()
        self.greens_tree.add_channel_current(k_chan, 0.766 * 1e6, -85.0)
        na_chan = channelcollection.Na_Ta()
        self.greens_tree.add_channel_current(na_chan, 1.71 * 1e6, 50.0)
        # fit leak current
        self.greens_tree.fit_leak_current(-75.0, 10.0)
        # set computational tree
        self.greens_tree.set_comp_tree()
        # set the impedances
        self.freqs = np.array([0.0])
        self.greens_tree.set_impedance(self.freqs)
        # create sov tree
        self.sov_tree = SOVTree(self.greens_tree)
        self.sov_tree.calc_sov_equations(maxspace_freq=100.0)

    @SKIP_ACTIVE_MECHS_WITH_CORENEURON
    def test_channel_fit(self):
        self.load_ball()
        locs = [(1, 0.5)]
        e_eqs = [-75.0, -55.0, -35.0, -15.0]
        # create compartment tree
        ctree = self.greens_tree.create_compartment_tree(locs)
        ctree.add_channel_current(channelcollection.Na_Ta(), 50.0)
        ctree.add_channel_current(channelcollection.Kv3_1(), -85.0)

        # create tree with only leak
        greens_tree_pas = GreensTree(self.greens_tree)
        greens_tree_pas[1].currents = {"L": greens_tree_pas[1].currents["L"]}
        greens_tree_pas.set_comp_tree()
        greens_tree_pas.set_impedance(self.freqs)
        # compute the passive impedance matrix
        z_mat_pas = greens_tree_pas.calc_impedance_matrix(locs)[0]

        # create tree with only potassium
        greens_tree_k = GreensTree(self.greens_tree)
        greens_tree_k[1].currents = {
            key: val for key, val in greens_tree_k[1].currents.items() if key != "Na_Ta"
        }
        # compute potassium impedance matrices
        z_mats_k = []
        for e_eq in e_eqs:
            greens_tree_k.set_v_ep(e_eq)
            greens_tree_k.set_comp_tree()
            greens_tree_k.set_impedance(self.freqs)
            z_mats_k.append(greens_tree_k.calc_impedance_matrix(locs))

        # create tree with only sodium
        greens_tree_na = GreensTree(self.greens_tree)
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
            greens_tree_na.set_impedance(self.freqs)
            z_mats_na.append(greens_tree_na.calc_impedance_matrix(locs))

        # compute combined impedance matrices
        z_mats_comb = []
        for e_eq in e_eqs:
            self.greens_tree.set_v_ep(e_eq)
            self.greens_tree.set_comp_tree()
            self.greens_tree.set_impedance(self.freqs)
            z_mats_comb.append(self.greens_tree.calc_impedance_matrix(locs))

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

        ctree1 = copy.deepcopy(ctree)
        ctree2 = copy.deepcopy(ctree)
        ctree3 = copy.deepcopy(ctree)
        ctree4 = copy.deepcopy(ctree)

        # fit paradigm 1 --> separate impedance matrices and separate fits
        # potassium channel fit
        for z_mat_k, e_eq in zip(z_mats_k, e_eqs):
            ctree1.compute_g_single_channel(
                "Kv3_1", z_mat_k, e_eq, self.freqs, other_channel_names=["L"]
            )
        ctree1.run_fit()
        # sodium channel fit
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            ctree1.compute_g_single_channel(
                "Na_Ta", z_mat_na, e_eq, self.freqs, sv=sv, other_channel_names=["L"]
            )
        ctree1.run_fit()

        # fit paradigm 2 --> separate impedance matrices, same fit
        for z_mat_k, e_eq in zip(z_mats_k, e_eqs):
            ctree2.compute_g_single_channel(
                "Kv3_1", z_mat_k, e_eq, self.freqs, all_channel_names=["Kv3_1", "Na_Ta"]
            )
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            ctree2.compute_g_single_channel(
                "Na_Ta",
                z_mat_na,
                e_eq,
                self.freqs,
                sv=sv,
                all_channel_names=["Kv3_1", "Na_Ta"],
            )
        ctree2.run_fit()

        # fit paradigm 3 --> same impedance matrices
        for z_mat_comb, e_eq in zip(z_mats_comb, e_eqs):
            ctree3.compute_g_channels(["Kv3_1", "Na_Ta"], z_mat_comb, e_eq, self.freqs)
        ctree3.run_fit()

        # fit paradigm 4 --> fit incrementally
        for z_mat_na, e_eq, sv in zip(z_mats_na, e_eqs_, svs):
            ctree4.compute_g_single_channel("Na_Ta", z_mat_na, e_eq, self.freqs, sv=sv)
        ctree4.run_fit()
        for z_mat_comb, e_eq in zip(z_mats_comb, e_eqs):
            ctree4.compute_g_single_channel(
                "Kv3_1",
                z_mat_comb,
                e_eq,
                self.freqs,
                other_channel_names=["Na_Ta", "L"],
            )
        ctree4.run_fit()

        # test if correct
        keys = ["L", "Na_Ta", "Kv3_1"]
        # soma surface (cm) for total conductance calculation
        a_soma = 4.0 * np.pi * (self.greens_tree[1].R * 1e-4) ** 2
        conds = np.array(
            [self.greens_tree[1].currents[key][0] * a_soma for key in keys]
        )
        # compartment models conductances
        cconds1 = np.array([ctree1[0].currents[key][0] for key in keys])
        cconds2 = np.array([ctree2[0].currents[key][0] for key in keys])
        cconds3 = np.array([ctree3[0].currents[key][0] for key in keys])
        cconds4 = np.array([ctree4[0].currents[key][0] for key in keys])
        assert np.allclose(conds, cconds1)
        assert np.allclose(conds, cconds2)
        assert np.allclose(conds, cconds3)
        assert np.allclose(conds, cconds4)

        # rename for further testing
        ctree = ctree1
        # frequency array
        ft = ke.FourierQuadrature(np.linspace(0.0, 50.0, 100))
        freqs = ft.s
        # compute impedance matrix
        v_h = -42.0
        # original
        self.greens_tree.set_v_ep(v_h)
        self.greens_tree.set_comp_tree()
        self.greens_tree.set_impedance(freqs)
        z_mat_orig = self.greens_tree.calc_impedance_matrix([(1.0, 0.5)])
        # potassium
        greens_tree_k.set_v_ep(v_h)
        greens_tree_k.set_comp_tree()
        greens_tree_k.set_impedance(freqs)
        z_mat_k = greens_tree_k.calc_impedance_matrix([(1, 0.5)])
        # sodium
        greens_tree_na.remove_expansion_points()
        greens_tree_na.set_v_ep(v_h)
        greens_tree_na.set_comp_tree()
        greens_tree_na.set_impedance(freqs)
        z_mat_na = greens_tree_na.calc_impedance_matrix([(1, 0.5)])
        # passive
        greens_tree_pas.set_comp_tree()
        greens_tree_pas.set_impedance(freqs)
        z_mat_pas = greens_tree_pas.calc_impedance_matrix([(1, 0.5)])

        # reduced impedance matrices
        ctree.remove_expansion_points()
        ctree.set_e_eq(v_h)
        z_mat_fit = ctree.calc_impedance_matrix(freqs=freqs)
        z_mat_fit_k = ctree.calc_impedance_matrix(
            channel_names=["L", "Kv3_1"], freqs=freqs
        )
        z_mat_fit_na = ctree.calc_impedance_matrix(
            channel_names=["L", "Na_Ta"], freqs=freqs
        )
        z_mat_fit_pas = ctree.calc_impedance_matrix(channel_names=["L"], freqs=freqs)

        assert np.allclose(z_mat_orig, z_mat_fit)
        assert np.allclose(z_mat_k, z_mat_fit_k)
        assert np.allclose(z_mat_na, z_mat_fit_na)
        assert np.allclose(z_mat_pas, z_mat_fit_pas)

        # test total current, conductance
        sv = svs[-1]
        p_open = sv["m"] ** 3 * sv["h"]
        # with p_open given
        g1 = ctree[0].calc_g_tot(
            ctree.channel_storage,
            channel_names=["L", "Na_Ta"],
            p_open_channels={"Na_Ta": p_open},
        )
        i1 = ctree[0].calc_g_tot(
            ctree.channel_storage,
            channel_names=["L", "Na_Ta"],
            p_open_channels={"Na_Ta": p_open},
        )
        # with expansion point given
        ctree.set_expansion_points({"Na_Ta": sv})
        g2 = ctree[0].calc_g_tot(ctree.channel_storage, channel_names=["L", "Na_Ta"])
        i2 = ctree[0].calc_g_tot(ctree.channel_storage, channel_names=["L", "Na_Ta"])
        # with e_eq given
        g3 = ctree[0].calc_g_tot(
            ctree.channel_storage, v=e_eqs[-1], channel_names=["L", "Na_Ta"]
        )
        i3 = ctree[0].calc_g_tot(
            ctree.channel_storage, v=e_eqs[-1], channel_names=["L", "Na_Ta"]
        )
        # with e_eq stored
        ctree.set_e_eq(e_eqs[-1])
        g4 = ctree[0].calc_g_tot(ctree.channel_storage, channel_names=["L", "Na_Ta"])
        i4 = ctree[0].calc_g_tot(ctree.channel_storage, channel_names=["L", "Na_Ta"])
        # check if correct
        assert np.abs(g1 - g2) < 1e-10
        assert np.abs(g1 - g3) < 1e-10
        assert np.abs(g1 - g4) < 1e-10
        assert np.abs(i1 - i2) < 1e-10
        assert np.abs(i1 - i3) < 1e-10
        assert np.abs(i1 - i4) < 1e-10
        # compare current, conductance
        g_ = ctree[0].calc_g_tot(ctree.channel_storage, channel_names=["Na_Ta"])
        i_ = ctree[0].calc_i_tot(ctree.channel_storage, channel_names=["Na_Ta"])
        assert np.abs(g_ * (e_eqs[-1] - ctree[0].currents["Na_Ta"][1]) - i_) < 1e-10

        # test leak fitting
        self.greens_tree.set_v_ep(-75.0)
        self.greens_tree.set_comp_tree()
        ctree.set_e_eq(-75.0)
        ctree.remove_expansion_points()
        ctree.fit_e_leak()
        assert (
            np.abs(ctree[0].currents["L"][1] - self.greens_tree[1].currents["L"][1])
            < 1e-10
        )


class TestCompartmentTreePlotting:
    def _init_tree_1(self):
        # 1   2
        #  \ /
        #   0
        croot = CompartmentNode(0, loc_idx=0)
        cnode1 = CompartmentNode(1, loc_idx=1)
        cnode2 = CompartmentNode(2, loc_idx=2)

        ctree = CompartmentTree(croot)
        ctree.add_node_with_parent(cnode1, croot)
        ctree.add_node_with_parent(cnode2, croot)

        self.ctree = ctree

    def _init_tree_2(self):
        # 3
        # |
        # 2
        # |
        # 1
        # |
        # 0
        croot = CompartmentNode(0, loc_idx=0)
        cnode1 = CompartmentNode(1, loc_idx=1)
        cnode2 = CompartmentNode(2, loc_idx=2)
        cnode3 = CompartmentNode(3, loc_idx=3)

        ctree = CompartmentTree(croot)
        ctree.add_node_with_parent(cnode1, croot)
        ctree.add_node_with_parent(cnode2, cnode1)
        ctree.add_node_with_parent(cnode3, cnode2)

        self.ctree = ctree

    def _init_tree_3(self):
        # 4 5 6 7   8
        #  \|/   \ /
        #   1  2  3
        #    \ | /
        #     \|/
        #      0
        cns = [CompartmentNode(ii, loc_idx=ii) for ii in range(9)]

        ctree = CompartmentTree(cns[0])
        # first order children
        ctree.add_node_with_parent(cns[1], cns[0])
        ctree.add_node_with_parent(cns[2], cns[0])
        ctree.add_node_with_parent(cns[3], cns[0])
        # second order children
        ctree.add_node_with_parent(cns[4], cns[1])
        ctree.add_node_with_parent(cns[5], cns[1])
        ctree.add_node_with_parent(cns[6], cns[1])
        ctree.add_node_with_parent(cns[7], cns[3])
        ctree.add_node_with_parent(cns[8], cns[3])

        self.ctree = ctree

    def test_plot(self, pshow=False):
        pl.figure("trees", figsize=(9, 4))
        ax1, ax2, ax3 = pl.subplot(131), pl.subplot(132), pl.subplot(133)

        self._init_tree_1()
        self.ctree.plot_dendrogram(ax1, plotargs={"lw": 1, "c": "k"})

        self._init_tree_2()
        self.ctree.plot_dendrogram(
            ax2,
            plotargs={"lw": 1, "c": "DarkGrey"},
            labelargs={"marker": "o", "ms": 6, "mfc": "y", "mec": "r"},
        )

        self._init_tree_3()
        labelargs = {
            0: {"marker": "o", "ms": 6, "mfc": "y", "mec": "r"},
            3: {"marker": "s", "ms": 10, "mfc": "c", "mec": "g"},
            5: {"marker": "v", "ms": 12, "mfc": "c", "mec": "k"},
        }
        nodelabels = {1: "1", 4: ":-o", 8: ":-)", 9: ":-("}
        textargs = {"fontsize": 10}
        self.ctree.plot_dendrogram(
            ax3,
            plotargs={"lw": 1, "c": "k"},
            labelargs=labelargs,
            nodelabels=nodelabels,
            textargs=textargs,
        )

        if pshow:
            pl.show()


if __name__ == "__main__":
    tcomp = TestCompartmentTree()
    tcomp.test_string_representation()
    tcomp.test_tree_derivation()
    tcomp.testFitting()
    tcomp.test_reordering()
    tcomp.test_location_mapping()
    tcomp.test_gss_fit()
    tcomp.test_c_fit()
    tcomp.test_pas_functionality()
    tcomp.test_channel_fit()

    tplot = TestCompartmentTreePlotting()
    tplot.test_plot(pshow=True)
