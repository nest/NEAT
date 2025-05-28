# -*- coding: utf-8 -*-
#
# test_phystree.py
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
import copy

from neat import PhysTree, MorphTree, GreensTree, CompartmentFitter

import channelcollection_for_tests as channelcollection
import channel_installer

channel_installer.load_or_install_neuron_test_channels()


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_morphologies")
)


class TestPhysTree:
    def load_tree(self, segments=False):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        """
        fname = "Ttree_segments.swc" if segments else "Ttree.swc"
        self.tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, fname), types=[1, 3, 4]
        )

    def load_segments_tree(self):
        """
        Load ball and stick dendrite where each segment is decreasing in radius

        1--4--5--6--7--8
        """
        self.tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball_and_stick_segments.swc"),
            types=[1, 3, 4],
        )

    def load_sticks_tree(self):
        """
        Load ball and some sticks dendrite
        """
        self.tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball_and_some_sticks.swc"),
            types=[1, 3, 4],
        )

    def test_string_representation(self):
        self.load_tree()

        # gmax as potential as float
        e_rev = 100.0
        g_max = 100.0
        channel = channelcollection.test_channel2()
        self.tree.add_channel_current(channel, g_max, e_rev)

        assert (
            str(self.tree) == ">>> PhysTree\n"
            "    PhysNode 1, Parent: None --- r_a = 0.0001 MOhm*cm, c_m = 1 uF/cm^2, v_ep = -75 mV, (g_test_channel2 = 100 uS/cm^2, e_test_channel2 = 100 mV)\n"
            "    PhysNode 4, Parent: 1 --- r_a = 0.0001 MOhm*cm, c_m = 1 uF/cm^2, v_ep = -75 mV, (g_test_channel2 = 100 uS/cm^2, e_test_channel2 = 100 mV)\n"
            "    PhysNode 5, Parent: 4 --- r_a = 0.0001 MOhm*cm, c_m = 1 uF/cm^2, v_ep = -75 mV, (g_test_channel2 = 100 uS/cm^2, e_test_channel2 = 100 mV)\n"
            "    PhysNode 6, Parent: 5 --- r_a = 0.0001 MOhm*cm, c_m = 1 uF/cm^2, v_ep = -75 mV, (g_test_channel2 = 100 uS/cm^2, e_test_channel2 = 100 mV)\n"
            "    PhysNode 7, Parent: 4 --- r_a = 0.0001 MOhm*cm, c_m = 1 uF/cm^2, v_ep = -75 mV, (g_test_channel2 = 100 uS/cm^2, e_test_channel2 = 100 mV)\n"
            "    PhysNode 8, Parent: 7 --- r_a = 0.0001 MOhm*cm, c_m = 1 uF/cm^2, v_ep = -75 mV, (g_test_channel2 = 100 uS/cm^2, e_test_channel2 = 100 mV)"
        )

        repr_str = (
            "['PhysTree', "
            "\"{'node index': 1, 'parent index': -1, 'content': '{}', 'xyz': array([0., 0., 0.]), 'R': '10', 'swc_type': 1, 'currents': {'test_channel2': '(100, 100)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 4, 'parent index': 1, 'content': '{}', 'xyz': array([100.,   0.,   0.]), 'R': '1', 'swc_type': 4, 'currents': {'test_channel2': '(100, 100)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 5, 'parent index': 4, 'content': '{}', 'xyz': array([100.,  50.,   0.]), 'R': '1', 'swc_type': 4, 'currents': {'test_channel2': '(100, 100)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 6, 'parent index': 5, 'content': '{}', 'xyz': array([100., 100.,   0.]), 'R': '0.5', 'swc_type': 4, 'currents': {'test_channel2': '(100, 100)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 7, 'parent index': 4, 'content': '{}', 'xyz': array([100., -50.,   0.]), 'R': '1', 'swc_type': 4, 'currents': {'test_channel2': '(100, 100)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 8, 'parent index': 7, 'content': '{}', 'xyz': array([ 100., -100.,    0.]), 'R': '0.5', 'swc_type': 4, 'currents': {'test_channel2': '(100, 100)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\""
            "]"
            "{'channel_storage': ['test_channel2']}"
        )

        assert repr(self.tree) == repr_str

    def test_leak_distr(self):
        self.load_tree()
        with pytest.raises(AssertionError):
            self.tree.fit_leak_current(-75.0, -10.0)
        # test simple distribution
        self.tree.fit_leak_current(-75.0, 10.0)
        for node in self.tree:
            assert np.abs(node.c_m - 1.0) < 1e-9
            assert np.abs(node.currents["L"][0] - 1.0 / (10.0 * 1e-3)) < 1e-9
            assert np.abs(node.v_ep + 75.0) < 1e-9
        # create complex distribution
        tau_distr = lambda x: x + 100.0
        for node in self.tree:
            d2s = self.tree.path_length({"node": node.index, "x": 1.0}, (1.0, 0.5))
            node.fit_leak_current(
                self.tree.channel_storage,
                e_eq_target=-75.0,
                tau_m_target=tau_distr(d2s),
            )
            assert np.abs(node.c_m - 1.0) < 1e-9
            assert np.abs(node.currents["L"][0] - 1.0 / (tau_distr(d2s) * 1e-3)) < 1e-9
            assert np.abs(node.v_ep + 75.0) < 1e-9

    def test_physiology_setting(self):
        self.load_tree()
        d2s = {1: 0.0, 4: 50.0, 5: 125.0, 6: 175.0, 7: 125.0, 8: 175.0}
        # passive parameters as float
        c_m = 1.0
        r_a = 100.0 * 1e-6
        self.tree.set_physiology(c_m, r_a)
        for node in self.tree:
            assert np.abs(node.c_m - c_m) < 1e-10
            assert np.abs(node.r_a - r_a) < 1e-10
        # passive parameters as function
        c_m = lambda x: 0.5 * x + 1.0
        r_a = lambda x: np.exp(0.01 * x) * 100 * 1e-6
        self.tree.set_physiology(c_m, r_a)
        for node in self.tree:
            assert np.abs(node.c_m - c_m(d2s[node.index])) < 1e-10
            assert np.abs(node.r_a - r_a(d2s[node.index])) < 1e-10
        # passive parameters as incomplete dict
        r_a = 100.0 * 1e-6
        c_m = {1: 1.0, 4: 1.2}
        with pytest.raises(KeyError):
            self.tree.set_physiology(c_m, r_a)
        # passive parameters as complete dict
        c_m.update({5: 1.1, 6: 0.9, 7: 0.8, 8: 1.0})
        self.tree.set_physiology(c_m, r_a)
        for node in self.tree:
            assert np.abs(node.c_m - c_m[node.index]) < 1e-10

        # equilibrium potential as float
        e_eq = -75.0
        self.tree.set_v_ep(e_eq)
        for node in self.tree:
            assert np.abs(node.v_ep - e_eq) < 1e-10
        # equilibrium potential as dict
        e_eq = {1: -75.0, 4: -74.0, 5: -73.0, 6: -72.0, 7: -71.0, 8: -70.0}
        self.tree.set_v_ep(e_eq)
        for node in self.tree:
            assert np.abs(node.v_ep - e_eq[node.index]) < 1e-10
        # equilibrium potential as function
        e_eq = lambda x: -70.0 + 0.1 * x
        self.tree.set_v_ep(e_eq)
        for node in self.tree:
            assert np.abs(node.v_ep - e_eq(d2s[node.index])) < 1e-10
        # as wrong type
        with pytest.raises(TypeError):
            self.tree.set_v_ep([])
            self.tree.set_physiology([], [])

        # leak as float
        g_l, e_l = 100.0, -75.0
        self.tree.set_leak_current(g_l, e_l)
        for node in self.tree:
            g, e = node.currents["L"]
            assert np.abs(g - g_l) < 1e-10
            assert np.abs(e - e_l) < 1e-10
        # equilibrium potential as dict
        g_l = {1: 101.0, 4: 103.0, 5: 105.0, 6: 107.0, 7: 108.0, 8: 109.0}
        e_l = {1: -75.0, 4: -74.0, 5: -73.0, 6: -72.0, 7: -71.0, 8: -70.0}
        self.tree.set_leak_current(g_l, e_l)
        for node in self.tree:
            g, e = node.currents["L"]
            assert np.abs(g - g_l[node.index]) < 1e-10
            assert np.abs(e - e_l[node.index]) < 1e-10
        # equilibrium potential as function
        g_l = lambda x: 100.0 + 0.05 * x
        e_l = lambda x: -70.0 + 0.05 * x
        self.tree.set_leak_current(g_l, e_l)
        for node in self.tree:
            g, e = node.currents["L"]
            assert np.abs(g - g_l(d2s[node.index])) < 1e-10
            assert np.abs(e - e_l(d2s[node.index])) < 1e-10
        # as wrong type
        with pytest.raises(TypeError):
            self.tree.set_leak_current([])
        # wrong sign
        with pytest.raises(AssertionError):
            self.tree.set_leak_current(e_l, g_l)

        # gmax as potential as float
        e_rev = 100.0
        g_max = 100.0
        channel = channelcollection.test_channel2()
        self.tree.add_channel_current(channel, g_max, e_rev)
        for node in self.tree:
            g_m = node.currents["test_channel2"][0]
            assert np.abs(g_m - g_max) < 1e-10
        # equilibrium potential as dict
        g_max = {1: 101.0, 4: 103.0, 5: 104.0, 6: 106.0, 7: 107.0, 8: 110.0}
        self.tree.add_channel_current(channel, g_max, e_rev)
        for node in self.tree:
            g_m = node.currents["test_channel2"][0]
            assert np.abs(g_m - g_max[node.index]) < 1e-10
        # equilibrium potential as function
        g_max = lambda x: 100.0 + 0.005 * x**2
        self.tree.add_channel_current(channel, g_max, e_rev)
        for node in self.tree:
            g_m = node.currents["test_channel2"][0]
            assert np.abs(g_m - g_max(d2s[node.index])) < 1e-10
        # test is channel is stored
        assert isinstance(
            self.tree.channel_storage[channel.__class__.__name__],
            channelcollection.test_channel2,
        )
        # check if error is thrown if an ionchannel is not give
        with pytest.raises(IOError):
            self.tree.add_channel_current("test_channel2", g_max, e_rev)
        # check if error is thrown if an ionchannel is not give
        with pytest.raises(AssertionError):
            self.tree.add_channel_current(channel, -10.0, e_rev)

    def test_membrane_functions(self):
        self.load_tree()
        self.tree.set_physiology(1.0, 100 * 1e-6)
        # passive parameters
        c_m = 1.0
        r_a = 100.0 * 1e-6
        e_eq = -75.0
        self.tree.set_physiology(c_m, r_a)
        self.tree.set_v_ep(e_eq)
        # channel
        p_open = 0.9 * 0.3**3 * 0.5**2 + 0.1 * 0.4**2 * 0.6**1  # test_channel2
        g_chan, e_chan = 100.0, 100.0
        channel = channelcollection.test_channel2()
        self.tree.add_channel_current(channel, g_chan, e_chan)
        # fit the leak current
        self.tree.fit_leak_current(-30.0, 10.0)

        # test if fit was correct
        for node in self.tree:
            tau_mem = c_m / (node.currents["L"][0] + g_chan * p_open) * 1e3
            assert np.abs(tau_mem - 10.0) < 1e-10
            e_eq = (
                node.currents["L"][0] * node.currents["L"][1] + g_chan * p_open * e_chan
            ) / (node.currents["L"][0] + g_chan * p_open)
            assert np.abs(e_eq - (-30.0)) < 1e-10

        # test if warning is raised for impossible to reach time scale
        with pytest.warns(UserWarning):
            tree = copy.deepcopy(self.tree)
            tree.fit_leak_current(-30.0, 100000.0)

        # total membrane conductance
        g_pas = self.tree[1].currents["L"][0] + g_chan * p_open
        i_pas = self.tree[1].currents["L"][0] * (
            -30.0 - self.tree[1].currents["L"][1]
        ) + g_chan * p_open * (-30.0 - e_chan)
        i_pas_ = self.tree[1].calc_i_tot(self.tree.channel_storage)
        g_pas_ = self.tree[1].calc_g_tot(self.tree.channel_storage)
        # check that total current is zero at equilibrium
        assert np.abs(i_pas) < 1e-10
        assert np.abs(i_pas_) < 1e-10
        # make passive membrane
        tree = copy.deepcopy(self.tree)
        tree.as_passive_membrane()
        # test if fit was correct
        for node in tree:
            assert np.abs(node.currents["L"][0] - g_pas) < 1e-10
            assert np.abs(node.currents["L"][1] - (-30.0)) < 1e-10
        # test if channels storage is empty
        assert len(tree.channel_storage) == 0
        # test if computational root was removed
        assert tree._computational_root is None

        # test partial passification
        tree = copy.deepcopy(self.tree)
        # channel
        g_chan1, e_chan1 = 50.0, -100.0
        channel1 = channelcollection.test_channel()
        tree.add_channel_current(channel1, g_chan, e_chan)
        # passify channel 2
        tree.as_passive_membrane(channel_names=["test_channel2"])
        for node in tree:
            assert set(node.currents.keys()) == {"test_channel", "L"}
            assert np.abs(node.currents["L"][0] - g_pas) < 1e-10
            assert np.abs(node.calc_i_tot(tree.channel_storage)) < 1e-10

    def test_comp_tree(self):
        self.load_tree(segments=True)

        # capacitance axial resistance constant
        c_m = 1.0
        r_a = 100.0 * 1e-6
        self.tree.set_physiology(c_m, r_a)
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            assert [n.index for n in self.tree] == [1, 8, 10, 12]
        # capacitance and axial resistance change
        c_m = lambda x: 1.0 if x < 200.0 else 1.6
        r_a = lambda x: 1.0 if x < 300.0 else 1.6
        self.tree.set_physiology(c_m, r_a)
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            assert [n.index for n in self.tree] == [1, 5, 6, 8, 10, 12]
        # leak current changes
        g_l = lambda x: 100.0 if x < 400.0 else 160.0
        self.tree.set_leak_current(g_l, -75.0)
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            assert [n.index for n in self.tree] == [1, 5, 6, 7, 8, 10, 12]
        # leak current & reversal change
        g_l = 100.0
        e_l = {ind: -75.0 for ind in [1, 4, 5, 6, 7, 8, 11, 12]}
        e_l.update({ind: -55.0 for ind in [9, 10]})
        self.tree.set_leak_current(g_l, e_l)
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            assert [n.index for n in self.tree] == [1, 5, 6, 8, 10, 12]
        # leak current & reversal change
        g_l = 100.0
        e_l = {ind: -75.0 for ind in [1, 4, 5, 6, 7, 8, 10, 11, 12]}
        e_l.update({9: -55.0})
        self.tree.set_leak_current(g_l, e_l)
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            assert [n.index for n in self.tree] == [1, 5, 6, 8, 9, 10, 12]
        # shunt
        self.tree[7].g_shunt = 1.0
        self.tree.set_comp_tree()
        with self.tree.as_computational_tree:
            assert [n.index for n in self.tree] == [1, 5, 6, 7, 8, 9, 10, 12]

    def test_create_new_tree(self):
        self.load_tree(segments=True)
        # gmax as potential as float
        e_rev = 100.0
        g_max = 100.0
        channel = channelcollection.test_channel2()
        self.tree.add_channel_current(
            channel, {node.index: float(node.index) for node in self.tree}, e_rev
        )

        locs = [(n.index, 0.5) for n in self.tree]
        with pytest.raises(ValueError):
            self.tree.create_new_tree(locs, new_tree=MorphTree())

        new_tree = self.tree.create_new_tree(locs)
        for new_node, orig_node in zip(new_tree, self.tree):
            for channel_name in orig_node.currents:
                g_new = new_node.currents[channel_name][0]
                g_orig = orig_node.currents[channel_name][0]
                assert g_new == pytest.approx(g_orig)

    def test_finite_diff_tree(self, rtol_param=5e-2, rtol_dx=1e-10, pprint=False):
        self.load_tree(segments=1)
        # set capacitance, axial resistance
        c_m = 1.0
        r_a = 100.0 * 1e-6
        self.tree.set_physiology(c_m, r_a)
        # set leak current
        g_l, e_l = 100.0, -75.0
        self.tree.set_leak_current(g_l, e_l)
        # set computational tree
        self.tree.set_comp_tree()

        def _check_dx(ctree, locs, dx):
            for n1 in ctree:
                if not ctree.is_root(n1):
                    l_ = self.tree.path_length(
                        locs[n1.loc_idx], locs[n1.parent_node.loc_idx]
                    )
                    assert l_ <= dx + rtol_dx

        # test structure
        ctree_fd, locs_fd = self.tree.create_finite_difference_tree(dx_max=100.0)
        assert len(ctree_fd) == len(locs_fd)
        assert len(ctree_fd) == 10
        _check_dx(ctree_fd, locs_fd, dx=100.0)

        ctree_fd, locs_fd = self.tree.create_finite_difference_tree(dx_max=101.0)
        assert len(ctree_fd) == len(locs_fd)
        assert len(ctree_fd) == 10
        _check_dx(ctree_fd, locs_fd, dx=101.0)

        ctree_fd, locs_fd = self.tree.create_finite_difference_tree(dx_max=60.0)
        assert len(ctree_fd) == len(locs_fd)
        assert len(ctree_fd) == 18
        _check_dx(ctree_fd, locs_fd, dx=60.0)

        ctree_fd, locs_fd = self.tree.create_finite_difference_tree(dx_max=40.0)
        assert len(ctree_fd) == len(locs_fd)
        assert len(ctree_fd) == 24
        _check_dx(ctree_fd, locs_fd, dx=40.0)

        # create finite difference for conductance values test
        ctree_fd, locs_fd = self.tree.create_finite_difference_tree(dx_max=10.0)
        assert len(ctree_fd) == len(locs_fd)
        assert len(ctree_fd) == 91  # soma + 9 segments with 10 compartments each
        _check_dx(ctree_fd, locs_fd, dx=10.0)

        def check_tree_equivalence(locs, ctree_fd, ctree_fit, with_ca=True):
            if pprint:
                print("\nchecking tree >>>")

            for ii, loc in enumerate(locs):
                node_fd = ctree_fd.get_nodes_from_loc_idxs(ii)
                node_fit = ctree_fit.get_nodes_from_loc_idxs(ii)

                # test capacitance match
                if pprint:
                    print(f"> loc {ii}")
                    print(f"    ca_fd = {node_fd.ca}, ca_fit = {node_fit.ca}")
                if with_ca:
                    assert np.abs(node_fd.ca - node_fit.ca) < rtol_param * np.max(
                        [node_fd.ca, node_fit.ca]
                    )

                # test coupling cond match
                if not ctree_fd.is_root(node_fd):
                    if pprint:
                        print(f"    gc_fd = {node_fd.g_c}, gc_fit = {node_fit.g_c}")
                    assert np.abs(node_fd.g_c - node_fit.g_c) < rtol_param * np.max(
                        [node_fd.g_c, node_fit.g_c]
                    )

                # test leak current match
                for key in node_fd.currents:
                    g_fd = node_fd.currents[key][0]
                    g_fit = node_fit.currents[key][0]
                    if pprint:
                        print(f"    g_{key}_fd = {g_fd}, g_{key}_fit = {g_fit}")
                    assert np.abs(g_fd - g_fit) < rtol_param * np.max([g_fd, g_fit])

            if pprint:
                print("<<< tree check done\n")

        # fit a compartmenttree to the same locations
        ctree_fd, locs_fd = self.tree.create_finite_difference_tree(dx_max=22.0)
        cfit = CompartmentFitter(self.tree, save_cache=False)
        ctree_fit, _ = cfit.fit_model(locs_fd)
        # check whether both trees have the same parameters
        check_tree_equivalence(locs_fd, ctree_fd, ctree_fit)

        # compute resistance matrix of all models
        gt = GreensTree(self.tree)
        gt.set_impedance(0.0)
        z_orig = gt.calc_impedance_matrix(locs_fd)
        z_fd = ctree_fd.calc_impedance_matrix()
        z_fit = ctree_fit.calc_impedance_matrix()
        assert np.allclose(z_orig, z_fd, atol=0.5)
        assert np.allclose(z_fit, z_fd, atol=2.0)

        ###################################################
        # test tree with varying conductance densities
        self.load_tree(segments=1)
        # set capacitance, axial resistance
        c_m = 1.0
        r_a = 100.0 * 1e-6
        self.tree.set_physiology(c_m, r_a)
        # set leak current
        e_l = -75.0
        g_l = lambda x: 100.0 + 100.0 * np.exp((x - 400.0) / 400)
        self.tree.set_leak_current(g_l, e_l, node_arg="apical")
        self.tree.set_leak_current(g_l, e_l, node_arg="basal")
        self.tree.set_leak_current(200.0, e_l, node_arg="somatic")
        # set potassium current
        self.tree.add_channel_current(
            channelcollection.Kv3_1(), 700.0, -85.0, node_arg="somatic"
        )
        self.tree.add_channel_current(
            channelcollection.Kv3_1(), 200.0, -85.0, node_arg="apical"
        )
        # set computational tree
        self.tree.set_comp_tree()

        # fit a compartmenttree to the same locations
        ctree_fd, locs_fd = self.tree.create_finite_difference_tree(dx_max=52.0)
        cfit = CompartmentFitter(self.tree, save_cache=False)
        ctree_fit, _ = cfit.fit_model(locs_fd)
        # check whether both trees have the same parameters
        check_tree_equivalence(locs_fd, ctree_fd, ctree_fit, with_ca=False)

        # compute resistance matrix of all models
        gt = GreensTree(self.tree)
        gt.set_impedance(0.0)
        z_orig = gt.calc_impedance_matrix(locs_fd)
        z_fit = ctree_fit.calc_impedance_matrix()
        z_fd = ctree_fd.calc_impedance_matrix()
        assert np.allclose(z_orig, z_fd, atol=0.5)
        assert np.allclose(z_fit, z_fd, atol=2.0)
        ###################################################

        ###################################################
        # test a different morphology
        self.load_sticks_tree()
        # set capacitance, axial resistance
        c_m = 1.0
        r_a = 100.0 * 1e-6
        self.tree.set_physiology(c_m, r_a)
        # set leak current
        g_l, e_l = 100.0, -75.0
        self.tree.set_leak_current(g_l, e_l)
        # set computational tree
        self.tree.set_comp_tree()

        # fit a compartmenttree to the same locations
        ctree_fd, locs_fd = self.tree.create_finite_difference_tree(dx_max=22.0)
        cfit = CompartmentFitter(self.tree, save_cache=False)
        ctree_fit, _ = cfit.fit_model(locs_fd)
        # check whether both trees have the same parameters
        check_tree_equivalence(locs_fd, ctree_fd, ctree_fit)

        # compute resistance matrix of all models
        gt = GreensTree(self.tree)
        gt.set_impedance(0.0)
        z_orig = gt.calc_impedance_matrix(locs_fd)
        z_fd = ctree_fd.calc_impedance_matrix()
        z_fit = ctree_fit.calc_impedance_matrix()
        assert np.allclose(z_orig, z_fd, atol=1.0)
        assert np.allclose(z_fit, z_fd, atol=1.0)
        ###################################################

        ###################################################
        # test a different morphology
        self.load_segments_tree()
        # set capacitance, axial resistance
        c_m = 1.0
        r_a = 100.0 * 1e-6
        self.tree.set_physiology(c_m, r_a)
        # set leak current
        g_l, e_l = 100.0, -75.0
        self.tree.set_leak_current(g_l, e_l)
        # set computational tree
        self.tree.set_comp_tree()

        # fit a compartmenttree to the same locations
        ctree_fd, locs_fd = self.tree.create_finite_difference_tree(dx_max=22.0)
        cfit = CompartmentFitter(self.tree, save_cache=False)
        ctree_fit, _ = cfit.fit_model(locs_fd)
        # check whether both trees have the same parameters
        check_tree_equivalence(locs_fd, ctree_fd, ctree_fit)

        # compute resistance matrix of all models
        gt = GreensTree(self.tree)
        gt.set_impedance(0.0)
        z_orig = gt.calc_impedance_matrix(locs_fd)
        z_fd = ctree_fd.calc_impedance_matrix()
        z_fit = ctree_fit.calc_impedance_matrix()
        assert np.allclose(z_orig, z_fd, atol=1.0)
        assert np.allclose(z_fit, z_fd, atol=1.0)
        ###################################################


if __name__ == "__main__":
    tphys = TestPhysTree()
    tphys.test_string_representation()
    tphys.test_leak_distr()
    tphys.test_physiology_setting()
    tphys.test_membrane_functions()
    tphys.test_comp_tree()
    tphys.test_create_new_tree()
    tphys.test_finite_diff_tree(pprint=True)
