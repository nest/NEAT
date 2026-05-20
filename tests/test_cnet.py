# -*- coding: utf-8 -*-
#
# test_cnet.py
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

from neat import netsim
from neat import NETNode, NET, Kernel
from neat import GreensTree, NeuronSimTree, SOVTree

from neat.channels.channelcollection import channelcollection

MORPHOLOGIES_PATH_PREFIX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_morphologies")
)

# load the default neuron model
import channel_installer

channel_installer.load_or_install_neuron_test_channels()


class TestCNET:
    def create_point_neurons(self, v_eq=-75.0):
        self.v_eq = v_eq
        self.dt = 0.025
        gh, eh = 50.0, -43.0
        h_chan = channelcollection.h()

        self.greens_tree = GreensTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc")
        )
        self.greens_tree.set_physiology(1.0, 100.0 / 1e6)
        self.greens_tree.add_channel_current(h_chan, gh, eh)
        self.greens_tree.fit_leak_current(v_eq, 10.0)
        self.greens_tree.set_v_ep(v_eq)
        self.greens_tree_pas = GreensTree(self.greens_tree)
        self.greens_tree_pas.as_passive_membrane()
        self.sim_tree = NeuronSimTree(self.greens_tree)
        # set the impedances
        self.greens_tree_pas.set_comp_tree()
        self.freqs = np.array([0.0])
        self.greens_tree_pas.set_impedance(self.freqs)
        # create sov tree
        self.sov_tree = SOVTree(self.greens_tree_pas)
        self.sov_tree.calc_sov_equations(maxspace_freq=50.0)

        z_inp = self.greens_tree_pas.calc_zf((1, 0.5), (1, 0.5))[0]
        alphas, gammas = self.sov_tree.get_sov_matrices(loc_arg=[(1.0, 0.5)])
        # create NET
        node_0 = NETNode(0, [0], [0], z_kernel=(alphas, gammas[:, 0] ** 2))
        net_py = NET()
        net_py.set_root(node_0)
        # check if correct
        assert np.abs(gammas[0, 0] ** 2 / np.abs(alphas[0]) - z_inp) < 1e-10
        assert np.abs(node_0.z_bar - z_inp) < 1e-10

        # to initialize neuron tree
        self.sim_tree.init_model(dt=self.dt)
        # add ion channel to NET simulator
        a_soma = 4.0 * np.pi * (self.sim_tree[1].R * 1e-4) ** 2
        self.cnet = netsim.NETSim(net_py, v_eq=self.v_eq)

        hchan = channelcollection.h()
        self.cnet.add_channel(hchan, 0, gh * a_soma, eh)

        # add the synapse
        # to neuron tree
        self.sim_tree.add_double_exp_synapse((1, 0.5), 0.2, 3.0, 0.0)
        self.sim_tree.set_spiketrain(0, 0.001, [5.0])
        # to net sim
        self.cnet.add_synapse(0, {"tau_r": 0.2, "tau_d": 3.0, "e_r": 0.0}, g_max=0.001)
        self.cnet.set_spiketimes(0, [5.0 + self.dt])

    def create_tree(self, reinitialize=1, v_eq=-75.0):
        """
        Create simple NET structure

        2     3
        |     |
        |     |
        ---1---
           |
           |
           0
           |
        """
        self.v_eq = v_eq
        loc_idx = np.array([0, 1, 2])

        # kernel constants
        alphas = 1.0 / np.array([0.5, 8.0])
        gammas = np.array([-1.0, 1.0])
        alphas_ = 1.0 / np.array([1.0])
        gammas_ = np.array([1.0])
        # nodes
        node_0 = NETNode(0, [0, 1, 2], [], z_kernel=(alphas, gammas))
        node_1 = NETNode(1, [0, 1, 2], [0], z_kernel=(alphas_, gammas_))
        node_2 = NETNode(2, [1], [1], z_kernel=(alphas_, gammas_))
        node_3 = NETNode(3, [2], [2], z_kernel=(alphas_, gammas_))
        # add nodes to tree
        net_py = NET()
        net_py.set_root(node_0)
        net_py.add_node_with_parent(node_1, node_0)
        net_py.add_node_with_parent(node_2, node_1)
        net_py.add_node_with_parent(node_3, node_1)
        # store
        self.net_py = net_py
        self.cnet = netsim.NETSim(net_py, v_eq=self.v_eq)

    def create_tree2(self, reinitialize=1, add_lin=True, v_eq=-75.0):
        """
        Create simple NET structure

                3     4
                |     |
                |     |
                ---2---
             1     |
             |     |
             ---0---
                |
        """
        self.v_eq = v_eq
        loc_idx = np.array([0, 1, 2])

        # kernel constants
        alphas = 1.0 / np.array([1.0])
        gammas = np.array([1.0])
        # nodes
        node_0 = NETNode(0, [0, 1, 2], [], z_kernel=(alphas, gammas))
        node_1 = NETNode(1, [0], [0], z_kernel=(alphas, gammas))
        node_2 = NETNode(2, [1, 2], [], z_kernel=(alphas, gammas))
        node_3 = NETNode(3, [1], [1], z_kernel=(alphas, gammas))
        node_4 = NETNode(4, [2], [2], z_kernel=(alphas, gammas))
        # add nodes to tree
        net_py = NET()
        net_py.set_root(node_0)
        net_py.add_node_with_parent(node_1, node_0)
        net_py.add_node_with_parent(node_2, node_0)
        net_py.add_node_with_parent(node_3, node_2)
        net_py.add_node_with_parent(node_4, node_2)
        # linear terms
        alphas = 1.0 / np.array([1.0])
        gammas = np.array([1.0])
        self.lin_terms = (
            {1: Kernel((alphas, gammas)), 2: Kernel((alphas, gammas))}
            if add_lin
            else {}
        )
        # store
        self.net_py = net_py
        self.cnet = netsim.NETSim(net_py, lin_terms=self.lin_terms, v_eq=self.v_eq)

    def create_tree3(self, reinitialize=1, add_lin=True, v_eq=-75.0):
        """
        Create simple NET structure

                         6
                4     5  |
                |     |  |
                |     |  |
             2  ---3---  |
             |     |     |
             ---1---     |
                   |     |
                   0------
                   |
        """
        self.v_eq = v_eq

        # kernel constants
        alphas = 1.0 / np.array([1.0])
        gammas = np.array([1.0])
        # nodes
        node_0 = NETNode(0, [0, 1, 2, 3], [], z_kernel=(alphas, gammas))
        node_1 = NETNode(1, [0, 1, 2], [], z_kernel=(alphas, gammas))
        node_2 = NETNode(2, [0], [0], z_kernel=(alphas, gammas))
        node_3 = NETNode(3, [1, 2], [], z_kernel=(alphas, gammas))
        node_4 = NETNode(4, [1], [1], z_kernel=(alphas, gammas))
        node_5 = NETNode(5, [2], [2], z_kernel=(alphas, gammas))
        node_6 = NETNode(6, [3], [3], z_kernel=(alphas, gammas))
        # add nodes to tree
        net_py = NET()
        net_py.set_root(node_0)
        net_py.add_node_with_parent(node_1, node_0)
        net_py.add_node_with_parent(node_2, node_1)
        net_py.add_node_with_parent(node_3, node_1)
        net_py.add_node_with_parent(node_4, node_3)
        net_py.add_node_with_parent(node_5, node_3)
        net_py.add_node_with_parent(node_6, node_0)
        # linear terms
        alphas = 1.0 / np.array([1.0])
        gammas = np.array([1.0])
        self.lin_terms = (
            {
                1: Kernel((alphas, gammas)),
                2: Kernel((alphas, gammas)),
                3: Kernel((alphas, gammas)),
            }
            if add_lin
            else {}
        )
        # store
        self.net_py = net_py
        self.cnet = netsim.NETSim(net_py, lin_terms=self.lin_terms)

    def test_io_functions(self):
        self.create_tree()
        # storing and reading voltages from node voltage
        vnode = np.array([8.0, 10.0, 12.0, 14.0])
        self.cnet.set_v_node_from_v_node(vnode)
        vnode_back1 = self.cnet.get_v_node()
        vnode_back2 = np.zeros(4)
        self.cnet.add_v_node_to_arr(vnode_back2)
        assert np.allclose(vnode_back1, vnode)
        assert np.allclose(vnode_back2, vnode)
        vloc_back1 = self.cnet.get_v_loc()
        vloc_back2 = np.zeros(3)
        self.cnet.add_v_loc_to_arr(vloc_back2)
        assert np.allclose(vloc_back1, np.array([18.0, 30.0, 32.0]) + self.v_eq)
        assert np.allclose(vloc_back2, np.array([18.0, 30.0, 32.0]) + self.v_eq)
        with pytest.raises(ValueError):
            self.cnet.set_v_node_from_v_node(np.zeros(3))
        with pytest.raises(ValueError):
            self.cnet.add_v_node_to_arr(np.zeros(3))
        with pytest.raises(ValueError):
            self.cnet.set_v_node_from_v_loc(np.zeros(4))
        with pytest.raises(ValueError):
            self.cnet.add_v_loc_to_arr(np.zeros(4))
        # storing and reading voltages from location voltage
        vloc = np.array([12.0, 14.0, 16.0]) + self.v_eq
        self.cnet.set_v_node_from_v_loc(vloc)
        vnode_back1 = self.cnet.get_v_node()
        vnode_back2 = np.zeros(4)
        self.cnet.add_v_node_to_arr(vnode_back2)
        assert np.allclose(vnode_back1, np.array([0.0, 12.0, 2.0, 4.0]))
        assert np.allclose(vnode_back2, np.array([0.0, 12.0, 2.0, 4.0]))
        vloc_back1 = self.cnet.get_v_loc()
        vloc_back2 = np.zeros(3)
        self.cnet.add_v_loc_to_arr(vloc_back2)
        assert np.allclose(vloc_back1, vloc)
        assert np.allclose(vloc_back2, vloc)
        with pytest.raises(ValueError):
            self.cnet.set_v_node_from_v_node(np.zeros(3))
        with pytest.raises(ValueError):
            self.cnet.add_v_node_to_arr(np.zeros(3))
        with pytest.raises(ValueError):
            self.cnet.set_v_node_from_v_loc(np.zeros(4))
        with pytest.raises(ValueError):
            self.cnet.add_v_loc_to_arr(np.zeros(4))

    def test_solver(self):
        self.create_tree()
        netp = self.net_py
        # test if single AMPA synapse agrees with analytical solution
        # add synapse
        self.cnet.add_synapse(1, "AMPA")
        g_syn = 1.0
        g_list = [np.array([]), np.array([g_syn]), np.array([])]
        # solve numerically
        v_loc = self.cnet.solve_newton(g_list)
        v_node = self.cnet.get_v_node()
        # solve analytically
        g_rescale = g_syn / (1.0 + netp[2].z_bar * g_syn)
        z_0plus1 = netp[0].z_bar + netp[1].z_bar
        v_0plus1 = (
            z_0plus1 * g_rescale / (1.0 + z_0plus1 * g_rescale) * (0.0 - self.v_eq)
        )
        v_2 = netp[2].z_bar * g_rescale * (0.0 - self.v_eq - v_0plus1)
        # test if both solutions agree
        assert np.abs(v_node[0] + v_node[1] - v_0plus1) < 1e-9
        assert np.abs(v_node[2] - v_2) < 1e-9
        assert np.abs(v_node[3] - 0.0) < 1e-9
        assert np.abs(v_loc[0] - self.v_eq - v_0plus1) < 1e-9
        assert np.abs(v_loc[1] - self.v_eq - v_0plus1 - v_2) < 1e-9
        assert np.abs(v_loc[2] - self.v_eq - v_0plus1) < 1e-9
        # test if AMPA and GABA synapses agree with analytical solution
        # add synapse
        self.cnet.add_synapse(2, "GABA")
        g_exc = 1.0
        g_inh = 1.0
        g_list = [np.array([]), np.array([g_exc]), np.array([g_inh])]
        # solve numerically
        v_loc = self.cnet.solve_newton(g_list)
        v_node = self.cnet.get_v_node()
        # solve analytically
        g_exc_ = g_exc / (1.0 + netp[2].z_bar * g_exc)
        g_inh_ = g_inh / (1.0 + netp[3].z_bar * g_inh)
        z_0plus1 = netp[0].z_bar + netp[1].z_bar
        v_0plus1 = z_0plus1 * g_exc_ / (1.0 + z_0plus1 * (g_exc_ + g_inh_)) * (
            0.0 - self.v_eq
        ) + z_0plus1 * g_inh_ / (1.0 + z_0plus1 * (g_exc_ + g_inh_)) * (
            -80.0 - self.v_eq
        )
        v_2 = netp[2].z_bar * g_exc_ * (0.0 - self.v_eq - v_0plus1)
        v_3 = netp[3].z_bar * g_inh_ * (-80.0 - self.v_eq - v_0plus1)
        # test if both solutions agree
        assert np.abs(v_node[0] + v_node[1] - v_0plus1) < 1e-9
        assert np.abs(v_node[2] - v_2) < 1e-9
        assert np.abs(v_node[3] - v_3) < 1e-9
        assert np.abs(v_loc[0] - self.v_eq - v_0plus1) < 1e-9
        assert np.abs(v_loc[1] - self.v_eq - v_0plus1 - v_2) < 1e-9
        assert np.abs(v_loc[2] - self.v_eq - v_0plus1 - v_3) < 1e-9
        # test if NMDA synapse is solved correctly
        # check if removing synapse works correctly
        self.cnet.remove_synapse_from_loc(1, 0)
        self.cnet.remove_synapse_from_loc(2, 0)
        with pytest.raises(IndexError):
            self.cnet.remove_synapse_from_loc(3, 0)
        with pytest.raises(IndexError):
            self.cnet.remove_synapse_from_loc(1, 2)
        # create NMDA synapse
        self.cnet.add_synapse(1, "NMDA")
        # solve for low conductance
        g_syn_low = 1.0
        g_list = [np.array([]), np.array([g_syn_low]), np.array([])]
        # solve numerically
        v_loc_low = self.cnet.solve_newton(g_list)
        v_node_low = self.cnet.get_v_node()
        # solve for high conductance
        g_syn_high = 4.0
        g_list = [np.array([]), np.array([g_syn_high]), np.array([])]
        # solve numerically
        v_loc_high = self.cnet.solve_newton(g_list)
        v_node_high = self.cnet.get_v_node()
        # solve for moderate conductance
        g_syn_middle = 2.0
        g_list = [np.array([]), np.array([g_syn_middle]), np.array([])]
        # solve numerically
        v_loc_middle_0 = self.cnet.solve_newton(g_list)
        v_node_middle_0 = self.cnet.get_v_node()
        v_loc_middle_1 = self.cnet.solve_newton(
            g_list, v_0=np.array([0.0, 0.0, 0.0]), v_alt=self.v_eq * np.ones(3)
        )
        v_node_middle_1 = self.cnet.get_v_node()
        # check if correct
        z_sum = netp[0].z_bar + netp[1].z_bar + netp[2].z_bar
        checkfun = lambda vv: (0.0 - vv) / (1.0 + 0.3 * np.exp(-0.1 * vv))
        vv = v_loc_low[1]
        assert np.abs(vv - self.v_eq - z_sum * g_syn_low * checkfun(vv)) < 1e-3
        vv = v_loc_high[1]
        assert np.abs(vv - self.v_eq - z_sum * g_syn_high * checkfun(vv)) < 1e-3
        vv = v_loc_middle_0[1]
        assert np.abs(vv - self.v_eq - z_sum * g_syn_middle * checkfun(vv)) < 1e-3
        vv = v_loc_middle_1[1]
        assert np.abs(vv - self.v_eq - z_sum * g_syn_middle * checkfun(vv)) < 1e-3
        assert np.abs(v_loc_middle_0[1] - v_loc_middle_1[1]) > 10.0

    def test_integration(self):
        tmax = 1000.0
        dt = 0.025
        self.create_tree()
        # add synapse and check additional synapse functions
        self.cnet.add_synapse(1, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(1, "AMPA+NMDA", g_max=1.0, nmda_ratio=5.0)
        assert self.cnet.syn_map_py[0] == {
            "loc_idxex": 1,
            "syn_index_at_loc": 0,
            "n_syn_at_loc": 1,
            "g_max": [dt * 0.1],
        }
        assert self.cnet.syn_map_py[1] == {
            "loc_idxex": 1,
            "syn_index_at_loc": 1,
            "n_syn_at_loc": 2,
            "g_max": [1.0, 5.0],
        }
        assert self.cnet.n_syn[1] == 3
        with pytest.raises(ValueError):
            self.cnet.add_synapse(1, "NONSENSE")
        with pytest.raises(IndexError):
            self.cnet.add_synapse(8, "AMPA")
        with pytest.raises(TypeError):
            self.cnet.add_synapse(1, ["NONSENSE LIST"])
        # check if synapse is correctly removed
        self.cnet.remove_synapse(1)
        assert len(self.cnet.syn_map_py) == 1
        # add spike times
        self.cnet.set_spiketimes(0, np.arange(dt / 2.0, tmax, dt))
        # run sim
        res = self.cnet.run_sim(
            tmax, dt, step_skip=1, rec_v_node=True, rec_g_syn_inds=[0]
        )
        v_loc_sim = res["v_loc"][:, -1]
        # solve newton
        v_loc_newton = self.cnet.solve_newton([res["g_syn"][0][0][-1]])
        # compare
        assert np.allclose(v_loc_sim, v_loc_newton, atol=0.5)
        # do again with other synapses
        self.cnet.remove_synapse(0)
        self.cnet.add_synapse(2, "GABA", g_max=dt * 0.1)
        self.cnet.add_synapse(1, "AMPA", g_max=dt * 0.1)
        # add spike times
        self.cnet.set_spiketimes(0, np.arange(dt / 2.0, tmax, dt))  # GABA synapse
        self.cnet.set_spiketimes(1, np.arange(dt / 2.0, tmax, dt))  # AMPA synapse
        # run sim
        res = self.cnet.run_sim(
            tmax, dt, step_skip=1, rec_v_node=True, rec_g_syn_inds=[0, 1]
        )
        v_loc_sim = res["v_loc"][:, -1]
        g_newton = [res["g_syn"][ii][0][-1] for ii in [0, 1]]
        # solve newton
        v_loc_newton = self.cnet.solve_newton(g_newton)
        # compare
        assert np.allclose(v_loc_sim, v_loc_newton, atol=0.5)

        # add NMDA synapse
        self.cnet.add_synapse(0, "AMPA+NMDA", g_max=dt * 0.1, nmda_ratio=5.0)
        self.cnet.add_synapse(1, "AMPA+NMDA", g_max=dt * 0.1, nmda_ratio=5.0)
        # set spiketimes for second synapse
        self.cnet.set_spiketimes(3, np.arange(dt / 2.0, tmax, dt))  # AMPA+NMDA synapse
        # remove first AMPA+NMDA synapse to see if spike times are correctly re-allocated
        assert len(self.cnet.spike_times_py[2]) == 0
        self.cnet.remove_synapse(2)
        for spk_tm in self.cnet.spike_times_py:
            assert len(spk_tm) > 0
        # run sim
        res = self.cnet.run_sim(
            tmax, dt, step_skip=1, rec_v_node=True, rec_g_syn_inds=[0, 1, 2]
        )
        v_loc_sim = res["v_loc"][:, -1]
        g_newton = [res["g_syn"][ii][0][-1] for ii in [0, 1, 2]]
        # solve newton
        v_loc_newton = self.cnet.solve_newton(g_newton)
        # compare
        assert np.allclose(v_loc_sim, v_loc_newton, atol=0.5)

        # test whether sparse storage works
        ss = 33  # number of timesteps not a multiple of storage step
        # set spiketimes
        self.cnet.set_spiketimes(0, np.array([5.0]))
        self.cnet.set_spiketimes(1, np.array([10.0]))
        # run sim
        res1 = self.cnet.run_sim(
            tmax, dt, step_skip=1, rec_v_node=True, rec_g_syn_inds=[0, 1, 2]
        )
        # set spiketimes
        self.cnet.set_spiketimes(0, np.array([5.0]))
        self.cnet.set_spiketimes(1, np.array([10.0]))
        # run sim
        res2 = self.cnet.run_sim(
            tmax, dt, step_skip=ss, rec_v_node=True, rec_g_syn_inds=[0, 1, 2]
        )
        # check if results are idendtical
        assert len(res2["t"]) == len(res2["v_loc"][0])
        np.allclose(res1["v_loc"][0][ss - 1 :][::ss], res2["v_loc"][0])
        np.allclose(res1["v_node"][0][ss - 1 :][::ss], res2["v_node"][0])
        np.allclose(res1["g_syn"][0][0][ss - 1 :][::ss], res2["g_syn"][0])
        # test whether sparse storage works
        ss = 20  # number of timesteps a multiple of storage step
        # set spiketimes
        self.cnet.set_spiketimes(0, np.array([5.0]))
        self.cnet.set_spiketimes(1, np.array([10.0]))
        # run sim
        res1 = self.cnet.run_sim(
            tmax, dt, step_skip=1, rec_v_node=True, rec_g_syn_inds=[0, 1, 2]
        )
        # set spiketimes
        self.cnet.set_spiketimes(0, np.array([5.0]))
        self.cnet.set_spiketimes(1, np.array([10.0]))
        # run sim
        res2 = self.cnet.run_sim(
            tmax, dt, step_skip=ss, rec_v_node=True, rec_g_syn_inds=[0, 1, 2]
        )
        # check if results are idendtical
        assert len(res2["t"]) == len(res2["v_loc"][0])
        np.allclose(res1["v_loc"][0][ss - 1 :][::ss], res2["v_loc"][0])
        np.allclose(res1["v_node"][0][ss - 1 :][::ss], res2["v_node"][0])
        np.allclose(res1["g_syn"][0][0][ss - 1 :][::ss], res2["g_syn"][0])

    def test_inversion(self):
        dt = 0.1

        # tests without linear terms
        # test with two non-leafs that integrate soma
        self.create_tree2(add_lin=False)
        # add synapses
        self.cnet.add_synapse(0, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(1, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(2, "AMPA", g_max=dt * 0.1)
        # initialize
        self.cnet.initialize(dt=dt, mode=1)
        # construct inputs
        g_in = self.cnet.recast_input([1.0, 1.0, 1.0])
        self.cnet._construct_input(np.array([self.v_eq, self.v_eq, self.v_eq]), g_in)
        # recursive matrix inversion
        self.cnet.invert_matrix()
        v_node = self.cnet.get_v_node()
        # construct full matrix
        mat, vec = self.cnet.get_mat_and_vec(dt=dt)
        # full matrix inversion
        v_sol = np.linalg.solve(mat, vec)
        # test
        assert np.allclose(v_sol, v_node)
        # test with two non-leafs that integrate soma
        self.create_tree3(add_lin=False)
        # add synapses
        self.cnet.add_synapse(0, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(1, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(2, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(3, "AMPA", g_max=dt * 0.1)
        # initialize
        self.cnet.initialize(dt=dt, mode=1)
        # construct inputs
        g_in = self.cnet.recast_input(np.ones(4))
        self.cnet._construct_input(self.v_eq * np.ones(4), g_in)
        # recursive matrix inversion
        self.cnet.invert_matrix()
        v_node = self.cnet.get_v_node()
        # construct full matrix
        mat, vec = self.cnet.get_mat_and_vec(dt=dt)
        # full matrix inversion
        v_sol = np.linalg.solve(mat, vec)
        # test
        assert np.allclose(v_sol, v_node)

        # tests with linear terms
        # test with one non-leafs that integrate soma
        self.create_tree2(add_lin=True)
        # add synapses
        self.cnet.add_synapse(0, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(1, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(2, "AMPA", g_max=dt * 0.1)
        # initialize
        self.cnet.initialize(dt=dt, mode=1)
        # construct inputs
        g_in = self.cnet.recast_input([1.0, 1.0, 1.0])
        self.cnet._construct_input(np.array([self.v_eq, self.v_eq, self.v_eq]), g_in)
        # recursive matrix inversion
        self.cnet.invert_matrix()
        v_node = self.cnet.get_v_node()
        # construct full matrix
        mat, vec = self.cnet.get_mat_and_vec(dt=dt)
        # full matrix inversion
        v_sol = np.linalg.solve(mat, vec)
        # test
        assert np.allclose(v_sol, v_node)
        # test with two non-leafs that integrate soma
        self.create_tree3(add_lin=True)
        # add synapses
        self.cnet.add_synapse(0, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(1, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(2, "AMPA", g_max=dt * 0.1)
        self.cnet.add_synapse(3, "AMPA", g_max=dt * 0.1)
        # initialize
        self.cnet.initialize(dt=dt, mode=1)
        # construct inputs
        g_in = self.cnet.recast_input(np.ones(4))
        self.cnet._construct_input(self.v_eq * np.ones(4), g_in)
        # recursive matrix inversion
        self.cnet.invert_matrix()
        v_node = self.cnet.get_v_node()
        # construct full matrix
        mat, vec = self.cnet.get_mat_and_vec(dt=dt)
        # full matrix inversion
        v_sol = np.linalg.solve(mat, vec)
        # test
        assert np.allclose(v_sol, v_node)

    def test_channel(self):
        self.create_point_neurons()
        # simulate neuron and NET model
        res_neuron = self.sim_tree.run(100)
        res_net = self.cnet.run_sim(100.0, self.dt)
        # test if traces equal
        assert np.allclose(res_neuron["v_m"][0, :-1], res_net["v_loc"][0, :], atol=0.1)


if __name__ == "__main__":
    tst = TestCNET()
    tst.test_io_functions()
    tst.test_solver()
    tst.test_integration()
    tst.test_inversion()
    tst.test_channel()
