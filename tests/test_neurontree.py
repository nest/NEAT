# -*- coding: utf-8 -*-
#
# test_neurontree.py
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

import neuron
from neuron import h

import pytest
import itertools

from neat import GreensTree
from neat import CompartmentNode, CompartmentTree
from neat import NeuronSimTree, NeuronCompartmentTree
import neat.tools.kernelextraction as ke

import channelcollection_for_tests as channelcollection
import channel_installer

channel_installer.load_or_install_neuron_test_channels()


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_morphologies")
)
colours = [
    "DeepPink",
    "Purple",
    "MediumSlateBlue",
    "Blue",
    "Teal",
    "ForestGreen",
    "DarkOliveGreen",
    "DarkGoldenRod",
    "DarkOrange",
    "Coral",
    "Red",
    "Sienna",
    "Black",
    "DarkGrey",
]


class TestNeuron:
    def load_T_tree_passive(self):
        """
        Load the T-tree morphology in memory with passive conductance

          6--5--4--7--8
                |
                |
                1
        """
        self.v_eq = -75.0
        self.dt = 0.025
        self.tmax = 100.0
        # for frequency derivation
        self.ft = ke.FourierQuadrature(np.arange(0.0, self.tmax, self.dt))
        # load the morphology
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "Tsovtree.swc")
        self.greenstree = GreensTree(fname, types=[1, 3, 4])
        self.greenstree.fit_leak_current(self.v_eq, 10.0)
        self.greenstree.set_comp_tree()
        self.greenstree.set_impedance(self.ft.s)
        # copy greenstree parameters into NEURON simulation tree
        self.neurontree = NeuronSimTree(self.greenstree)
        self.neurontree.set_default_tree("computational")

    def load_T_tree_active(self):
        """
        Load the T-tree morphology in memory with h-current

          6--5--4--7--8
                |
                |
                1
        """
        self.v_eq = -75.0
        self.dt = 0.025
        self.tmax = 100.0
        # for frequency derivation
        self.ft = ke.FourierQuadrature(np.arange(0.0, self.tmax, self.dt))
        # load the morphology
        h_chan = channelcollection.h()
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "Tsovtree.swc")
        self.greenstree = GreensTree(fname, types=[1, 3, 4])
        self.greenstree.add_channel_current(h_chan, 50.0, -43.0)
        self.greenstree.fit_leak_current(self.v_eq, 10.0)
        self.greenstree.set_comp_tree()
        self.greenstree.set_impedance(self.ft.s)
        # copy greenstree parameters into NEURON simulation tree
        self.neurontree = NeuronSimTree(self.greenstree)
        self.neurontree.set_default_tree("computational")

    def load_T_tree_test_channel(self):
        """
        Load the T-tree morphology in memory with h-current

          6--5--4--7--8
                |
                |
                1
        """
        self.v_eq = -75.0
        self.dt = 0.025
        self.tmax = 100.0
        # for frequency derivation
        self.ft = ke.FourierQuadrature(np.arange(0.0, self.tmax, self.dt))
        # load the morphology
        test_chan = channelcollection.test_channel2()
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "Tsovtree.swc")
        self.greenstree = GreensTree(fname, types=[1, 3, 4])
        self.greenstree.add_channel_current(test_chan, 50.0, -23.0)
        self.greenstree.fit_leak_current(self.v_eq, 10.0)
        self.greenstree.set_comp_tree()
        self.greenstree.set_impedance(self.ft.s)
        # copy greenstree parameters into NEURON simulation tree
        self.neurontree = NeuronSimTree(self.greenstree)
        self.neurontree.set_default_tree("computational")

    def load_T_tree_test_channel_soma(self):
        """
        Load the T-tree morphology in memory with h-current

          6--5--4--7--8
                |
                |
                1
        """
        self.v_eq = -75.0
        self.dt = 0.025
        self.tmax = 100.0
        # for frequency derivation
        self.ft = ke.FourierQuadrature(np.arange(0.0, self.tmax, self.dt))
        # load the morphology
        test_chan = channelcollection.test_channel2()
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "Tsovtree.swc")
        self.greenstree = GreensTree(fname, types=[1, 3, 4])
        self.greenstree.add_channel_current(
            test_chan, 50.0, 23.0, node_arg=[self.greenstree[1]]
        )
        self.greenstree.fit_leak_current(self.v_eq, 10.0)
        self.greenstree.set_comp_tree()
        self.greenstree.set_impedance(self.ft.s)
        # copy greenstree parameters into NEURON simulation tree
        self.neurontree = NeuronSimTree(self.greenstree)
        self.neurontree.set_default_tree("computational")

    def test_passive(self, pplot=False):
        self.load_T_tree_passive()
        # set of locations
        locs = [(1, 0.5), (4, 0.5), (4, 1.0), (5, 0.5), (6, 0.5), (7, 0.5), (8, 0.5)]
        # compute impedance matrix with Green's function
        zf_mat_gf = self.greenstree.calc_impedance_matrix(locs)
        z_mat_gf = zf_mat_gf[self.ft.ind_0s].real
        # convert impedance matrix to time domain
        zk_mat_gf = np.zeros((len(self.ft.t), len(locs), len(locs)))
        for ii, jj in itertools.product(list(range(len(locs))), list(range(len(locs)))):
            zk_mat_gf[:, ii, jj] = self.ft.ft_inv(zf_mat_gf[:, ii, jj])[1].real * 1e-3
        # test the steady state impedance matrix
        z_mat_neuron = self.neurontree.calc_impedance_matrix(
            locs,
            dt=self.dt,
            t_calibrate=100.0,
            v_init=self.v_eq,
            factor_lambda=25.0,
        )
        assert np.allclose(z_mat_gf, z_mat_neuron, atol=1.0)
        # test the temporal matrix
        tk, zk_mat_neuron = self.neurontree.calc_impulse_response_matrix(
            locs,
            dt=self.dt,
            t_calibrate=100.0,
            v_init=self.v_eq,
            factor_lambda=25.0,
            dstep=-2,
        )
        nt = min(zk_mat_gf.shape[0], zk_mat_neuron.shape[0])
        assert np.allclose(
            zk_mat_gf[int(2.0 / self.dt) : nt, :, :],
            zk_mat_neuron[int(2.0 / self.dt) : nt, :, :],
            atol=0.2,
        )
        if pplot:
            # plot kernels
            pl.figure()
            cc = 0
            for ii in range(len(locs)):
                jj = 0
                while jj <= ii:
                    pl.plot(
                        tk, zk_mat_neuron[:nt, ii, jj], c=colours[cc % len(colours)]
                    )
                    pl.plot(
                        tk,
                        zk_mat_gf[:nt, ii, jj],
                        ls="--",
                        lw=2,
                        c=colours[cc % len(colours)],
                    )
                    cc += 1
                    jj += 1
            pl.show()

    def test_active(self, pplot=False):
        self.load_T_tree_active()
        # set of locations
        locs = [(1, 0.5), (4, 0.5), (6, 0.5), (7, 0.5), (8, 0.5)]
        # compute impedance matrix with Green's function
        zf_mat_gf = self.greenstree.calc_impedance_matrix(locs)
        z_mat_gf = zf_mat_gf[self.ft.ind_0s].real
        # convert impedance matrix to time domain
        zk_mat_gf = np.zeros((len(self.ft.t), len(locs), len(locs)))
        for ii, jj in itertools.product(list(range(len(locs))), list(range(len(locs)))):
            zk_mat_gf[:, ii, jj] = self.ft.ft_inv(zf_mat_gf[:, ii, jj])[1].real * 1e-3
        # test the steady state impedance matrix
        z_mat_neuron = self.neurontree.calc_impedance_matrix(
            locs,
            t_dur=500.0,
            dt=self.dt,
            t_calibrate=100.0,
            v_init=self.v_eq,
            factor_lambda=25.0,
        )
        assert np.allclose(z_mat_gf, z_mat_neuron, atol=5.0)
        # test the temporal matrix
        tk, zk_mat_neuron = self.neurontree.calc_impulse_response_matrix(
            locs,
            dt=self.dt,
            t_calibrate=100.0,
            v_init=self.v_eq,
            factor_lambda=25.0,
            dstep=-2,
        )
        nt = min(zk_mat_gf.shape[0], zk_mat_neuron.shape[0])
        assert np.allclose(
            zk_mat_gf[int(2.0 / self.dt) : nt, :, :],
            zk_mat_neuron[int(2.0 / self.dt) : nt, :, :],
            atol=0.5,
        )
        if pplot:
            # plot kernels
            pl.figure()
            cc = 0
            for ii in range(len(locs)):
                jj = 0
                while jj <= ii:
                    pl.plot(
                        tk, zk_mat_neuron[:nt, ii, jj], c=colours[cc % len(colours)]
                    )
                    pl.plot(
                        tk,
                        zk_mat_gf[:nt, ii, jj],
                        ls="--",
                        lw=2,
                        c=colours[cc % len(colours)],
                    )
                    cc += 1
                    jj += 1
            pl.show()

    @pytest.mark.parametrize("use_coreneuron", [False, True])
    def test_channel_recording(self, use_coreneuron):
        self.load_T_tree_test_channel()
        # set of locations
        locs = [(1, 0.5), (4, 0.5), (4, 1.0), (5, 0.5), (6, 0.5), (7, 0.5), (8, 0.5)]
        # create simulation tree
        self.neurontree.init_model(t_calibrate=10.0, factor_lambda=10.0)
        self.neurontree.store_locs(locs, name="rec locs")
        # run test simulation
        res = self.neurontree.run(1.0, record_from_channels=True, use_coreneuron=use_coreneuron)
        # check if results are stored correctly
        assert set(res["chan"]["test_channel2"].keys()) == {
            "a00",
            "a01",
            "a10",
            "a11",
            "p_open",
        }
        # check if values are correct
        assert np.allclose(res["chan"]["test_channel2"]["a00"], 0.3)
        assert np.allclose(res["chan"]["test_channel2"]["a01"], 0.5)
        assert np.allclose(res["chan"]["test_channel2"]["a10"], 0.4)
        assert np.allclose(res["chan"]["test_channel2"]["a11"], 0.6)
        assert np.allclose(
            res["chan"]["test_channel2"]["p_open"],
            0.9 * 0.3**3 * 0.5**2 + 0.1 * 0.4**2 * 0.6**1,
        )
        # check if shape is correct
        n_loc, n_step = len(locs), len(res["t"])
        assert res["chan"]["test_channel2"]["a00"].shape == (n_loc, n_step)
        assert res["chan"]["test_channel2"]["a01"].shape == (n_loc, n_step)
        assert res["chan"]["test_channel2"]["a10"].shape == (n_loc, n_step)
        assert res["chan"]["test_channel2"]["a11"].shape == (n_loc, n_step)
        assert res["chan"]["test_channel2"]["p_open"].shape == (n_loc, n_step)
        # channel only at soma
        self.load_T_tree_test_channel_soma()
        # create simulation tree
        self.neurontree.init_model(t_calibrate=100.0, factor_lambda=10.0)
        self.neurontree.store_locs(locs, name="rec locs")
        # run test simulation
        res = self.neurontree.run(10.0, record_from_channels=True, use_coreneuron=use_coreneuron)
        # check if results are stored correctly
        assert set(res["chan"]["test_channel2"].keys()) == {
            "a00",
            "a01",
            "a10",
            "a11",
            "p_open",
        }
        # check if values are correct
        assert np.allclose(res["chan"]["test_channel2"]["a00"][0, :], 0.3)
        assert np.allclose(res["chan"]["test_channel2"]["a01"][0, :], 0.5)
        assert np.allclose(res["chan"]["test_channel2"]["a10"][0, :], 0.4)
        assert np.allclose(res["chan"]["test_channel2"]["a11"][0, :], 0.6)
        assert np.allclose(
            res["chan"]["test_channel2"]["p_open"][0, :],
            0.9 * 0.3**3 * 0.5**2 + 0.1 * 0.4**2 * 0.6**1,
        )
        assert np.allclose(res["chan"]["test_channel2"]["a00"][1:, :], 0.0)
        assert np.allclose(res["chan"]["test_channel2"]["a01"][1:, :], 0.0)
        assert np.allclose(res["chan"]["test_channel2"]["a10"][1:, :], 0.0)
        assert np.allclose(res["chan"]["test_channel2"]["a11"][1:, :], 0.0)
        assert np.allclose(res["chan"]["test_channel2"]["p_open"][1:, :], 0.0)
        # check if shape is correct
        n_loc, n_step = len(locs), len(res["t"])
        assert res["chan"]["test_channel2"]["a00"].shape == (n_loc, n_step)
        assert res["chan"]["test_channel2"]["a01"].shape == (n_loc, n_step)
        assert res["chan"]["test_channel2"]["a10"].shape == (n_loc, n_step)
        assert res["chan"]["test_channel2"]["a11"].shape == (n_loc, n_step)
        assert res["chan"]["test_channel2"]["p_open"].shape == (n_loc, n_step)

    def test_recording_timestep(self):
        self.load_T_tree_test_channel()
        # set of locations
        locs = [(1, 0.5), (4, 0.5), (4, 1.0), (5, 0.5), (6, 0.5), (7, 0.5), (8, 0.5)]
        # test simulation 1
        self.neurontree.init_model(t_calibrate=10.0, dt=0.1, factor_lambda=10.0)
        self.neurontree.store_locs(locs, name="rec locs")
        res1 = self.neurontree.run(
            10.0, downsample=10, dt_rec=None, record_from_channels=True
        )
        self.neurontree.delete_model()
        # test simulation 2
        self.load_T_tree_test_channel()
        self.neurontree.init_model(t_calibrate=10.0, dt=0.1, factor_lambda=10.0)
        self.neurontree.store_locs(locs, name="rec locs")
        res2 = self.neurontree.run(
            10.0, downsample=1, dt_rec=1.0, record_from_channels=True, use_coreneuron=use_coreneuron
        )
        self.neurontree.delete_model()

        assert len(res1["t"]) == len(res2["t"])
        assert res1["v_m"].shape == res2["v_m"].shape


class TestReducedNeuron:
    def add_locinds(self):
        for ii, cn in enumerate(self.ctree):
            cn.loc_idx = ii

    def load_two_compartment_model(self, w_locinds=True):
        # simple two compartment model
        pnode = CompartmentNode(0, ca=1.5e-5, g_l=2e-3)
        self.ctree = CompartmentTree(pnode)
        cnode = CompartmentNode(1, ca=2e-6, g_l=3e-4, g_c=4e-3)
        self.ctree.add_node_with_parent(cnode, pnode)

        if w_locinds:
            self.add_locinds()

    def load_T_model(self, w_locinds=True):
        # simple T compartment model
        pnode = CompartmentNode(0, ca=1.5e-5, g_l=2e-3)
        self.ctree = CompartmentTree(pnode)
        cnode = CompartmentNode(1, ca=2e-6, g_l=3e-4, g_c=4e-3)
        self.ctree.add_node_with_parent(cnode, pnode)
        lnode0 = CompartmentNode(2, ca=1.5e-6, g_l=2.5e-4, g_c=3e-3)
        self.ctree.add_node_with_parent(lnode0, cnode)
        lnode1 = CompartmentNode(3, ca=1.5e-6, g_l=2.5e-4, g_c=5e-3)
        self.ctree.add_node_with_parent(lnode1, cnode)

        if w_locinds:
            self.add_locinds()

    def load_three_compartment_model(self, w_locinds=True):
        # simple 3 compartment model
        pnode = CompartmentNode(0, ca=1.9e-6, g_l=1.8e-3)
        self.ctree = CompartmentTree(pnode)
        cnode = CompartmentNode(1, ca=2.4e-6, g_l=0.3e-4, g_c=3.9)
        self.ctree.add_node_with_parent(cnode, pnode)
        lnode0 = CompartmentNode(2, ca=1.9e-6, g_l=0.3e-4, g_c=3.8e-3)
        self.ctree.add_node_with_parent(lnode0, cnode)

        if w_locinds:
            self.add_locinds()

    def load_multi_dend_model(self, w_locinds=True):
        # simple 3 compartment model
        pnode = CompartmentNode(0, ca=1.9e-6, g_l=1.8e-3)
        self.ctree = CompartmentTree(pnode)
        cnode0 = CompartmentNode(1, ca=2.4e-6, g_l=0.3e-4, g_c=3.9)
        self.ctree.add_node_with_parent(cnode0, pnode)
        cnode1 = CompartmentNode(2, ca=1.9e-6, g_l=0.4e-4, g_c=3.8e-3)
        self.ctree.add_node_with_parent(cnode1, pnode)
        cnode2 = CompartmentNode(3, ca=1.3e-6, g_l=0.5e-4, g_c=2.7e-2)
        self.ctree.add_node_with_parent(cnode2, pnode)

        if w_locinds:
            self.add_locinds()

    def test_neuroncompartmentree_instantiation(self):
        self.load_multi_dend_model()
        # correct initialization
        neuron_sim_tree = NeuronCompartmentTree(self.ctree)

        # initialization from incorrect tree
        with pytest.raises(ValueError):
            neuron_sim_tree = NeuronCompartmentTree(GreensTree())

    def test_geometry1(self):
        fake_c_m = 1.0
        fake_r_a = 100.0 * 1e-6
        factor_r_a = 1e-6

        ## test method 1
        # test with two compartments
        self.load_two_compartment_model()
        ctree = self.ctree
        # check if fake geometry is correct
        points, _ = ctree.compute_fake_geometry(
            fake_c_m=fake_c_m, fake_r_a=fake_r_a, factor_r_a=1e-6, delta=1e-14, method=1
        )
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            h.pt3dadd(*points[ii][0], sec=comps[-1])
            h.pt3dadd(*points[ii][1], sec=comps[-1])
            h.pt3dadd(*points[ii][2], sec=comps[-1])
            h.pt3dadd(*points[ii][3], sec=comps[-1])
            comps[-1].Ra = fake_r_a * 1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1
        # check areas
        assert np.abs(comps[0](0.5).area() * 1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area() * 1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        # check whether resistances are correct
        assert np.abs(comps[0](0.5).ri() - 1.0) < 1e-6
        assert (
            np.abs((comps[1](0.5).ri() - 1.0 / ctree[1].g_c) / comps[1](0.5).ri())
            < 1e-6
        )
        assert (
            np.abs(
                (comps[1](0.5).ri() * factor_r_a - comps[1](1.0).ri())
                / comps[1](1.0).ri()
            )
            < 1e-6
        )

        # test with three compartments
        self.load_three_compartment_model()
        ctree = self.ctree
        # check if fake geometry is correct
        points, _ = ctree.compute_fake_geometry(
            fake_c_m=fake_c_m, fake_r_a=fake_r_a, factor_r_a=1e-6, delta=1e-14, method=1
        )
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            h.pt3dadd(*points[ii][0], sec=comps[-1])
            h.pt3dadd(*points[ii][1], sec=comps[-1])
            h.pt3dadd(*points[ii][2], sec=comps[-1])
            h.pt3dadd(*points[ii][3], sec=comps[-1])
            comps[-1].Ra = fake_r_a * 1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1
        # check areas
        assert np.abs(comps[0](0.5).area() * 1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area() * 1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        assert np.abs(comps[2](0.5).area() * 1e-8 * fake_c_m - ctree[2].ca) < 1e-12
        # check whether resistances are correct
        assert np.abs(comps[0](0.5).ri() - 1.0) < 1e-6
        assert (
            np.abs((comps[1](0.5).ri() - 1.0 / ctree[1].g_c) / comps[1](0.5).ri())
            < 1e-6
        )
        assert (
            np.abs((comps[2](0.5).ri() - 1.0 / ctree[2].g_c) / comps[2](0.5).ri())
            < 1e-6
        )
        assert (
            np.abs(
                (comps[1](0.5).ri() * factor_r_a - comps[1](1.0).ri())
                / comps[1](1.0).ri()
            )
            < 1e-6
        )
        assert (
            np.abs(
                (comps[2](0.5).ri() * factor_r_a - comps[2](1.0).ri())
                / comps[2](1.0).ri()
            )
            < 1e-6
        )

        # test the T model
        self.load_T_model()
        ctree = self.ctree
        # check if fake geometry is correct
        points, _ = ctree.compute_fake_geometry(
            fake_c_m=fake_c_m, fake_r_a=fake_r_a, factor_r_a=1e-6, delta=1e-14, method=1
        )
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            h.pt3dadd(*points[ii][0], sec=comps[-1])
            h.pt3dadd(*points[ii][1], sec=comps[-1])
            h.pt3dadd(*points[ii][2], sec=comps[-1])
            h.pt3dadd(*points[ii][3], sec=comps[-1])
            comps[-1].Ra = fake_r_a * 1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1
        # check areas
        assert np.abs(comps[0](0.5).area() * 1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area() * 1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        assert np.abs(comps[2](0.5).area() * 1e-8 * fake_c_m - ctree[2].ca) < 1e-12
        assert np.abs(comps[3](0.5).area() * 1e-8 * fake_c_m - ctree[3].ca) < 1e-12
        # check whether resistances are correct
        assert np.abs(comps[0](0.5).ri() - 1.0) < 1e-6
        assert (
            np.abs((comps[1](0.5).ri() - 1.0 / ctree[1].g_c) / comps[1](0.5).ri())
            < 1e-6
        )
        assert (
            np.abs((comps[2](0.5).ri() - 1.0 / ctree[2].g_c) / comps[2](0.5).ri())
            < 1e-6
        )
        assert (
            np.abs((comps[3](0.5).ri() - 1.0 / ctree[3].g_c) / comps[3](0.5).ri())
            < 1e-6
        )
        assert (
            np.abs(
                (comps[1](0.5).ri() * factor_r_a - comps[1](1.0).ri())
                / comps[1](1.0).ri()
            )
            < 1e-6
        )
        assert (
            np.abs(
                (comps[2](0.5).ri() * factor_r_a - comps[2](1.0).ri())
                / comps[2](1.0).ri()
            )
            < 1e-6
        )
        assert (
            np.abs(
                (comps[3](0.5).ri() * factor_r_a - comps[3](1.0).ri())
                / comps[3](1.0).ri()
            )
            < 1e-6
        )

    def test_impedance_properties_1(self):
        fake_c_m = 1.0
        fake_r_a = 100.0 * 1e-6
        # create the two compartment model without locinds
        self.load_two_compartment_model(w_locinds=False)
        ctree = self.ctree
        # check if error is raised if loc_idxs have not been set
        with pytest.raises(AttributeError):
            ctree.calc_impedance_matrix()

        # create the two compartment model with locinds
        self.load_two_compartment_model()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calc_impedance_matrix()
        # create a neuron model
        sim_tree = NeuronCompartmentTree(
            ctree, fake_c_m=fake_c_m, fake_r_a=fake_r_a, method=1
        )
        z_mat_sim = sim_tree.calc_impedance_matrix([(0, 0.5), (1, 0.5)])

        # create the three compartmental model
        self.load_three_compartment_model()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calc_impedance_matrix()
        # create a neuron model
        sim_tree = NeuronCompartmentTree(
            ctree, fake_c_m=fake_c_m, fake_r_a=fake_r_a, method=1
        )
        z_mat_sim = sim_tree.calc_impedance_matrix([(0, 0.5), (1, 0.5), (2, 0.5)])

        # create the T compartmental model
        self.load_T_model()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calc_impedance_matrix()
        # create a neuron model
        sim_tree = NeuronCompartmentTree(
            ctree, fake_c_m=fake_c_m, fake_r_a=fake_r_a, method=1
        )
        z_mat_sim = sim_tree.calc_impedance_matrix(
            [(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]
        )
        assert np.allclose(z_mat_sim, z_mat_comp)

        # create the multidend model
        self.load_multi_dend_model()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calc_impedance_matrix()
        # create a neuron model
        sim_tree = NeuronCompartmentTree(
            ctree, fake_c_m=fake_c_m, fake_r_a=fake_r_a, method=1
        )
        z_mat_sim = sim_tree.calc_impedance_matrix(
            [(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]
        )
        assert np.allclose(z_mat_sim, z_mat_comp)

    def test_geometry2(self):
        fake_c_m = 1.0
        fake_r_a = 100.0 * 1e-6
        factor_r_a = 1e-6

        ## test method 2
        # test with two compartments
        self.load_two_compartment_model()
        ctree = self.ctree
        # check if fake geometry is correct
        lengths, radii = ctree.compute_fake_geometry(
            fake_c_m=fake_c_m, fake_r_a=fake_r_a, factor_r_a=1e-6, delta=1e-14, method=2
        )
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            comps[-1].diam = 2.0 * radii[ii] * 1e4
            comps[-1].L = lengths[ii] * 1e4
            comps[-1].Ra = fake_r_a * 1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1

        # check areas
        assert np.abs(comps[0](0.5).area() * 1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area() * 1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        # check whether resistances are correct
        assert (
            np.abs((comps[1](0.5).ri() - 1.0 / ctree[1].g_c) / comps[1](0.5).ri())
            < 1e-12
        )
        assert (
            np.abs((comps[1](0.5).ri() - comps[1](1.0).ri()) / comps[1](1.0).ri())
            < 1e-12
        )

        # test with three compartments
        self.load_three_compartment_model()
        ctree = self.ctree
        # check if fake geometry is correct
        lengths, radii = ctree.compute_fake_geometry(
            fake_c_m=fake_c_m, fake_r_a=fake_r_a, factor_r_a=1e-6, delta=1e-14, method=2
        )
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            comps[-1].diam = 2.0 * radii[ii] * 1e4
            comps[-1].L = lengths[ii] * 1e4
            comps[-1].Ra = fake_r_a * 1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1

        # check areas
        assert np.abs(comps[0](0.5).area() * 1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area() * 1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        assert np.abs(comps[2](0.5).area() * 1e-8 * fake_c_m - ctree[2].ca) < 1e-12
        # check whether resistances are correct
        assert (
            np.abs((comps[1](0.5).ri() - 1.0 / ctree[1].g_c) / comps[1](0.5).ri())
            < 1e-12
        )
        assert (
            np.abs((comps[1](0.5).ri() - comps[1](1.0).ri()) / comps[1](1.0).ri())
            < 1e-12
        )
        assert (
            np.abs((comps[2](0.5).ri() - 1.0 / ctree[2].g_c) / comps[2](0.5).ri())
            < 1e-12
        )
        assert (
            np.abs((comps[2](0.5).ri() - comps[2](1.0).ri()) / comps[2](1.0).ri())
            < 1e-12
        )

        # test the T model
        self.load_T_model()
        ctree = self.ctree
        # check if fake geometry is correct
        lengths, radii = ctree.compute_fake_geometry(
            fake_c_m=fake_c_m, fake_r_a=fake_r_a, factor_r_a=1e-6, delta=1e-14, method=2
        )
        # create a neuron comparemtns
        comps = []
        for ii, node in enumerate(ctree):
            comps.append(h.Section())
            comps[-1].diam = 2.0 * radii[ii] * 1e4
            comps[-1].L = lengths[ii] * 1e4
            comps[-1].Ra = fake_r_a * 1e6
            comps[-1].cm = fake_c_m
            comps[-1].nseg = 1

        # check areas
        assert np.abs(comps[0](0.5).area() * 1e-8 * fake_c_m - ctree[0].ca) < 1e-12
        assert np.abs(comps[1](0.5).area() * 1e-8 * fake_c_m - ctree[1].ca) < 1e-12
        assert np.abs(comps[2](0.5).area() * 1e-8 * fake_c_m - ctree[2].ca) < 1e-12
        assert np.abs(comps[3](0.5).area() * 1e-8 * fake_c_m - ctree[3].ca) < 1e-12
        # check whether resistances are correct
        assert (
            np.abs((comps[1](0.5).ri() - 1.0 / ctree[1].g_c) / comps[1](0.5).ri())
            < 1e-12
        )
        assert (
            np.abs((comps[1](0.5).ri() - comps[1](1.0).ri()) / comps[1](1.0).ri())
            < 1e-12
        )
        assert (
            np.abs((comps[2](0.5).ri() - 1.0 / ctree[2].g_c) / comps[2](0.5).ri())
            < 1e-12
        )
        assert (
            np.abs((comps[2](0.5).ri() - comps[2](1.0).ri()) / comps[2](1.0).ri())
            < 1e-12
        )
        assert (
            np.abs((comps[3](0.5).ri() - 1.0 / ctree[3].g_c) / comps[3](0.5).ri())
            < 1e-12
        )
        assert (
            np.abs((comps[3](0.5).ri() - comps[3](1.0).ri()) / comps[3](1.0).ri())
            < 1e-12
        )

    def test_impedance_properties_2(self):
        fake_c_m = 1.0
        fake_r_a = 100.0 * 1e-6
        # create the two compartment model
        self.load_two_compartment_model()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calc_impedance_matrix()
        # create a neuron model
        sim_tree = NeuronCompartmentTree(
            ctree,
            fake_c_m=fake_c_m,
            fake_r_a=fake_r_a,
            method=2,
        )
        z_mat_sim = sim_tree.calc_impedance_matrix([(0, 0.5), (1, 0.5)])
        assert np.allclose(z_mat_sim, z_mat_comp, atol=1e-2)

        # create the three compartmental model
        self.load_three_compartment_model()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calc_impedance_matrix()
        # create a neuron model
        sim_tree = NeuronCompartmentTree(
            ctree,
            fake_c_m=fake_c_m,
            fake_r_a=fake_r_a,
            method=2,
        )
        z_mat_sim = sim_tree.calc_impedance_matrix([(0, 0.5), (1, 0.5), (2, 0.5)])
        assert np.allclose(z_mat_sim, z_mat_comp)

        # create the T compartmental model
        self.load_T_model()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calc_impedance_matrix()
        # create a neuron model
        sim_tree = NeuronCompartmentTree(
            ctree,
            fake_c_m=fake_c_m,
            fake_r_a=fake_r_a,
            method=2,
        )
        z_mat_sim = sim_tree.calc_impedance_matrix(
            [(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]
        )
        assert np.allclose(z_mat_sim, z_mat_comp)

        # create the multidend model
        self.load_multi_dend_model()
        ctree = self.ctree
        # compute the impedance matrix exactly
        z_mat_comp = ctree.calc_impedance_matrix()
        # create a neuron model
        sim_tree = NeuronCompartmentTree(
            ctree,
            fake_c_m=fake_c_m,
            fake_r_a=fake_r_a,
            method=2,
        )
        z_mat_sim = sim_tree.calc_impedance_matrix(
            [(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]
        )
        assert np.allclose(z_mat_sim, z_mat_comp)



class TestStimuli:
    def _create_ball(self):
        """
        Load the T-tree morphology in memory with passive conductance

          6--5--4--7--8
                |
                |
                1
        """
        self.v_eq = -75.0
        # load the morphology
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc")
        self.tree = NeuronSimTree(fname, types=[1, 3, 4])
        self.tree.fit_leak_current(self.v_eq, 10.0)
        self.tree.set_comp_tree()

    def test_i_clamp(self, pplot=False):
        # parameter setup
        dt = 0.1
        tmax = 1000.0
        t_calibrate = 50.
        amp = 0.01

        self._create_ball()
        self.tree.init_model(t_calibrate=t_calibrate, dt=dt)
        # add step current clamp
        self.tree.add_i_clamp(
            loc=(1, 0.5),
            delay=3 * tmax / 10,
            dur= 4 * tmax / 10.,
            amp=0.01,
        )

        res = self.tree.run(tmax, record_from_iclamps=True, use_coreneuron=True)
        print(res['v_m'])

        if pplot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(res['t'], res['i_clamp'][0,:], label='I Clamp')
            plt.plot(res['t'], res['v_m'][0,:], label='V Membrane')
            plt.xlabel('Time [ms]')
            plt.legend()
            plt.show()

    def _voltage_statistics(self, delta_t, mu_ou, sigma_ou, tau_ou):
        """
        Stationary autocovariance C_v(Δ).
        
        Parameters
        ----------
        delta_t : array_like
            Time lag(s) Δ
        sigma_ou : float
            OU standard deviation parameter [nA]
        tau_ou : float
            OU timescale [ms]
        mu_ou : float
            OU mean [nA]
        """
        mu_ou *= 1e-3 # nA to uA
        sigma_ou *= 1e-3 # nA to uA

        soma = self.tree[1]
        tau = soma.c_m / soma.currents['L'][0] * 1e3  # s to ms
        e_l = soma.currents['L'][1]
        ca = soma.c_m * 4 * np.pi * soma.R**2 * 1e-8  # um^2 to cm^2

        # voltage mean
        mean = e_l + tau * mu_ou / ca # ms * V / s = mV

        delta_t = np.abs(np.asarray(delta_t))

        prefactor = sigma_ou**2 / ca**2 # (uA / uF)^2 = (V / s)^2
        scale = (tau**2 * tau_ou) / (tau_ou**2 - tau**2) 

        # autocovarianvce
        autocov = prefactor * scale * (
            tau_ou * np.exp(-delta_t / tau_ou)
            - tau * np.exp(-delta_t / tau)
        ) # (V / s)^2 * ms^2 = mV^2

        return mean, autocov
    
    def _autocovariance(self, x, max_lag=None):
        """
        Empirical autocovariance of a 1D time series.

        Parameters
        ----------
        x : array_like
            Time series data (1D).
        max_lag : int or None
            Maximum lag (in samples). If None, uses N-1.

        Returns
        -------
        lags : ndarray
            Array of integer lags.
        cov : ndarray
            Autocovariance at each lag.
        """
        x = np.asarray(x)
        x = x - np.mean(x)

        N = x.size
        if max_lag is None:
            max_lag = N - 1

        cov = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            cov[k] = np.dot(x[:N-k], x[k:]) / (N - k)

        lags = np.arange(max_lag + 1)
        return lags, cov

    def _autocovariance_fft(self, x, max_lag=None):
        """
        Empirical autocovariance using FFT.

        Parameters
        ----------
        x : array_like
            Time series data (1D).
        max_lag : int or None
            Maximum lag (in samples). If None, uses N-1.

        Returns
        -------
        lags : ndarray
            Array of integer lags.
        cov : ndarray
            Autocovariance at each lag.
        """
        x = np.asarray(x)
        x = x - np.mean(x)

        N = x.size
        if max_lag is None:
            max_lag = N - 1

        # Zero-pad to avoid circular convolution
        nfft = 2 * N
        fx = np.fft.fft(x, n=nfft)
        acf = np.fft.ifft(fx * np.conjugate(fx)).real

        acf = acf[:max_lag + 1]
        normalization = N - np.arange(max_lag + 1)

        return np.arange(max_lag + 1), acf / normalization

    def test_ou_processes(self, pplot=False):
        # parameter setup
        dt = 0.1
        tmax = 10000.0
        t_calibrate = 50.
        tau_ou = 5.0
        sigma_ou = 0.005
        mu_ou = 0.05

        self._create_ball()
        self.tree.init_model(t_calibrate=t_calibrate, dt=dt)
        # add OU process
        self.tree.add_ou_clamp(
            loc=(1, 0.5),
            mean=mu_ou,
            stdev=sigma_ou,
            tau=tau_ou,
            delay=-t_calibrate / 2.,
            dur=tmax + t_calibrate / 2.,
            seed=46,
        )

        res = self.tree.run(self.tmax, record_from_iclamps=True)

        mu_ou_empirical = np.mean(res['i_clamp'][0,:])
        cov_ou_empirical = np.cov(res['i_clamp'][0,:])

        print(f"OU mean: exact={mu_ou:.10f} nA, empirical={mu_ou_empirical:.10f} nA")
        print(f"OU variance: exact={sigma_ou**2:.10f} nA^2, empirical={cov_ou_empirical:.10f} nA^2")
        
        # check current statistics
        assert np.abs(mu_ou_empirical - mu_ou) / np.abs(mu_ou) < 0.05
        assert np.abs(cov_ou_empirical - sigma_ou**2) / sigma_ou**2 < 0.05

        mu_exact, cov_exact = self._voltage_statistics(
            delta_t=res['t'],
            mu_ou=mu_ou,
            sigma_ou=sigma_ou,
            tau_ou=tau_ou,
        )
        mu_empirical = np.mean(res['v_m'][0, :])
        cov_np = np.cov(res['v_m'][0, :])
        _, cov_empirical = self._autocovariance_fft(res['v_m'][0, :])

        print(f"Voltage mean: exact={mu_exact:.10f} mV, empirical={mu_empirical:.10f} mV")
        print(f"Voltage variance: exact={cov_exact[0]:.10f} mV^2, empirical={cov_np:.10f} mV^2")

        # check voltage statistics
        assert np.abs(mu_empirical - mu_exact) / np.abs(mu_exact) < 0.05
        assert np.abs(cov_np - cov_exact[0]) / cov_exact[0] < 0.1
        assert np.mean(np.abs(cov_empirical[:500] - cov_exact[:500]) / cov_exact[0]) < 0.05

        if pplot:
            pl.figure(figsize=(12, 5))
            ax = pl.subplot(1, 3, 1)
            ax.plot(res['t'], res['v_m'][0, :])
            ax.axhline(mu_exact)
            ax.axhline(mu_empirical, color='orange')
            ax = pl.subplot(1, 3, 2)
            ax.plot(res['t'][:500], cov_exact[:500], label='Exact')
            ax.plot(res['t'][:500], cov_empirical[:500], label='Empirical')
            ax.axhline(cov_np, color='gray', linestyle='--', label='Empirical variance')
            ax.legend(loc=0)
            pl.show()


def debug_print(pstr):

    print(pstr)
    try:
        print(os.listdir(os.path.join(neat.__path__[0], "simulations/neuron/tmp/multichannel_test/arm64")))
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    tn = TestNeuron()
    # tn.test_passive(pplot=True)
    # tn.test_active(pplot=True)
    tn.test_channel_recording(use_coreneuron=True)
    # tn.test_recording_timestep()

    # trn = TestReducedNeuron()
    # trn.test_geometry1()
    # trn.test_impedance_properties_1()
    # trn.test_geometry2()
    # trn.test_impedance_properties_2()

    # ts = TestStimuli()
    # ts.test_i_clamp(pplot=True)
    # ts.test_ou_processes()