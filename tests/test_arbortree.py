# -*- coding: utf-8 -*-
#
# test_arbortree.py
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
import itertools

arbor = pytest.importorskip("arbor")

from neat import PhysTree, GreensTree
from neat import CompartmentNode, CompartmentTree
from neat import ArborSimTree, ArborCompartmentTree
from neat import NeuronSimTree, NeuronCompartmentTree
import neat.tools.kernelextraction as ke

import channelcollection_for_tests as channelcollection
import channel_installer

channel_installer.load_or_install_neuron_test_channels()
channel_installer.load_or_install_arbor_test_channels()


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


class TestArbor:
    def _load_T_tree(self, channel=None, g=50.0, e_rev=-43.0, node_arg=None):
        """
        Load the T-tree morphology in memory, with an optional ion channel

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
        if channel is not None:
            node_arg = None if node_arg is None else [self.greenstree[node_arg]]
            self.greenstree.add_channel_current(channel, g, e_rev, node_arg=node_arg)
        self.greenstree.fit_leak_current(self.v_eq, 10.0)
        self.greenstree.set_comp_tree()
        self.greenstree.set_impedance(self.ft.s)
        # copy greenstree parameters into Arbor simulation tree
        self.arbortree = ArborSimTree(self.greenstree)
        self.arbortree.set_default_tree("computational")

    def load_T_tree_passive(self):
        self._load_T_tree()

    def load_T_tree_active(self):
        self._load_T_tree(channel=channelcollection.h(), g=50.0, e_rev=-43.0)

    def load_T_tree_test_channel(self):
        self._load_T_tree(
            channel=channelcollection.test_channel2(), g=50.0, e_rev=-23.0
        )

    def load_T_tree_test_channel_soma(self):
        self._load_T_tree(
            channel=channelcollection.test_channel2(), g=50.0, e_rev=23.0, node_arg=1
        )

    def _compare_kernels(self, locs, z_kwargs, atol_z, atol_zk, pplot=False):
        # compute impedance matrix with Green's function
        zf_mat_gf = self.greenstree.calc_impedance_matrix(locs)
        z_mat_gf = zf_mat_gf[self.ft.ind_0s].real
        # convert impedance matrix to time domain
        zk_mat_gf = np.zeros((len(self.ft.t), len(locs), len(locs)))
        for ii, jj in itertools.product(list(range(len(locs))), list(range(len(locs)))):
            zk_mat_gf[:, ii, jj] = self.ft.ft_inv(zf_mat_gf[:, ii, jj])[1].real * 1e-3
        # test the steady state impedance matrix
        z_mat_arbor = self.arbortree.calc_impedance_matrix(
            locs,
            dt=self.dt,
            t_calibrate=100.0,
            v_init=self.v_eq,
            factor_lambda=25.0,
            **z_kwargs,
        )
        assert np.allclose(z_mat_gf, z_mat_arbor, atol=atol_z)
        # test the temporal matrix
        tk, zk_mat_arbor = self.arbortree.calc_impulse_response_matrix(
            locs,
            dt=self.dt,
            t_calibrate=100.0,
            v_init=self.v_eq,
            factor_lambda=25.0,
            dstep=-2,
        )
        nt = min(zk_mat_gf.shape[0], zk_mat_arbor.shape[0])
        assert np.allclose(
            zk_mat_gf[int(2.0 / self.dt) : nt, :, :],
            zk_mat_arbor[int(2.0 / self.dt) : nt, :, :],
            atol=atol_zk,
        )
        if pplot:
            # plot kernels
            pl.figure()
            cc = 0
            for ii in range(len(locs)):
                jj = 0
                while jj <= ii:
                    pl.plot(tk, zk_mat_arbor[:nt, ii, jj], c=colours[cc % len(colours)])
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

    def test_passive(self, pplot=False):
        self.load_T_tree_passive()
        locs = [(1, 0.5), (4, 0.5), (4, 1.0), (5, 0.5), (6, 0.5), (7, 0.5), (8, 0.5)]
        self._compare_kernels(locs, {}, atol_z=1.0, atol_zk=0.2, pplot=pplot)

    def test_active(self, pplot=False):
        self.load_T_tree_active()
        locs = [(1, 0.5), (4, 0.5), (6, 0.5), (7, 0.5), (8, 0.5)]
        self._compare_kernels(
            locs, {"t_dur": 500.0}, atol_z=5.0, atol_zk=0.5, pplot=pplot
        )

    def _check_test_channel2_recording(self, res, n_loc, idx):
        # check if results are stored correctly
        assert set(res["chan"]["test_channel2"].keys()) == {
            "a00",
            "a01",
            "a10",
            "a11",
            "p_open",
        }
        # check if values are correct
        assert np.allclose(res["chan"]["test_channel2"]["a00"][idx], 0.3)
        assert np.allclose(res["chan"]["test_channel2"]["a01"][idx], 0.5)
        assert np.allclose(res["chan"]["test_channel2"]["a10"][idx], 0.4)
        assert np.allclose(res["chan"]["test_channel2"]["a11"][idx], 0.6)
        assert np.allclose(
            res["chan"]["test_channel2"]["p_open"][idx],
            0.9 * 0.3**3 * 0.5**2 + 0.1 * 0.4**2 * 0.6**1,
        )
        # check if shape is correct
        n_step = len(res["t"])
        for key in ["a00", "a01", "a10", "a11", "p_open"]:
            assert res["chan"]["test_channel2"][key].shape == (n_loc, n_step)

    def test_channel_recording(self):
        self.load_T_tree_test_channel()
        # set of locations
        locs = [(1, 0.5), (4, 0.5), (4, 1.0), (5, 0.5), (6, 0.5), (7, 0.5), (8, 0.5)]
        # create simulation tree
        self.arbortree.init_model(t_calibrate=10.0, factor_lambda=10.0)
        self.arbortree.store_locs(locs, name="rec locs")
        # run test simulation
        res = self.arbortree.run(1.0, record_from_channels=True)
        self.arbortree.delete_model()
        self._check_test_channel2_recording(res, len(locs), slice(None))

        # channel only at soma
        self.load_T_tree_test_channel_soma()
        # create simulation tree
        self.arbortree.init_model(t_calibrate=100.0, factor_lambda=10.0)
        self.arbortree.store_locs(locs, name="rec locs")
        # run test simulation
        res = self.arbortree.run(10.0, record_from_channels=True)
        self.arbortree.delete_model()
        self._check_test_channel2_recording(res, len(locs), 0)
        for key in ["a00", "a01", "a10", "a11", "p_open"]:
            assert np.allclose(res["chan"]["test_channel2"][key][1:, :], 0.0)

    def test_recording_timestep(self):
        self.load_T_tree_test_channel()
        # set of locations
        locs = [(1, 0.5), (4, 0.5), (4, 1.0), (5, 0.5), (6, 0.5), (7, 0.5), (8, 0.5)]
        # test simulation 1
        self.arbortree.init_model(t_calibrate=10.0, dt=0.1, factor_lambda=10.0)
        self.arbortree.store_locs(locs, name="rec locs")
        res1 = self.arbortree.run(
            10.0, downsample=10, dt_rec=None, record_from_channels=True
        )
        self.arbortree.delete_model()
        # test simulation 2
        self.load_T_tree_test_channel()
        self.arbortree.init_model(t_calibrate=10.0, dt=0.1, factor_lambda=10.0)
        self.arbortree.store_locs(locs, name="rec locs")
        res2 = self.arbortree.run(
            10.0, downsample=1, dt_rec=1.0, record_from_channels=True
        )
        self.arbortree.delete_model()

        assert len(res1["t"]) == len(res2["t"])
        assert res1["v_m"].shape == res2["v_m"].shape
        assert np.allclose(res1["t"], res2["t"])
        assert np.allclose(res1["v_m"], res2["v_m"])

    def test_calc_e_eq(self):
        self.load_T_tree_active()
        self.arbortree.set_simulation_parameters(t_calibrate=0.0, dt=0.1)
        v_eq = self.arbortree.calc_e_eq(t_dur=500.0)

        nodes = list(self.arbortree)
        assert v_eq.shape == (len(nodes),)
        # the leak was fitted to yield an equilibrium potential of -75 mV
        assert np.allclose(v_eq, self.v_eq, atol=1e-3)
        assert np.allclose([node.v_ep for node in nodes], v_eq)

    def test_neuron_equivalence(self):
        """
        Test whether Arbor and NEURON simulations of a spiking model with ion
        channels and calcium dynamics yield the same results
        """
        tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball_and_axon.swc"),
            types=[1, 2, 3, 4],
        )
        tree.set_physiology(1.0, 100.0 / 1e6)
        # parameters of `TestConcMechs.load_axon_tree()` in `test_concmechs.py`
        k_chan = channelcollection.SKv3_1()
        tree.add_channel_current(k_chan, 0.653374 * 1e6, -85.0, node_arg=[tree[1]])
        tree.add_channel_current(k_chan, 0.196957 * 1e6, -85.0, node_arg="axonal")
        na_chan = channelcollection.NaTa_t()
        tree.add_channel_current(na_chan, 3.418459 * 1e6, 50.0, node_arg="axonal")
        ca_chan = channelcollection.Ca_HVA()
        tree.add_channel_current(ca_chan, 0.000792 * 1e6, 132.458, node_arg=[tree[1]])
        tree.add_channel_current(ca_chan, 0.000138 * 1e6, 132.458, node_arg="axonal")
        sk_chan = channelcollection.SK_E2()
        tree.add_channel_current(sk_chan, 0.653374 * 1e6, -85.0, node_arg=[tree[1]])
        tree.add_channel_current(sk_chan, 0.196957 * 1e6, -85.0, node_arg="axonal")
        tree.set_leak_current(0.000091 * 1e6, -62.442793, node_arg=[tree[1]])
        tree.set_leak_current(0.000094 * 1e6, -79.315740, node_arg="axonal")
        for gamma, tau, node_arg in [
            (0.000893, 605.033222, [tree[1]]),
            (0.003923, 20.715642, "axonal"),
        ]:
            tree.add_conc_mech(
                "ca",
                params={
                    "tau": tau,
                    "gamma": gamma * 1e4 / (2.0 * 0.1 * 96485.3321) * 1e-6,
                },
                node_arg=node_arg,
            )
        tree.set_comp_tree()

        locs = [(1, 0.5), (4, 0.5), (4, 1.0), (5, 0.5), (5, 1.0)]
        res = {}
        for sim_tree in [NeuronSimTree(tree), ArborSimTree(tree)]:
            # identical number of compartments per node
            sim_tree.init_model(t_calibrate=100.0, factor_lambda=10)
            sim_tree.store_locs(locs, name="rec locs")
            sim_tree.add_i_clamp(locs[0], 0.8, 10.0, 100.0)
            res[type(sim_tree)] = sim_tree.run(
                150.0,
                record_from_channels=True,
                record_concentrations=["ca"],
                record_currents=["ca", "k"],
                spike_rec_loc=locs[0],
            )
            sim_tree.delete_model()
        res_neuron, res_arbor = res[NeuronSimTree], res[ArborSimTree]

        assert res_neuron["v_m"].shape == res_arbor["v_m"].shape
        assert np.allclose(res_neuron["t"], res_arbor["t"])
        # subthreshold responses
        i0 = int(10.0 / 0.025)
        assert np.allclose(
            res_neuron["v_m"][:, :i0], res_arbor["v_m"][:, :i0], atol=1e-2
        )
        for channel_name, chan_res in res_neuron["chan"].items():
            for var, trace in chan_res.items():
                assert np.allclose(
                    trace[:, :i0],
                    res_arbor["chan"][channel_name][var][:, :i0],
                    atol=1e-3,
                )
        assert np.allclose(res_neuron["ik"][:, :i0], res_arbor["ik"][:, :i0], rtol=1e-3)
        # spiking responses
        assert len(res_neuron["spikes"]) > 2
        assert len(res_neuron["spikes"]) == len(res_arbor["spikes"])
        assert np.allclose(res_neuron["spikes"], res_arbor["spikes"], atol=0.2)
        assert np.allclose(res_neuron["ca"], res_arbor["ca"], rtol=5e-2)


class TestReducedArbor:
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

    def load_models(self):
        for load_model in [
            self.load_two_compartment_model,
            self.load_three_compartment_model,
            self.load_T_model,
            self.load_multi_dend_model,
        ]:
            load_model()
            yield self.ctree

    def test_arborcompartmenttree_instantiation(self):
        self.load_multi_dend_model()
        # correct initialization
        arbor_sim_tree = ArborCompartmentTree(self.ctree)

        # initialization from incorrect tree
        with pytest.raises(ValueError):
            arbor_sim_tree = ArborCompartmentTree(GreensTree())

    def test_discretization(self):
        for ctree in self.load_models():
            sim_tree = ArborCompartmentTree(ctree)
            sim_tree.init_model()
            cell = arbor.cable_cell(
                sim_tree.segment_tree,
                sim_tree.decor,
                arbor.label_dict(),
                sim_tree.cv_policy,
            )
            cv_data = arbor.cv_data(cell)
            # one CV per compartment, connected as the compartments
            assert cv_data.num_cv == len(ctree)
            assert sorted(len(cv_data.children(ii)) for ii in range(len(ctree))) == (
                sorted(len(node.child_nodes) for node in ctree)
            )

    def test_impedance_properties(self):
        for ctree in self.load_models():
            # compute the impedance matrix exactly
            z_mat_comp = ctree.calc_impedance_matrix()
            # simulate the Arbor model
            sim_tree = ArborCompartmentTree(ctree)
            z_mat_sim = sim_tree.calc_impedance_matrix(
                [(node.index, 0.5) for node in ctree]
            )
            assert np.allclose(z_mat_sim, z_mat_comp)

        # check if error is raised if loc_idxs have not been set
        self.load_two_compartment_model(w_locinds=False)
        with pytest.raises(AttributeError):
            self.ctree.calc_impedance_matrix()

    def test_neuron_equivalence(self, pplot=False):
        """
        Test whether Arbor and NEURON simulations of a reduced model with ion
        channels and calcium dynamics yield the same results
        """
        self.load_T_model()
        channels = [
            channelcollection.SKv3_1(),
            channelcollection.Ca_HVA(),
            channelcollection.SK_E2(),
        ]
        for channel, e_rev in zip(channels, [-85.0, 132.458, -85.0]):
            self.ctree.add_channel_current(channel, e_rev)
        for node in self.ctree:
            node.currents["SKv3_1"][0] = 1e-2
            node.currents["Ca_HVA"][0] = 1e-4
            node.currents["SK_E2"][0] = 5e-3
            node.add_conc_mech("ca", tau=100.0)
            node.concmechs["ca"].gamma = 1e-3

        loc_idxs = [0, 1, 2, 3]
        res = {}
        for sim_tree in [
            NeuronCompartmentTree(self.ctree),
            ArborCompartmentTree(self.ctree),
        ]:
            sim_tree.init_model(t_calibrate=100.0)
            sim_tree.set_rec_locs(loc_idxs)
            sim_tree.add_i_clamp(0, 0.1, 10.0, 50.0)
            sim_tree.add_i_clamp(3, 0.05, 30.0, 20.0)
            res[type(sim_tree)] = sim_tree.run(
                100.0, record_concentrations=["ca"], record_from_channels=True
            )
            sim_tree.delete_model()
        res_neuron = res[NeuronCompartmentTree]
        res_arbor = res[ArborCompartmentTree]

        if pplot:
            pl.figure()
            for ii in loc_idxs:
                pl.plot(res_neuron["t"], res_neuron["v_m"][ii], c=colours[ii])
                pl.plot(res_arbor["t"], res_arbor["v_m"][ii], ls="--", c=colours[ii])
            pl.show()

        assert np.max(res_neuron["v_m"]) > res_neuron["v_m"][0, 0] + 1.0
        # `NeuronCompartmentTree` implements the coupling conductances only
        # approximately, cf. `TestReducedNeuron.test_impedance_properties_2()`
        assert np.allclose(res_neuron["v_m"], res_arbor["v_m"], atol=5e-2)
        assert np.allclose(res_neuron["ca"], res_arbor["ca"], rtol=1e-3)
        assert np.allclose(
            res_neuron["chan"]["Ca_HVA"]["p_open"],
            res_arbor["chan"]["Ca_HVA"]["p_open"],
            atol=1e-3,
        )


class TestStimuli:
    def _create_ball(self):
        """
        Load the ball morphology in memory with passive conductance
        """
        self.v_eq = -75.0
        # load the morphology
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc")
        self.tree = ArborSimTree(fname, types=[1, 3, 4])
        self.tree.fit_leak_current(self.v_eq, 10.0)
        self.tree.set_comp_tree()

    def _input_resistance(self):
        soma = self.tree[1]
        area = 4.0 * np.pi * (soma.R * 1e-4) ** 2  # cm^2
        return 1.0 / (soma.currents["L"][0] * area)  # MOhm

    def test_i_clamp(self, pplot=False):
        # parameter setup
        dt = 0.1
        tmax = 1000.0
        t_calibrate = 50.0
        amp = 0.01
        delay = 3 * tmax / 10.0
        dur = 4 * tmax / 10.0

        self._create_ball()
        self.tree.init_model(t_calibrate=t_calibrate, dt=dt)
        # add step current clamp
        self.tree.add_i_clamp(loc=(1, 0.5), delay=delay, dur=dur, amp=amp)

        res = self.tree.run(tmax, record_from_iclamps=True)
        self.tree.delete_model()

        # injected current is positive
        i_on = np.logical_and(res["t"] >= delay + dt, res["t"] < delay + dur - dt)
        i_off = np.logical_or(res["t"] < delay - dt, res["t"] > delay + dur + dt)
        assert np.allclose(res["i_clamp"][0, i_on], amp)
        assert np.allclose(res["i_clamp"][0, i_off], 0.0)
        # steady state voltage
        i_ss = np.argmin(np.abs(res["t"] - (delay + dur - 1.0)))
        assert np.abs(res["v_m"][0, 0] - self.v_eq) < 1e-6
        assert (
            np.abs(res["v_m"][0, i_ss] - (self.v_eq + amp * self._input_resistance()))
            < 1e-3
        )

        if pplot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(res["t"], res["i_clamp"][0, :], label="I Clamp")
            plt.plot(res["t"], res["v_m"][0, :], label="V Membrane")
            plt.xlabel("Time [ms]")
            plt.legend()
            plt.show()

    def test_sin_clamp(self):
        # parameter setup
        dt = 0.025
        tmax = 300.0
        t_calibrate = 50.0
        amp, bias, freq, phase = 0.02, 0.005, 20.0, 0.7
        delay, dur = 50.0, 200.0

        self._create_ball()
        self.tree.init_model(t_calibrate=t_calibrate, dt=dt)
        self.tree.add_sin_clamp(
            loc=(1, 0.5),
            amp=amp,
            delay=delay,
            dur=dur,
            bias=bias,
            freq=freq,
            phase=phase,
        )
        res = self.tree.run(tmax, record_from_iclamps=True)
        self.tree.delete_model()

        i_on = np.logical_and(res["t"] >= delay + dt, res["t"] < delay + dur - dt)
        i_off = np.logical_or(res["t"] < delay - dt, res["t"] > delay + dur + dt)
        i_exact = (
            amp * np.sin(2.0 * np.pi * freq * (res["t"] - delay) * 1e-3 + phase) + bias
        )
        assert np.allclose(res["i_clamp"][0, i_on], i_exact[i_on])
        assert np.allclose(res["i_clamp"][0, i_off], 0.0)

    def test_v_clamp(self):
        dt = 0.025
        e_c = -50.0
        dur = 100.0

        self._create_ball()
        self.tree.init_model(t_calibrate=0.0, dt=dt)
        self.tree.add_v_clamp(loc=(1, 0.5), e_c=e_c, dur=dur)
        res = self.tree.run(200.0, record_from_vclamps=True)
        self.tree.delete_model()

        i_on = np.logical_and(res["t"] > 10.0, res["t"] < dur - dt)
        i_off = res["t"] > dur + 50.0
        # clamped voltage, up to the voltage drop over the series resistance
        assert np.allclose(res["v_m"][0, i_on], e_c, atol=1e-2)
        # depolarizing clamp current
        i_ss = (e_c - self.v_eq) / self._input_resistance()
        assert np.allclose(res["i_vclamp"][0, i_on], i_ss, rtol=1e-3)
        # after the clamp, the voltage decays back to rest
        assert np.allclose(res["i_vclamp"][0, i_off], 0.0)
        assert np.allclose(res["v_m"][0, -1], self.v_eq, atol=1e-2)

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
        mu_ou *= 1e-3  # nA to uA
        sigma_ou *= 1e-3  # nA to uA

        soma = self.tree[1]
        tau = soma.c_m / soma.currents["L"][0] * 1e3  # s to ms
        e_l = soma.currents["L"][1]
        ca = soma.c_m * 4 * np.pi * soma.R**2 * 1e-8  # um^2 to cm^2

        # voltage mean
        mean = e_l + tau * mu_ou / ca  # ms * V / s = mV

        delta_t = np.abs(np.asarray(delta_t))

        prefactor = sigma_ou**2 / ca**2  # (uA / uF)^2 = (V / s)^2
        scale = (tau**2 * tau_ou) / (tau_ou**2 - tau**2)

        # autocovarianvce
        autocov = (
            prefactor
            * scale
            * (tau_ou * np.exp(-delta_t / tau_ou) - tau * np.exp(-delta_t / tau))
        )  # (V / s)^2 * ms^2 = mV^2

        return mean, autocov

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

        acf = acf[: max_lag + 1]
        normalization = N - np.arange(max_lag + 1)

        return np.arange(max_lag + 1), acf / normalization

    def test_ou_processes(self, pplot=False):
        # parameter setup
        dt = 0.1
        tmax = 10000.0
        t_calibrate = 50.0
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
            delay=-t_calibrate / 2.0,
            dur=tmax + t_calibrate / 2.0,
            seed=46,
        )

        res = self.tree.run(tmax, record_from_iclamps=True)
        self.tree.delete_model()

        mu_ou_empirical = np.mean(res["i_clamp"][0, :])
        cov_ou_empirical = np.cov(res["i_clamp"][0, :])

        print(f"OU mean: exact={mu_ou:.10f} nA, empirical={mu_ou_empirical:.10f} nA")
        print(
            f"OU variance: exact={sigma_ou**2:.10f} nA^2, empirical={cov_ou_empirical:.10f} nA^2"
        )

        # check current statistics
        assert np.abs(mu_ou_empirical - mu_ou) / np.abs(mu_ou) < 0.05
        assert np.abs(cov_ou_empirical - sigma_ou**2) / sigma_ou**2 < 0.05

        mu_exact, cov_exact = self._voltage_statistics(
            delta_t=res["t"],
            mu_ou=mu_ou,
            sigma_ou=sigma_ou,
            tau_ou=tau_ou,
        )
        mu_empirical = np.mean(res["v_m"][0, :])
        cov_np = np.cov(res["v_m"][0, :])
        _, cov_empirical = self._autocovariance_fft(res["v_m"][0, :])

        print(
            f"Voltage mean: exact={mu_exact:.10f} mV, empirical={mu_empirical:.10f} mV"
        )
        print(
            f"Voltage variance: exact={cov_exact[0]:.10f} mV^2, empirical={cov_np:.10f} mV^2"
        )

        # check voltage statistics
        assert np.abs(mu_empirical - mu_exact) / np.abs(mu_exact) < 0.05
        assert np.abs(cov_np - cov_exact[0]) / cov_exact[0] < 0.1
        assert (
            np.mean(np.abs(cov_empirical[:500] - cov_exact[:500]) / cov_exact[0]) < 0.05
        )

        # the same seed yields the same noise, a different seed different noise
        res_ = {}
        for seed in [46, 47]:
            self._create_ball()
            self.tree.init_model(t_calibrate=t_calibrate, dt=dt)
            self.tree.add_ou_clamp(
                loc=(1, 0.5),
                mean=mu_ou,
                stdev=sigma_ou,
                tau=tau_ou,
                delay=-t_calibrate / 2.0,
                dur=tmax + t_calibrate / 2.0,
                seed=seed,
            )
            res_[seed] = self.tree.run(100.0, record_from_iclamps=True)
            self.tree.delete_model()
        assert np.allclose(res_[46]["i_clamp"], res["i_clamp"][:, : len(res_[46]["t"])])
        assert not np.allclose(res_[46]["i_clamp"], res_[47]["i_clamp"])

        if pplot:
            pl.figure(figsize=(12, 5))
            ax = pl.subplot(1, 3, 1)
            ax.plot(res["t"], res["v_m"][0, :])
            ax.axhline(mu_exact)
            ax.axhline(mu_empirical, color="orange")
            ax = pl.subplot(1, 3, 2)
            ax.plot(res["t"][:500], cov_exact[:500], label="Exact")
            ax.plot(res["t"][:500], cov_empirical[:500], label="Empirical")
            ax.axhline(cov_np, color="gray", linestyle="--", label="Empirical variance")
            ax.legend(loc=0)
            pl.show()

    def test_input_spiketrain(self, pplot=False):
        # parameter setup
        dt = 0.1
        tmax = 100.0
        t_calibrate = 50.0
        spktms1 = [10.0, 20.0, 40.0]
        spktms2 = [15.0, 25.0, 45.0]

        self._create_ball()
        self.tree.init_model(t_calibrate=t_calibrate, dt=dt)
        self.tree.add_double_exp_synapse(loc=(1, 0.5), tau1=0.2, tau2=3.0, e_r=0.0)
        self.tree.add_double_exp_synapse(loc=(1, 0.5), tau1=0.2, tau2=3.0, e_r=0.0)
        self.tree.set_spiketrain(0, 0.01, spktms1)
        self.tree.set_spiketrain(1, 0.005, spktms2)

        res = self.tree.run(tmax)
        self.tree.delete_model()

        assert (
            np.max(res["v_m"][0]) > res["v_m"][0, 0] + 0.1
        )  # check for depolarization

        if pplot:
            pl.plot(res["t"], res["v_m"][0, :])
            pl.show()

    def test_synapses_neuron_equivalence(self, pplot=False):
        """
        Test whether the point processes in Arbor yield the same responses as
        their NEURON counterparts
        """
        dt = 0.025
        tmax = 300.0
        t_calibrate = 50.0
        spktms = [10.0, 20.0, 40.0, 150.0]

        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc")
        res = {}
        for sim_tree_class in [NeuronSimTree, ArborSimTree]:
            sim_tree = sim_tree_class(fname, types=[1, 3, 4])
            sim_tree.fit_leak_current(-75.0, 10.0)
            sim_tree.set_comp_tree()
            sim_tree.init_model(t_calibrate=t_calibrate, dt=dt)
            sim_tree.store_locs([(1, 0.5)], name="rec locs")
            sim_tree.add_shunt((1, 0.5), 0.001, -60.0)
            sim_tree.add_exp_synapse((1, 0.5), tau=2.0, e_r=0.0)
            sim_tree.add_double_exp_synapse((1, 0.5), tau1=0.2, tau2=3.0, e_r=-80.0)
            sim_tree.add_double_exp_current((1, 0.5), tau1=0.2, tau2=3.0)
            sim_tree.add_nmda_synapse((1, 0.5), tau=2.0, tau_nmda=80.0, e_r=0.0)
            sim_tree.add_double_exp_nmda_synapse(
                (1, 0.5), tau1=0.2, tau2=3.0, tau1_nmda=0.2, tau2_nmda=43.0
            )
            for syn_idx, weight in enumerate([0.005, 0.002, 0.05, 0.002, 0.003]):
                sim_tree.set_spiketrain(syn_idx, weight, spktms)
            res[sim_tree_class] = sim_tree.run(tmax, record_from_syns=True)
            sim_tree.delete_model()
        res_neuron, res_arbor = res[NeuronSimTree], res[ArborSimTree]

        if pplot:
            pl.figure()
            ax = pl.subplot(121)
            ax.plot(res_neuron["t"], res_neuron["v_m"][0], "b")
            ax.plot(res_arbor["t"], res_arbor["v_m"][0], "r--")
            ax = pl.subplot(122)
            for ii in range(len(res_neuron["i_syn"])):
                ax.plot(res_neuron["t"], res_neuron["i_syn"][ii], c=colours[ii])
                ax.plot(res_arbor["t"], res_arbor["i_syn"][ii], ls="--", c=colours[ii])
            pl.show()

        assert np.max(res_neuron["v_m"][0]) > res_neuron["v_m"][0, 0] + 1.0
        assert np.allclose(res_neuron["v_m"], res_arbor["v_m"], atol=0.05)
        # NEURON records the currents of the time step that ends at the
        # recording time, Arbor the currents at the recording time
        assert np.allclose(
            res_neuron["i_syn"][:, 1:], res_arbor["i_syn"][:, :-1], atol=2e-3
        )


if __name__ == "__main__":
    ta = TestArbor()
    ta.test_passive(pplot=True)
    ta.test_active(pplot=True)
    ta.test_channel_recording()
    ta.test_recording_timestep()
    ta.test_calc_e_eq()
    ta.test_neuron_equivalence()

    tra = TestReducedArbor()
    tra.test_discretization()
    tra.test_impedance_properties()
    tra.test_neuron_equivalence(pplot=True)

    ts = TestStimuli()
    ts.test_i_clamp(pplot=True)
    ts.test_sin_clamp()
    ts.test_v_clamp()
    ts.test_ou_processes(pplot=True)
    ts.test_input_spiketrain(pplot=True)
    ts.test_synapses_neuron_equivalence(pplot=True)
