# -*- coding: utf-8 -*-
#
# test_nesttree.py
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
import time
import pytest

try:
    import nest
    import nest.lib.hl_api_exceptions as nestexceptions
except ImportError as e:
    # pytest.skip("NEST not installed", allow_module_level=True)
    pass

from neat import PhysTree
from neat import CompartmentNode, CompartmentTree
from neat import CompartmentFitter, NeuronCompartmentTree
from neat import NestCompartmentNode, NestCompartmentTree, load_nest_model

import channelcollection_for_tests as channelcollection
import channel_installer

channel_installer.load_or_install_neuron_test_channels()


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_morphologies")
)


class TestNest:
    def load_two_compartment_model(self):
        # simple two compartment model
        pnode = CompartmentNode(0, ca=1.5e-5, g_l=2e-3)
        self.ctree = CompartmentTree(pnode)
        cnode = CompartmentNode(1, ca=2e-6, g_l=3e-4, g_c=4e-3)
        self.ctree.add_node_with_parent(cnode, pnode)

        for ii, cn in enumerate(self.ctree):
            cn.loc_idx = ii

    def test_model_construction(self):
        self.load_two_compartment_model()

        nct = NestCompartmentTree(self.ctree)
        cm_model = nct.init_model("cm_default", 1, suffix="")

        compartments_info = cm_model.compartments
        assert compartments_info[0]["comp_idx"] == 0
        assert compartments_info[0]["parent_idx"] == -1
        assert compartments_info[1]["comp_idx"] == 1
        assert compartments_info[1]["parent_idx"] == 0

    def load_ball(self):
        """
        Load point neuron model
        """
        self.tree = PhysTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc"))
        # capacitance and axial resistance
        self.tree.set_physiology(0.8, 100.0 / 1e6)
        # ion channels
        self.k_chan = channelcollection.Kv3_1()
        self.tree.add_channel_current(self.k_chan, 0.766 * 1e6, -85.0)
        self.na_chan = channelcollection.NaTa_t()
        self.tree.add_channel_current(self.na_chan, 1.71 * 1e6, 50.0)
        # fit leak current
        self.tree.fit_leak_current(-75.0, 10.0)
        # set equilibirum potententials
        self.tree.set_v_ep(-75.0)
        # set computational tree
        self.tree.set_comp_tree()

        cfit = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        self.ctree, _ = cfit.fit_model([(1, 0.5)])

    def test_initialization(self):
        dt = 0.1
        nest.ResetKernel()
        channel_installer.load_or_install_nest_test_channels()
        nest.SetKernelStatus(dict(resolution=dt))

        v_eq = -65.0
        self.load_ball()
        self.tree.fit_leak_current(v_eq, 10.0)
        # set computational tree
        self.tree.set_comp_tree()
        # fit the tree again
        cfit = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        self.ctree, _ = cfit.fit_model([(1, 0.5)])
        csimtree_nest = NestCompartmentTree(self.ctree)
        nestmodel = csimtree_nest.init_model("multichannel_test", 1)
        mm = nest.Create(
            "multimeter",
            1,
            {
                "record_from": ["v_comp0", "m_Kv3_10", "m_NaTa_t0", "h_NaTa_t0"],
                "interval": dt,
            },
        )
        nest.Connect(mm, nestmodel)
        # simulate
        nest.Simulate(400.0)
        res_nest = nest.GetStatus(mm, "events")[0]

        sv_na = self.na_chan.compute_varinf(v_eq)
        sv_k = self.k_chan.compute_varinf(v_eq)

        assert np.abs(res_nest["v_comp0"][0] - v_eq) < 1e-8
        assert np.abs(res_nest["m_Kv3_10"][0] - sv_k["m"]) < 1e-8
        assert np.abs(res_nest["m_NaTa_t0"][0] - sv_na["m"]) < 1e-8
        assert np.abs(res_nest["h_NaTa_t0"][0] - sv_na["h"]) < 1e-8
        assert np.abs(res_nest["v_comp0"][-1] - v_eq) < 1e-8
        assert np.abs(res_nest["m_Kv3_10"][-1] - sv_k["m"]) < 1e-8
        assert np.abs(res_nest["m_NaTa_t0"][-1] - sv_na["m"]) < 1e-8
        assert np.abs(res_nest["h_NaTa_t0"][-1] - sv_na["h"]) < 1e-8

    def test_single_comp_nest_neuron_comparison(self, pplot=False):
        dt = 0.001
        # nest.ResetKernel()
        # channel_installer.load_or_install_nest_test_channels()
        # nest.SetKernelStatus(dict(resolution=dt))

        self.load_ball()
        csimtree_neuron = NeuronCompartmentTree(self.ctree)
        csimtree_neuron.init_model(dt=dt, t_calibrate=200.0)
        csimtree_neuron.store_locs([(0, 0.5)], name="rec locs")
        csimtree_neuron.add_double_exp_synapse((0, 0.5), 0.2, 3.0, 0.0)
        csimtree_neuron.set_spiketrain(0, 0.001, [20.0, 23.0, 40.0])
        res_neuron = csimtree_neuron.run(200.0)

        # csimtree_nest = NestCompartmentTree(self.ctree)
        # nestmodel = csimtree_nest.init_model("multichannel_test", 1)
        # # inputs
        # nestmodel.receptors = [
        #     {
        #         "comp_idx": 0,
        #         "receptor_type": "i_AMPA",
        #         "params": {"e_AMPA": 0.0, "tau_r_AMPA": 0.2, "tau_d_AMPA": 3.0},
        #     }
        # ]
        # sg = nest.Create("spike_generator", 1, {"spike_times": [220.0, 223.0, 240.0]})
        # nest.Connect(
        #     sg,
        #     nestmodel,
        #     syn_spec={
        #         "synapse_model": "static_synapse",
        #         "weight": 0.001,
        #         "delay": 3 * dt,
        #         "receptor_type": 0,
        #     },
        # )
        # # voltage recording
        # mm = nest.Create("multimeter", 1, {"record_from": ["v_comp0"], "interval": dt})
        # nest.Connect(mm, nestmodel)
        # # simulate
        # nest.Simulate(400.0)
        # res_nest = nest.GetStatus(mm, "events")[0]

        # idx0 = int(200.0 / dt)
        # res_nest["times"] = res_nest["times"][idx0:] - res_nest["times"][idx0]
        # res_nest["v_comp0"] = res_nest["v_comp0"][idx0:]
        # v0 = res_nest["v_comp0"][0]

        # idx1 = min(len(res_neuron["v_m"][0]), len(res_nest["v_comp0"]))
        # assert (
        #     np.sqrt(
        #         np.mean((res_nest["v_comp0"][:idx1] - res_neuron["v_m"][0][:idx1]) ** 2)
        #     )
        #     < 0.05
        # )
        # assert np.allclose(
        #     res_nest["v_comp0"][:idx1], res_neuron["v_m"][0][:idx1], atol=4.0
        # )

        if pplot:
            pl.plot(res_neuron["t"][:], res_neuron["v_m"][0][:], "rx-")
            # pl.plot(res_neuron["t"][:idx1], res_neuron["v_m"][0][:idx1], "rx-")
            # pl.plot(res_nest["times"][:idx1], res_nest["v_comp0"][:idx1], "bo--")
            pl.show()

    def load_axon_tree(self):
        """
        Parameters taken from a BBP SST model for a subset of ion channels
        """
        tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball_and_axon.swc"),
            types=[1, 2, 3, 4],
        )
        # capacitance and axial resistance
        tree.set_physiology(1.0, 100.0 / 1e6)
        # ion channels
        k_chan = channelcollection.SKv3_1()
        tree.add_channel_current(k_chan, 0.653374 * 1e6, -85.0, node_arg=[tree[1]])
        tree.add_channel_current(k_chan, 0.196957 * 1e6, -85.0, node_arg="axonal")
        na_chan = channelcollection.NaTa_t()
        tree.add_channel_current(na_chan, 3.418459 * 1e6, 50.0, node_arg="axonal")
        ca_chan = channelcollection.Ca_HVA()
        tree.add_channel_current(
            ca_chan, 0.000792 * 1e6, 132.4579341637009, node_arg=[tree[1]]
        )
        tree.add_channel_current(
            ca_chan, 0.000138 * 1e6, 132.4579341637009, node_arg="axonal"
        )
        # passive leak current
        tree.set_leak_current(0.000091 * 1e6, -62.442793, node_arg=[tree[1]])
        tree.set_leak_current(0.000094 * 1e6, -79.315740, node_arg="axonal")

        # simplify
        locs = [(1, 0.5), (4.0, 0.5), (5, 0.5)]
        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        self.ctree, _ = cfit.fit_model(locs)

    def test_axon_nest_neuron_comparison(self, pplot=False):
        dt = 0.001
        nest.ResetKernel()
        channel_installer.load_or_install_nest_test_channels()
        nest.SetKernelStatus(dict(resolution=dt))

        self.load_axon_tree()

        csimtree_neuron = NeuronCompartmentTree(self.ctree)
        csimtree_neuron.init_model(dt=dt, t_calibrate=200.0)
        csimtree_neuron.store_locs([(0, 0.5), (1, 0.5), (2.0, 0.5)], name="rec locs")
        csimtree_neuron.add_double_exp_synapse((0, 0.5), 0.2, 3.0, 0.0)
        csimtree_neuron.set_spiketrain(0, 0.001, [20.0, 23.0, 40.0])
        res_neuron = csimtree_neuron.run(200.0)

        csimtree_nest = NestCompartmentTree(self.ctree)
        nestmodel = csimtree_nest.init_model("multichannel_test", 1)
        # inputs
        nestmodel.receptors = [
            {
                "comp_idx": 0,
                "receptor_type": "i_AMPA",
                "params": {"e_AMPA": 0.0, "tau_r_AMPA": 0.2, "tau_d_AMPA": 3.0},
            }
        ]
        sg = nest.Create("spike_generator", 1, {"spike_times": [220.0, 223.0, 240.0]})
        nest.Connect(
            sg,
            nestmodel,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": 0.001,
                "delay": dt,
                "receptor_type": 0,
            },
        )
        # voltage recording
        mm = nest.Create(
            "multimeter",
            1,
            {"record_from": ["v_comp0", "v_comp1", "v_comp2"], "interval": dt},
        )
        nest.Connect(mm, nestmodel)
        # simulate
        nest.Simulate(400.0)
        res_nest = nest.GetStatus(mm, "events")[0]

        idx0 = int(200.0 / dt)
        res_nest["times"] = res_nest["times"][idx0:] - res_nest["times"][idx0]
        res_nest["v_comp0"] = res_nest["v_comp0"][idx0:]
        res_nest["v_comp1"] = res_nest["v_comp1"][idx0:]
        res_nest["v_comp2"] = res_nest["v_comp2"][idx0:]
        v0 = res_nest["v_comp0"][0]

        idx1 = min(len(res_neuron["v_m"][0]), len(res_nest["v_comp0"]))
        assert (
            np.sqrt(
                np.mean((res_nest["v_comp0"][:idx1] - res_neuron["v_m"][0][:idx1]) ** 2)
            )
            < 0.05
        )
        assert np.allclose(
            res_nest["v_comp0"][:idx1], res_neuron["v_m"][0][:idx1], atol=1.0
        )

        if pplot:
            pl.figure(figsize=(15, 6))
            ax = pl.subplot(131)
            ax.plot(res_neuron["t"], res_neuron["v_m"][0], "rx-")
            ax.plot(res_nest["times"], res_nest["v_comp0"], "bo--")
            ax = pl.subplot(132)
            ax.plot(res_neuron["t"], res_neuron["v_m"][1], "rx-")
            ax.plot(res_nest["times"], res_nest["v_comp1"], "bo--")
            ax = pl.subplot(133)
            ax.plot(res_neuron["t"], res_neuron["v_m"][2], "rx-")
            ax.plot(res_nest["times"], res_nest["v_comp2"], "bo--")
            pl.show()

    def load_T_tree(self):
        """
        Parameters taken from a BBP SST model for a subset of ion channels
        """
        tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "Ttree_segments.swc"),
            types=[1, 2, 3, 4],
        )
        # capacitance and axial resistance
        tree.set_physiology(1.0, 100.0 / 1e6)
        # ion channels
        k_chan = channelcollection.SKv3_1()
        tree.add_channel_current(k_chan, 0.653374 * 1e6, -85.0, node_arg=[tree[1]])
        na_chan = channelcollection.NaTa_t()
        # tree.add_channel_current(na_chan, 3.418459 * 1e6, 50., node_arg=[tree[1]])
        tree.add_channel_current(na_chan, 0.15 * 1e6, 50.0, node_arg=[tree[1]])
        ca_chan = channelcollection.Ca_HVA()
        # tree.add_channel_current(ca_chan, 0.000792 * 1e6, 132.4579341637009, node_arg=[tree[1]])
        tree.add_channel_current(
            ca_chan, 0.005 * 1e6, 132.4579341637009, node_arg=[tree[1]]
        )
        # passive leak current
        tree.fit_leak_current(-70.0, 15.0)

        # simplify
        locs = [(n.index, 0.5) for n in tree]
        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        self.ctree, _ = cfit.fit_model(locs)

    def test_dend_nest_neuron_comparison(self, pplot=False):
        dt = 0.01
        tmax = 400.0
        tcal = 500.0
        t1 = 200.0
        idx0 = int(tcal / dt) - 2
        nest.ResetKernel()
        channel_installer.load_or_install_nest_test_channels()
        nest.SetKernelStatus(dict(resolution=dt))

        self.load_T_tree()

        rec_idx = 7
        dend_idx = 9

        clocs = [(ii, 0.5) for ii in range(len(self.ctree))]
        csimtree_neuron = NeuronCompartmentTree(self.ctree)
        csimtree_neuron.init_model(dt=dt, t_calibrate=tcal)
        csimtree_neuron.store_locs(clocs, name="rec locs")
        csimtree_neuron.add_double_exp_synapse(clocs[0], 0.2, 3.0, 0.0)
        csimtree_neuron.set_spiketrain(0, 0.005, [t1 + 20.0, t1 + 23.0, t1 + 40.0])
        csimtree_neuron.add_double_exp_synapse(clocs[dend_idx], 0.2, 3.0, 0.0)
        csimtree_neuron.set_spiketrain(1, 0.005, [t1 + 70.0, t1 + 74.0, t1 + 85.0])
        res_neuron = csimtree_neuron.run(tmax, record_from_channels=True, pprint=True)

        csimtree_nest = NestCompartmentTree(self.ctree)
        nestmodel = csimtree_nest.init_model("multichannel_test", 1)
        # inputs
        nestmodel.receptors = [
            {
                "comp_idx": 0,
                "receptor_type": "i_AMPA",
                "params": {"e_AMPA": 0.0, "tau_r_AMPA": 0.2, "tau_d_AMPA": 3.0},
            },
            {
                "comp_idx": dend_idx,
                "receptor_type": "i_AMPA",
                "params": {"e_AMPA": 0.0, "tau_r_AMPA": 0.2, "tau_d_AMPA": 3.0},
            },
        ]
        sg = nest.Create(
            "spike_generator",
            1,
            {"spike_times": [tcal + t1 + 20.0, tcal + t1 + 23.0, tcal + t1 + 40.0]},
        )
        nest.Connect(
            sg,
            nestmodel,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": 0.005,
                "delay": dt,
                "receptor_type": 0,
            },
        )
        sg_ = nest.Create(
            "spike_generator",
            1,
            {"spike_times": [tcal + t1 + 70.0, tcal + t1 + 74.0, tcal + t1 + 85.0]},
        )
        nest.Connect(
            sg_,
            nestmodel,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": 0.005,
                "delay": dt,
                "receptor_type": 1,
            },
        )
        # voltage recording
        mm = nest.Create(
            "multimeter",
            1,
            {
                "record_from": [f"v_comp{ii}" for ii in range(len(self.ctree))]
                + [f"h_Ca_HVA{ii}" for ii in range(len(self.ctree))]
                + [f"m_Ca_HVA{ii}" for ii in range(len(self.ctree))]
                + [f"h_NaTa_t{ii}" for ii in range(len(self.ctree))]
                + [f"m_NaTa_t{ii}" for ii in range(len(self.ctree))],
                "interval": dt,
            },
        )
        nest.Connect(mm, nestmodel)
        # simulate
        nest.Prepare()
        nest.Run(tmax + tcal)
        nest.Cleanup()
        res_nest = nest.GetStatus(mm, "events")[0]

        res_nest["times"] = res_nest["times"][idx0:] - res_nest["times"][idx0]
        for ii in range(len(self.ctree)):
            res_nest[f"v_comp{ii}"] = res_nest[f"v_comp{ii}"][idx0:]
            res_nest[f"h_Ca_HVA{ii}"] = res_nest[f"h_Ca_HVA{ii}"][idx0:]
            res_nest[f"m_Ca_HVA{ii}"] = res_nest[f"m_Ca_HVA{ii}"][idx0:]
            res_nest[f"h_NaTa_t{ii}"] = res_nest[f"h_NaTa_t{ii}"][idx0:]
            res_nest[f"m_NaTa_t{ii}"] = res_nest[f"m_NaTa_t{ii}"][idx0:]

        idx1 = min(len(res_neuron["v_m"][0]), len(res_nest["v_comp0"]))
        for ii in range(len(self.ctree)):
            v0 = res_nest[f"v_comp{ii}"][0]
            v_maxdiff = np.max(
                np.abs(res_nest[f"v_comp{ii}"][:idx1] - res_neuron["v_m"][ii, :idx1])
            )
            v_meandiff = np.mean(
                np.abs(res_nest[f"v_comp{ii}"][:idx1] - res_neuron["v_m"][ii, :idx1])
            )
            assert v_maxdiff < 2 and v_meandiff < 0.005

        if pplot:
            pl.figure("v", figsize=(15, 6))
            ax = pl.subplot(131)
            ax.plot(res_neuron["t"], res_neuron["v_m"][0], "r-")
            ax.plot(res_nest["times"], res_nest["v_comp0"], "b--")
            ax = pl.subplot(132)
            ax.plot(res_neuron["t"], res_neuron["v_m"][rec_idx], "r-")
            ax.plot(res_nest["times"], res_nest[f"v_comp{rec_idx}"], "b--")
            ax = pl.subplot(133)
            ax.plot(res_neuron["t"], res_neuron["v_m"][dend_idx], "r-")
            ax.plot(res_nest["times"], res_nest[f"v_comp{dend_idx}"], "b--")

            pl.figure("Ca_HVA", figsize=(15, 6))
            ax = pl.subplot(131)
            ax.plot(res_neuron["t"], res_neuron["chan"]["Ca_HVA"]["m"][0], "r-")
            ax.plot(res_neuron["t"], res_neuron["chan"]["Ca_HVA"]["h"][0], "g-")
            ax.plot(res_nest["times"], res_nest[f"m_Ca_HVA0"], "r--", lw=2)
            ax.plot(res_nest["times"], res_nest[f"h_Ca_HVA0"], "g--", lw=2)
            ax = pl.subplot(132)
            ax.plot(res_neuron["t"], res_neuron["chan"]["Ca_HVA"]["m"][rec_idx], "r-")
            ax.plot(res_neuron["t"], res_neuron["chan"]["Ca_HVA"]["h"][rec_idx], "g-")
            ax.plot(res_nest["times"], res_nest[f"m_Ca_HVA{rec_idx}"], "r--", lw=2)
            ax.plot(res_nest["times"], res_nest[f"h_Ca_HVA{rec_idx}"], "g--", lw=2)
            ax = pl.subplot(133)
            ax.plot(res_neuron["t"], res_neuron["chan"]["Ca_HVA"]["m"][dend_idx], "r-")
            ax.plot(res_neuron["t"], res_neuron["chan"]["Ca_HVA"]["h"][dend_idx], "g-")
            ax.plot(res_nest["times"], res_nest[f"m_Ca_HVA{dend_idx}"], "r--", lw=2)
            ax.plot(res_nest["times"], res_nest[f"h_Ca_HVA{dend_idx}"], "g--", lw=2)

            pl.figure("NaTa_t", figsize=(15, 6))
            ax = pl.subplot(131)
            ax.plot(res_neuron["t"], res_neuron["chan"]["NaTa_t"]["m"][0], "r-")
            ax.plot(res_neuron["t"], res_neuron["chan"]["NaTa_t"]["h"][0], "g-")
            ax.plot(res_nest["times"], res_nest[f"m_NaTa_t0"], "r--", lw=2)
            ax.plot(res_nest["times"], res_nest[f"h_NaTa_t0"], "g--", lw=2)
            ax = pl.subplot(132)
            ax.plot(res_neuron["t"], res_neuron["chan"]["NaTa_t"]["m"][rec_idx], "r-")
            ax.plot(res_neuron["t"], res_neuron["chan"]["NaTa_t"]["h"][rec_idx], "g-")
            ax.plot(res_nest["times"], res_nest[f"m_NaTa_t{rec_idx}"], "r--", lw=2)
            ax.plot(res_nest["times"], res_nest[f"h_NaTa_t{rec_idx}"], "g--", lw=2)
            ax = pl.subplot(133)
            ax.plot(res_neuron["t"], res_neuron["chan"]["NaTa_t"]["m"][dend_idx], "r-")
            ax.plot(res_neuron["t"], res_neuron["chan"]["NaTa_t"]["h"][dend_idx], "g-")
            ax.plot(res_nest["times"], res_nest[f"m_NaTa_t{dend_idx}"], "r--", lw=2)
            ax.plot(res_nest["times"], res_nest[f"h_NaTa_t{dend_idx}"], "g--", lw=2)
            pl.show()


if __name__ == "__main__":
    tn = TestNest()
    # tn.test_model_construction()
    # tn.test_initialization()
    tn.test_single_comp_nest_neuron_comparison(pplot=True)
    # tn.test_axon_nest_neuron_comparison(pplot=True)
    # tn.test_dend_nest_neuron_comparison(pplot=True)
