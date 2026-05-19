# -*- coding: utf-8 -*-
#
# test_brian2tree.py
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

import os

import matplotlib.pyplot as pl
import numpy as np
import pytest

try:
    import brian2
except ImportError:
    pytest.skip("Brian2 not installed", allow_module_level=True)

from neat import PhysTree
from neat import CompartmentNode, CompartmentTree
from neat import CompartmentFitter, NeuronCompartmentTree
from neat import Brian2CompartmentTree

import channelcollection_for_tests as channelcollection
import channel_installer

channel_installer.load_or_install_neuron_test_channels()


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_morphologies")
)


class TestBrian2:
    def load_two_compartment_model(self):
        pnode = CompartmentNode(0, ca=1.5e-5, g_l=2e-3)
        self.ctree = CompartmentTree(pnode)
        cnode = CompartmentNode(1, ca=2e-6, g_l=3e-4, g_c=4e-3)
        self.ctree.add_node_with_parent(cnode, pnode)

        for ii, cn in enumerate(self.ctree):
            cn.loc_idx = ii

    def test_model_construction(self):
        self.load_two_compartment_model()

        bct = Brian2CompartmentTree(self.ctree)
        bmodel = bct.init_model()

        assert len(bmodel.v) == 2
        assert bct.get_compartment_index(0) == 0
        assert bct.get_compartment_index(1) == 1

    def load_ball(self):
        self.tree = PhysTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc"))
        self.tree.set_physiology(0.8, 100.0 / 1e6)
        self.k_chan = channelcollection.Kv3_1()
        self.tree.add_channel_current(self.k_chan, 0.766 * 1e6, -85.0)
        self.na_chan = channelcollection.NaTa_t()
        self.tree.add_channel_current(self.na_chan, 1.71 * 1e6, 50.0)
        self.tree.fit_leak_current(-75.0, 10.0)
        self.tree.set_v_ep(-75.0)
        self.tree.set_comp_tree()

        cfit = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        self.ctree, _ = cfit.fit_model([(1, 0.5)])

    def _run_brian2_model(
        self,
        simtree,
        rec_idxs,
        current_steps=None,
        tmax=200.0,
        tcal=200.0,
        dt=0.001,
        record_vars=None,
    ):
        if current_steps is None:
            current_steps = []
        if record_vars is None:
            record_vars = []

        brian2.start_scope()
        brian2.defaultclock.dt = dt * brian2.ms

        bmodel = simtree.init_model(
            threshold="v > 1e9*volt",
            refractory=0.0 * brian2.ms,
        )

        rec_vars = ["v"] + record_vars
        mon = brian2.StateMonitor(bmodel, rec_vars, record=rec_idxs)
        net = brian2.Network(brian2.collect())

        current_steps = [
            {
                "comp_idx": int(step["comp_idx"]),
                "amp": float(step["amp"]),
                "delay": float(step["delay"]),
                "dur": float(step["dur"]),
            }
            for step in current_steps
        ]

        event_times = {0.0, float(tmax)}
        for step in current_steps:
            event_times.add(step["delay"])
            event_times.add(step["delay"] + step["dur"])
        event_times = sorted(event_times)

        bmodel.I_ext = 0.0 * brian2.nA
        net.run(tcal * brian2.ms)

        for t0, t1 in zip(event_times[:-1], event_times[1:]):
            bmodel.I_ext = 0.0 * brian2.nA
            for step in current_steps:
                if step["delay"] <= t0 < step["delay"] + step["dur"]:
                    bmodel.I_ext[step["comp_idx"]] = step["amp"] * brian2.nA
            if t1 > t0:
                net.run((t1 - t0) * brian2.ms)

        t = np.array(mon.t / brian2.ms)
        mask = t >= tcal

        res = {
            "t": t[mask] - tcal,
            "v_m": np.array(
                [np.array(mon.v[ii] / brian2.mV)[mask] for ii in range(len(rec_idxs))]
            ),
        }
        for var in record_vars:
            res[var] = np.array(
                [np.array(getattr(mon, var)[ii])[mask] for ii in range(len(rec_idxs))]
            )

        return res

    def _run_neuron_model(
        self,
        simtree,
        rec_locs,
        current_steps=None,
        tmax=200.0,
        tcal=200.0,
        dt=0.001,
        record_channels=False,
    ):
        if current_steps is None:
            current_steps = []

        simtree.init_model(dt=dt, t_calibrate=tcal)
        simtree.store_locs(rec_locs, name="rec locs")

        for step in current_steps:
            simtree.add_i_clamp(
                step["comp_idx"], step["amp"], step["delay"], step["dur"]
            )

        return simtree.run(tmax, record_from_channels=record_channels)

    def test_initialization(self):
        dt = 0.1
        v_eq = -65.0
        self.load_ball()
        self.tree.fit_leak_current(v_eq, 10.0)
        self.tree.set_comp_tree()

        cfit = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        self.ctree, _ = cfit.fit_model([(1, 0.5)])

        bsimtree = Brian2CompartmentTree(
            self.ctree, channel_storage=self.ctree.channel_storage
        )
        res_brian2 = self._run_brian2_model(
            bsimtree,
            rec_idxs=[0],
            tmax=400.0,
            tcal=0.0,
            dt=dt,
            record_vars=["m_Kv3_1", "m_NaTa_t", "h_NaTa_t"],
        )

        sv_na = self.na_chan.compute_varinf(v_eq)
        sv_k = self.k_chan.compute_varinf(v_eq)

        assert np.abs(res_brian2["v_m"][0][0] - v_eq) < 1e-8
        assert np.abs(res_brian2["m_Kv3_1"][0][0] - sv_k["m"]) < 1e-8
        assert np.abs(res_brian2["m_NaTa_t"][0][0] - sv_na["m"]) < 1e-8
        assert np.abs(res_brian2["h_NaTa_t"][0][0] - sv_na["h"]) < 1e-8
        assert np.abs(res_brian2["v_m"][0][-1] - v_eq) < 1e-6
        assert np.abs(res_brian2["m_Kv3_1"][0][-1] - sv_k["m"]) < 1e-6
        assert np.abs(res_brian2["m_NaTa_t"][0][-1] - sv_na["m"]) < 1e-6
        assert np.abs(res_brian2["h_NaTa_t"][0][-1] - sv_na["h"]) < 1e-6

    def test_single_comp_brian2_neuron_comparison(self, pplot=False):
        dt = 0.001
        self.load_ball()

        current_steps = [{"comp_idx": 0, "amp": 0.01, "delay": 20.0, "dur": 40.0}]
        rec_locs = [(0, 0.5)]

        res_neuron = self._run_neuron_model(
            NeuronCompartmentTree(self.ctree),
            rec_locs,
            current_steps=current_steps,
            tmax=200.0,
            tcal=200.0,
            dt=dt,
        )
        res_brian2 = self._run_brian2_model(
            Brian2CompartmentTree(
                self.ctree, channel_storage=self.ctree.channel_storage
            ),
            rec_idxs=[0],
            current_steps=current_steps,
            tmax=200.0,
            tcal=200.0,
            dt=dt,
        )

        idx = min(len(res_neuron["t"]), len(res_brian2["t"]))
        assert (
            np.sqrt(
                np.mean((res_brian2["v_m"][0][:idx] - res_neuron["v_m"][0][:idx]) ** 2)
            )
            < 0.2
        )
        assert np.allclose(
            res_brian2["v_m"][0][:idx], res_neuron["v_m"][0][:idx], atol=1.0
        )

        if pplot:
            pl.plot(res_neuron["t"][:idx], res_neuron["v_m"][0][:idx], "rx-")
            pl.plot(res_brian2["t"][:idx], res_brian2["v_m"][0][:idx], "bo--")
            pl.show()

    def load_axon_tree(self):
        tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball_and_axon.swc"),
            types=[1, 2, 3, 4],
        )
        tree.set_physiology(1.0, 100.0 / 1e6)
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
        tree.set_leak_current(0.000091 * 1e6, -62.442793, node_arg=[tree[1]])
        tree.set_leak_current(0.000094 * 1e6, -79.315740, node_arg="axonal")

        locs = [(1, 0.5), (4.0, 0.5), (5, 0.5)]
        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        self.ctree, _ = cfit.fit_model(locs)

    def test_axon_brian2_neuron_comparison(self, pplot=False):
        dt = 0.001
        self.load_axon_tree()

        current_steps = [{"comp_idx": 0, "amp": 0.01, "delay": 20.0, "dur": 40.0}]
        rec_locs = [(0, 0.5), (1, 0.5), (2.0, 0.5)]

        res_neuron = self._run_neuron_model(
            NeuronCompartmentTree(self.ctree),
            rec_locs,
            current_steps=current_steps,
            tmax=200.0,
            tcal=200.0,
            dt=dt,
        )
        res_brian2 = self._run_brian2_model(
            Brian2CompartmentTree(
                self.ctree, channel_storage=self.ctree.channel_storage
            ),
            rec_idxs=[0, 1, 2],
            current_steps=current_steps,
            tmax=200.0,
            tcal=200.0,
            dt=dt,
        )

        idx = min(len(res_neuron["t"]), len(res_brian2["t"]))
        for ii in range(3):
            assert (
                np.sqrt(
                    np.mean(
                        (res_brian2["v_m"][ii][:idx] - res_neuron["v_m"][ii][:idx]) ** 2
                    )
                )
                < 0.3
            )
            assert np.allclose(
                res_brian2["v_m"][ii][:idx], res_neuron["v_m"][ii][:idx], atol=1.5
            )

        if pplot:
            pl.figure(figsize=(15, 6))
            for ii in range(3):
                ax = pl.subplot(1, 3, ii + 1)
                ax.plot(res_neuron["t"][:idx], res_neuron["v_m"][ii][:idx], "rx-")
                ax.plot(res_brian2["t"][:idx], res_brian2["v_m"][ii][:idx], "bo--")
            pl.show()

    def load_T_tree(self):
        tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, "Ttree_segments.swc"),
            types=[1, 2, 3, 4],
        )
        tree.set_physiology(1.0, 100.0 / 1e6)
        k_chan = channelcollection.SKv3_1()
        tree.add_channel_current(k_chan, 0.653374 * 1e6, -85.0, node_arg=[tree[1]])
        na_chan = channelcollection.NaTa_t()
        tree.add_channel_current(na_chan, 0.15 * 1e6, 50.0, node_arg=[tree[1]])
        ca_chan = channelcollection.Ca_HVA()
        tree.add_channel_current(
            ca_chan, 0.005 * 1e6, 132.4579341637009, node_arg=[tree[1]]
        )
        tree.fit_leak_current(-70.0, 15.0)

        locs = [(n.index, 0.5) for n in tree]
        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        self.ctree, _ = cfit.fit_model(locs)

    def test_dend_brian2_neuron_comparison(self, pplot=False):
        dt = 0.01
        tmax = 400.0
        tcal = 500.0
        t1 = 200.0
        rec_idx = 7
        dend_idx = 9

        self.load_T_tree()

        current_steps = [
            {"comp_idx": 0, "amp": 0.01, "delay": t1 + 20.0, "dur": 20.0},
            {"comp_idx": dend_idx, "amp": 0.01, "delay": t1 + 70.0, "dur": 20.0},
        ]
        clocs = [(ii, 0.5) for ii in range(len(self.ctree))]
        record_vars = ["m_Ca_HVA", "h_Ca_HVA", "m_NaTa_t", "h_NaTa_t"]

        res_neuron = self._run_neuron_model(
            NeuronCompartmentTree(self.ctree),
            clocs,
            current_steps=current_steps,
            tmax=tmax,
            tcal=tcal,
            dt=dt,
            record_channels=True,
        )
        res_brian2 = self._run_brian2_model(
            Brian2CompartmentTree(
                self.ctree, channel_storage=self.ctree.channel_storage
            ),
            rec_idxs=list(range(len(self.ctree))),
            current_steps=current_steps,
            tmax=tmax,
            tcal=tcal,
            dt=dt,
            record_vars=record_vars,
        )

        idx = min(len(res_neuron["t"]), len(res_brian2["t"]))
        for ii in range(len(self.ctree)):
            v_maxdiff = np.max(
                np.abs(res_brian2["v_m"][ii][:idx] - res_neuron["v_m"][ii][:idx])
            )
            v_meandiff = np.mean(
                np.abs(res_brian2["v_m"][ii][:idx] - res_neuron["v_m"][ii][:idx])
            )
            assert v_maxdiff < 3.0
            assert v_meandiff < 0.05

        assert (
            np.max(
                np.abs(
                    res_brian2["m_Ca_HVA"][0][:idx]
                    - res_neuron["chan"]["Ca_HVA"]["m"][0][:idx]
                )
            )
            < 0.05
        )
        assert (
            np.max(
                np.abs(
                    res_brian2["h_Ca_HVA"][0][:idx]
                    - res_neuron["chan"]["Ca_HVA"]["h"][0][:idx]
                )
            )
            < 0.05
        )
        assert (
            np.max(
                np.abs(
                    res_brian2["m_NaTa_t"][0][:idx]
                    - res_neuron["chan"]["NaTa_t"]["m"][0][:idx]
                )
            )
            < 0.05
        )
        assert (
            np.max(
                np.abs(
                    res_brian2["h_NaTa_t"][0][:idx]
                    - res_neuron["chan"]["NaTa_t"]["h"][0][:idx]
                )
            )
            < 0.05
        )

        if pplot:
            pl.figure("v", figsize=(15, 6))
            ax = pl.subplot(131)
            ax.plot(res_neuron["t"][:idx], res_neuron["v_m"][0][:idx], "r-")
            ax.plot(res_brian2["t"][:idx], res_brian2["v_m"][0][:idx], "b--")
            ax = pl.subplot(132)
            ax.plot(res_neuron["t"][:idx], res_neuron["v_m"][rec_idx][:idx], "r-")
            ax.plot(res_brian2["t"][:idx], res_brian2["v_m"][rec_idx][:idx], "b--")
            ax = pl.subplot(133)
            ax.plot(res_neuron["t"][:idx], res_neuron["v_m"][dend_idx][:idx], "r-")
            ax.plot(res_brian2["t"][:idx], res_brian2["v_m"][dend_idx][:idx], "b--")
            pl.show()


if __name__ == "__main__":
    tb = TestBrian2()
    tb.test_single_comp_brian2_neuron_comparison(pplot=True)
