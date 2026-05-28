# -*- coding: utf-8 -*-
#
# test_ionchannels.py
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
import sympy as sp

import pytest
import pickle
import os, shutil

from neat import IonChannel

import channelcollection_for_tests as channelcollection


class test_channels:
    def test_basic(self):
        tcn = channelcollection.test_channel()
        v_arr = np.linspace(-80.0, -10.0, 10)

        factors = np.array([5.0, 1.0])
        powers = np.array([[3, 3, 1], [2, 2, 1]])

        varnames = np.array(
            [
                [sp.symbols("a00"), sp.symbols("a01"), sp.symbols("a02")],
                [sp.symbols("a10"), sp.symbols("a11"), sp.symbols("a12")],
            ]
        )

        # state variable asymptotic values
        def varinf(v):
            aux = np.ones_like(v) if isinstance(v, np.ndarray) else 1.0
            return np.array(
                [
                    [
                        1.0 / (1.0 + np.exp((v - 30.0) / 100.0)),
                        1.0 / (1.0 + np.exp((-v + 30.0) / 100.0)),
                        -10.0 * aux,
                    ],
                    [
                        2.0 / (1.0 + np.exp((v - 30.0) / 100.0)),
                        2.0 / (1.0 + np.exp((-v + 30.0) / 100.0)),
                        -30.0 * aux,
                    ],
                ]
            )

        # state variable functions
        def dvarinf_dv(v):
            aux = np.ones_like(v) if isinstance(v, np.ndarray) else 1.0
            vi_aux = varinf(v)
            return np.array(
                [
                    [
                        -vi_aux[0, 0, :] * (1 - vi_aux[0, 0, :]) / 100.0,
                        vi_aux[0, 1, :] * (1 - vi_aux[0, 1, :]) / 100.0,
                        0.0 * aux,
                    ],
                    [
                        -vi_aux[1, 0, :] * (1 - vi_aux[1, 0, :] / 2.0) / 100.0,
                        vi_aux[1, 1, :] * (1 - vi_aux[1, 1, :] / 2.0) / 100.0,
                        0.0 * aux,
                    ],
                ]
            )

        # state variable relaxation time scale
        def taurel(v):
            aux = np.ones_like(v) if isinstance(v, np.ndarray) else 1.0
            return np.array(
                [[1.0 * aux, 2.0 * aux, 1.0 * aux], [2.0 * aux, 2.0 * aux, 3.0 * aux]]
            )

        # test whether activations are correct
        var_inf = varinf(v_arr)
        var_inf_chan = tcn.compute_varinf(v_arr)
        for ind, varname in np.ndenumerate(varnames):
            assert np.allclose(var_inf[ind], var_inf_chan[varname])

        # test whether open probability is correct
        p_open = np.sum(
            factors[:, np.newaxis] * np.product(var_inf ** powers[:, :, np.newaxis], 1),
            0,
        )
        p_open_ = tcn.compute_p_open(v_arr)
        assert np.allclose(p_open_, p_open)

        # test whether derivatives are correct
        dp_dx_chan, df_dv_chan, df_dx_chan = tcn.compute_derivatives(v_arr)

        # first: derivatives of open probability
        for ind, varname in np.ndenumerate(varnames):
            dp_dx = (
                factors[ind[0]]
                * powers[ind]
                * np.prod(var_inf[ind[0]] ** powers[ind[0]][:, np.newaxis], 0)
                / var_inf[ind]
            )
            assert np.allclose(dp_dx_chan[varname], dp_dx)

        # second: derivatives of state variable functions to voltage
        df_dv = dvarinf_dv(v_arr) / taurel(v_arr)
        for ind, varname in np.ndenumerate(varnames):
            assert np.allclose(df_dv[ind], df_dv_chan[varname])

        # third: derivatives of state variable functions to state variables
        df_dx = -1.0 / taurel(v_arr)
        for ind, varname in np.ndenumerate(varnames):
            assert np.allclose(df_dx[ind], df_dx_chan[varname])


def sp_exp(x):
    return sp.exp(x, evaluate=False)


def test_ionchannel_simplified(remove=True):
    if not os.path.exists("mech/"):
        os.mkdir("mech/")

    na = channelcollection.Na_Ta()

    p_o = na.compute_p_open(-35.0)
    assert np.allclose(p_o, 0.002009216860105564)

    l_s = na.compute_lin_sum(-35.0, 0.0, 50.0)
    assert np.allclose(l_s, -0.00534261017220376)

    na.write_mod_file("mech/")

    sk = channelcollection.SK()
    sk.write_mod_file("mech/")

    if remove:
        shutil.rmtree("mech/")


def test_pickling():
    # pickle and restore
    na_ta_channel = channelcollection.Na_Ta()
    s = pickle.dumps(na_ta_channel)
    new_na_ta_channel = pickle.loads(s)

    # multiple pickles
    s = pickle.dumps(na_ta_channel)
    s = pickle.dumps(na_ta_channel)
    new_na_ta_channel = pickle.loads(s)

    assert True  # reaching this means we didn't encounter an error


def test_broadcasting():
    na_ta = channelcollection.Na_Ta()

    v = np.array([-73.234, -50.325, -25.459])
    s = np.array([0.0, 10.0, 20.0, 40.0]) * 1j

    # error must be raised if arguments are not broadcastable
    with pytest.raises(ValueError):
        na_ta.compute_lin_sum(v, s)

    # check if broadcasting rules are applied correctly for voltage and frequency
    ll = na_ta.compute_lin_sum(v[:, None], s[None, :])
    l1 = na_ta.compute_linear(v[:, None], s[None, :])
    l2 = na_ta.compute_p_open(v[:, None])

    assert ll.shape == (3, 4)
    assert l1.shape == (3, 4)
    assert l2.shape == (3, 1)
    assert np.allclose(ll, (na_ta._get_reversal(None) - v[:, None]) * l1 - l2)

    # check if broadcasting rules are applied correctly for state variables
    sv = {"m": 0.2, "h": 0.4}
    ll = na_ta.compute_lin_sum(v[:, None], s[None, :], **sv)
    assert ll.shape == (3, 4)

    sv = {"m": np.array([0.1, 0.2, 0.3]), "h": np.array([0.9, 0.6, 0.3])}
    with pytest.raises(ValueError):
        ll = na_ta.compute_lin_sum(v[:, None], s[None, :], **sv)

    sv_ = {"m": sv["m"][:, None], "h": sv["h"][:, None]}
    ll = na_ta.compute_lin_sum(v[:, None], s[None, :], **sv_)
    assert ll.shape == (3, 4)

    sv__ = {"m": sv["m"][:, None, None], "h": sv["h"][None, None, :]}
    l_ = na_ta.compute_lin_sum(v[:, None, None], s[None, :, None], **sv__)
    assert l_.shape == (3, 4, 3)
    for ii in range(4):
        assert np.allclose(
            [ll[0, ii], ll[1, ii], ll[2, ii]],
            [l_[0, ii, 0], l_[1, ii, 1], l_[2, ii, 2]],
        )

    # test braodcasting for piecewise channel
    pwc = channelcollection.PiecewiseChannel()
    varinf = pwc.compute_varinf(v)
    tauinf = pwc.compute_tauinf(v)

    assert np.allclose(varinf["a"], np.array([0.1, 0.1, 0.9]))
    assert np.allclose(varinf["b"], np.array([0.8, 0.8, 0.2]))
    assert np.allclose(tauinf["a"], np.array([10.0, 10.0, 20.0]))
    assert np.allclose(tauinf["b"], np.array([0.1, 0.1, 50.0]))


class TestDirectDependencies:
    """
    Tests for first-class direct p_open(v, c, x) dependencies on voltage and concentration.

    xfail-marked tests describe the target behaviour and are expected to fail until
    IonChannel is updated to support non-state-variable symbols in p_open.
    """

    # --- concentration in p_open ---

    def test_conc_dep_statevars(self):
        """ca declared in self.conc must not appear in statevars even when in p_open."""
        ch = channelcollection.ConcDepChan()
        assert sp.symbols("m") in ch.statevars
        assert sp.symbols("ca") not in ch.statevars
        assert len(ch.statevars) == 1

    def test_conc_dep_linearization_dc(self):
        """At DC, compute_lin_conc must match d/d_ca[(e-v)*p_ss] via finite difference."""
        ch = channelcollection.ConcDepChan()
        v0, e = -40.0, 50.0
        ca0 = ch.conc[sp.symbols("ca")]

        def p_ss(ca):
            minf = 1.0 / (1.0 + np.exp(-(v0 + 30.0) / 10.0))
            return minf / (1.0 + ca)

        dca = ca0 * 1e-5
        fd = (e - v0) * (p_ss(ca0 + dca) - p_ss(ca0 - dca)) / (2.0 * dca)
        neat_val = ch.compute_lin_conc(v0, 0.0, "ca", e=e)
        assert np.allclose(neat_val, fd, rtol=1e-4)

    def test_conc_dep_mod_file(self, tmp_path):
        """MOD file must emit exactly one 'USEION ca READ cai WRITE ica' line."""
        ch = channelcollection.ConcDepChan()
        ch.write_mod_file(str(tmp_path))
        text = (tmp_path / "IConcDepChan.mod").read_text()
        useion_lines = [l.strip() for l in text.splitlines() if "USEION ca" in l]
        assert len(useion_lines) == 1
        assert "READ cai" in useion_lines[0]
        assert "WRITE ica" in useion_lines[0]

    # --- voltage in p_open ---

    def test_volt_dep_statevars(self):
        """v in p_open must not appear in statevars (existing NEAT behaviour, must not regress)."""
        ch = channelcollection.VoltDepChan()
        assert sp.symbols("m") in ch.statevars
        assert sp.symbols("v") not in ch.statevars
        assert len(ch.statevars) == 1

    def test_volt_dep_linearization_dc(self):
        """At DC, compute_lin_sum must match d/dv[(e-v)*p_ss] via finite difference."""
        ch = channelcollection.VoltDepChan()
        v0, e = -40.0, ch.default_params["e"]

        def p_ss(v):
            minf = 1.0 / (1.0 + np.exp(-(v + 30.0) / 10.0))
            return minf / (1.0 + np.exp(-v / 10.0))

        dv = 1e-5
        fd = ((e - (v0 + dv)) * p_ss(v0 + dv) - (e - (v0 - dv)) * p_ss(v0 - dv)) / (2.0 * dv)
        neat_val = ch.compute_lin_sum(v0, 0.0, e=e)
        assert np.allclose(neat_val, fd, rtol=1e-4)

    # --- regressions for existing channels ---

    def test_regression_existing_channels(self):
        """Channels with no direct v/c in p_open must retain their current numerical outputs."""
        na = channelcollection.Na_Ta()
        assert np.allclose(na.compute_p_open(-35.0), 0.002009216860105564)
        assert np.allclose(na.compute_lin_sum(-35.0, 0.0, 50.0), -0.00534261017220376)

    def test_regression_sk_statevars(self):
        """SK channel: ca appears only in state-variable kinetics, not in p_open itself."""
        sk = channelcollection.SK()
        assert sp.symbols("ca") not in sk.statevars
        assert sp.symbols("z") in sk.statevars
        assert len(sk.statevars) == 1


def _ghk_D(v, cai, cao, temp):
    """Numerical GHK driving force matching ghk_expr."""
    KTF = (25 / 293.15) * (temp + 273.15)
    f = KTF / 2
    z = v / f
    efun = 1 - z / 2 if abs(z) < 1e-4 else z / (np.exp(z) - 1)
    return -f * (1 - (cai / cao) * np.exp(z)) * efun


class TestGHKDrivingForce:
    """
    Tests for GHK driving-force support (Step 2–5 of IonChannel_ghk_extension_TODO.md).

    Tests marked xfail describe the target behaviour and are expected to fail until
    IonChannel is updated with driving_force / conc_ext / f_driving_force support.
    """

    def test_ghk_statevars(self):
        """ca must not be in statevars; ca_ext must not appear in lambda arg lists."""
        ch = channelcollection.GHKChan()
        assert sp.symbols("m") in ch.statevars
        assert sp.symbols("ca") not in ch.statevars
        assert sp.symbols("ca_ext") not in ch.statevars
        # ca_ext is substituted numerically — it must not end up in sp_c
        assert sp.symbols("ca_ext") not in [sp.symbols(str(c)) for c in ch.sp_c]

    def test_ghk_linearization_voltage_dc(self):
        """At DC, compute_lin_sum must match -d/dv[p_ss(v)*D(v,cai0,cao0)] via FD."""
        from neat.factorydefaults import DefaultPhysiology

        ch = channelcollection.GHKChan()
        cfg = DefaultPhysiology()
        v0 = -40.0
        cai0 = ch.conc[sp.symbols("ca")]
        cao0 = cfg.conc_ext["ca"]
        temp = cfg.temp

        def p_ss(v):
            return 1.0 / (1.0 + np.exp(-(v + 30.0) / 10.0))

        dv = 1e-5
        fd = -(
            p_ss(v0 + dv) * _ghk_D(v0 + dv, cai0, cao0, temp)
            - p_ss(v0 - dv) * _ghk_D(v0 - dv, cai0, cao0, temp)
        ) / (2 * dv)
        neat_val = ch.compute_lin_sum(v0, 0.0)
        assert np.allclose(neat_val, fd, rtol=1e-4)

    def test_ghk_linearization_conc_int_dc(self):
        """At DC, compute_lin_conc('ca') must match -d/d_cai[p_ss*D(v,cai,cao0)] via FD."""
        from neat.factorydefaults import DefaultPhysiology

        ch = channelcollection.GHKChan()
        cfg = DefaultPhysiology()
        v0 = -40.0
        cai0 = ch.conc[sp.symbols("ca")]
        cao0 = cfg.conc_ext["ca"]
        temp = cfg.temp

        def p_ss(v):
            return 1.0 / (1.0 + np.exp(-(v + 30.0) / 10.0))

        dca = cai0 * 1e-5
        fd = (
            -p_ss(v0)
            * (_ghk_D(v0, cai0 + dca, cao0, temp) - _ghk_D(v0, cai0 - dca, cao0, temp))
            / (2 * dca)
        )
        neat_val = ch.compute_lin_conc(v0, 0.0, "ca")
        assert np.allclose(neat_val, fd, rtol=1e-4)

    def test_ghk_mod_file(self, tmp_path):
        """MOD must have 'USEION ca READ cai, cao WRITE ica' and no 'ca_ext' in output."""
        ch = channelcollection.GHKChan()
        ch.write_mod_file(str(tmp_path))
        text = (tmp_path / "IGHKChan.mod").read_text()
        useion_lines = [l.strip() for l in text.splitlines() if "USEION ca" in l]
        assert len(useion_lines) == 1
        assert "cai" in useion_lines[0]
        assert "cao" in useion_lines[0]
        assert "WRITE ica" in useion_lines[0]
        assert "ca_ext" not in text

    def test_ohmic_regression(self):
        """Na_Ta and SK give identical outputs regardless of GHK driving-force layer."""
        na = channelcollection.Na_Ta()
        assert np.allclose(na.compute_p_open(-35.0), 0.002009216860105564)
        assert np.allclose(na.compute_lin_sum(-35.0, 0.0, 50.0), -0.00534261017220376)

        sk = channelcollection.SK()
        assert sp.symbols("z") in sk.statevars
        assert sp.symbols("ca") not in sk.statevars


if __name__ == "__main__":
    tcns = test_channels()
    tcns.test_basic()
    test_ionchannel_simplified()
    test_broadcasting()
