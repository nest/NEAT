# -*- coding: utf-8 -*-
#
# channelcollection_for_tests.py
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

import sympy as sp

from neat.channels.ionchannels import IonChannel


def ghk_expr(ion: str) -> sp.Expr:
    """
    GHK driving force as a sympy expression, for use in IonChannel.driving_force.

    Symbols: v (mV), <ion> (intracellular mM), <ion>_ext (extracellular mM),
    temp (degC). The _ext and temp symbols are substituted numerically by
    IonChannel._substitute_defaults before lambdifying.
    """
    v, temp = sp.symbols("v temp")
    ci = sp.symbols(ion)
    co = sp.symbols(ion + "_ext")
    KTF = (25 / 293.15) * (temp + 273.15)
    f = KTF / 2
    z = v / f
    efun = sp.Piecewise((1 - z / 2, sp.Abs(z) < 1e-4), (z / (sp.exp(z) - 1), True))
    return -f * (1 - (ci / co) * sp.exp(z)) * efun


def vtrap(x, y):
    """
    Function to stabelize limit cases of 0/0 that occur in certain channel
    model rate equations
    """
    return sp.Piecewise(
        (y - x / 2.0, sp.Abs(x / y) < 1e-6), (x / (sp.exp(x / y) - 1), True)
    )


class test_channel(IonChannel):
    """
    Simple channel to test basic functionality
    """

    def define(self):
        # open probability
        self.p_open = "5 * a00**3 * a01**3 * a02 + a10**2 * a11**2 * a12"

        # state variables activation
        self.varinf = {}
        self.varinf["a00"] = "1. / (1. + exp((v-30.) / 100.))"
        self.varinf["a01"] = "1. / (1. + exp((-v+30.)/ 100.))"
        self.varinf["a02"] = "-10."
        self.varinf["a10"] = "2. / (1. + exp((v-30.) / 100.))"
        self.varinf["a11"] = "2. / (1. + exp((-v+30.)/ 100.))"
        self.varinf["a12"] = "-30."
        # state variable time scales
        self.tauinf = {}
        self.tauinf["a00"] = "1."
        self.tauinf["a01"] = "2."
        self.tauinf["a02"] = "1."
        self.tauinf["a10"] = "2."
        self.tauinf["a11"] = "2."
        self.tauinf["a12"] = "3."
        # default reversal
        self.e = -23.0


class test_channel2(IonChannel):
    """
    Simple channel to test basic functionality
    """

    def define(self):
        # open probability
        self.p_open = ".9 * a00**3 * a01**2 + .1 * a10**2 * a11"
        # state variables
        self.varinf = {"a00": ".3", "a01": ".5", "a10": ".4", "a11": ".6"}
        self.tauinf = {"a00": "1.", "a01": "2.", "a10": "2.", "a11": "2."}
        # default reversal
        self.e = -23.0


class PiecewiseChannel(IonChannel):
    def define(self):
        # open probability
        self.p_open = "a+b"
        # state variables
        v = sp.symbols("v")
        self.varinf = {
            "a": sp.Piecewise((0.1, v < -50.0), (0.9, True)),
            "b": sp.Piecewise((0.8, v < -50.0), (0.2, True)),
        }
        self.tauinf = {
            "a": sp.Piecewise((10.0, v < -50.0), (20.0, True)),
            "b": sp.Piecewise((0.1, v < -50.0), (50.0, True)),
        }
        # default reversal
        self.e = -28.0


class Na_Ta(IonChannel):
    def define(self):
        """
        (Colbert and Pan, 2002)

        Used in (Hay, 2011)
        """
        self.ion = "na"
        # concentrations the ion channel depends on
        self.conc = {}
        # define channel open probability
        self.p_open = "h * m ** 3"
        # define activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "0.182 * (v + 38.) / (1. - exp(-(v + 38.) / 6.))"  # 1/ms
        self.beta["m"] = "-0.124 * (v + 38.) / (1. - exp( (v + 38.) / 6.))"  # 1/ms
        self.alpha["h"] = "-0.015 * (v + 66.) / (1. - exp( (v + 66.) / 6.))"  # 1/ms
        self.beta["h"] = "0.015 * (v + 66.) / (1. - exp(-(v + 66.) / 6.))"  # 1/ms
        # temperature factor for time-scale
        self.q10 = 2.95


class Kv3_1(IonChannel):
    def define(self):
        self.ion = "k"
        # define channel open probability
        self.p_open = "m"
        # define activation functions
        self.varinf = {"m": "1. / (1. + exp(-(v - 18.70) /  9.70))"}
        self.tauinf = {"m": "4. / (1. + exp(-(v + 46.56) / 44.14))"}  # ms


class SK(IonChannel):
    def define(self):
        """
        SK-type calcium-activated potassium current (Kohler et al., 1996)
        used in (Hay et al., 2011)
        """
        self.ion = "k"
        self.conc = ["ca"]
        # define channel open probability
        self.p_open = "z"
        # activation functions
        self.varinf = {"z": "1. / (1. + (0.00043 / ca)**4.8)"}
        self.tauinf = {"z": "1."}  # ms


class h(IonChannel):
    def define(self):
        """
        Hcn channel from (Bal and Oertel, 2000)
        """
        # define channel open probability
        self.p_open = ".8 * hf + .2 * hs"
        # define activation functions
        self.varinf, self.tauinf = {}, {}
        self.varinf["hf"] = "1. / (1. + exp((v + 82.) / 7.))"
        self.varinf["hs"] = "1. / (1. + exp((v + 82.) / 7.))"
        self.tauinf["hf"] = "40."
        self.tauinf["hs"] = "300."
        # default reversal
        self.e = -43.0


class NaTa_t(IonChannel):
    """
    Colbert and Pan 2002
    """

    def define(self):
        self.ion = "na"
        # concentrations the ion channel depends on
        self.conc = {}
        # define channel open probability
        self.p_open = "m ** 3 * h"
        # define activation functions
        (
            self.alpha,
            self.beta,
        ) = (
            {},
            {},
        )
        # NOTE : in the mod file they trapped the case where denominator is 0
        v = sp.symbols("v")
        self.alpha["m"] = 0.182 * vtrap(-(v + 38.0), 6.0)
        self.beta["m"] = 0.124 * vtrap((v + 38.0), 6.0)
        self.alpha["h"] = 0.015 * vtrap(v + 66.0, 6.0)
        self.beta["h"] = 0.015 * vtrap(-(v + 66.0), 6.0)

        # temperature factor for time-scale
        self.q10 = "2.3**((34-21)/10)"


class SKv3_1(IonChannel):
    """
    Characterization of a Shaw-related potassium channel family in rat brain,
    The EMBO Journal, vol.11, no.7,2473-2486 (1992)
    """

    def define(self):
        self.ion = "k"
        self.p_open = "z"
        # activation functions
        self.varinf = {"z": " 1/(1+exp(((v -(18.700))/(-9.700))))"}
        self.tauinf = {"z": "0.2*20.000/(1+exp(((v -(-46.560))/(-44.140))))"}


class Ca_HVA(IonChannel):
    """
    Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993
    """

    def define(self):
        self.ion = "ca"
        # concentrations the ion channel depends on
        self.conc = {}
        # define channel open probability
        self.p_open = "m ** 2 * h "
        # define activation functions
        (
            self.alpha,
            self.beta,
        ) = (
            {},
            {},
        )
        self.alpha["m"] = "(0.055*(-27-v))/(exp((-27-v)/3.8) - 1)"
        self.beta["m"] = "(0.94*exp((-75.-v)/17))"
        self.alpha["h"] = "(0.000457*exp((-13-v)/50))"
        self.beta["h"] = "(0.0065/(exp((-v-15)/28)+1))"


class Ca_LVAst(IonChannel):
    """
    Comment: LVA ca channel. Note: mtau is an approximation from the plots
    Reference: Avery and Johnston 1996, tau from Randall 1997
    Comment: shifted by -10 mv to correct for junction potential
    Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21
    """

    # NOTE: this is not exactly equal to the BBP due to the vshirft
    def define(self):
        self.ion = "ca"
        # concentrations the ion channel depends on
        self.conc = {}
        # define channel open probability
        self.p_open = "m ** 2 * h"
        # define activation functions
        (
            self.tauinf,
            self.varinf,
        ) = (
            {},
            {},
        )
        self.varinf["m"] = "1.0000/(1+ exp((v + 40.000)/-6))"
        self.tauinf["m"] = "5.0000 + 20.0000/(1+exp((v + 35.000)/5))"
        self.varinf["h"] = "1.0000/(1+ exp((v + 90.000)/6.4))"
        self.tauinf["h"] = "20.0000 + 50.0000/(1+exp((v + 50.000)/7))"
        # temperature dependence
        self.q10 = "2.3**((34-21)/10)"


class SK_E2(IonChannel):
    """
    SK-type calcium-activated potassium current (Kohler et al., 1996)
    used in (Hay et al., 2011)
    """

    def define(self):
        self.ion = "k"
        self.conc = ["ca"]
        self.p_open = "z"
        # activation functions
        # self.varinf = {'z': '1. / (1. + (0.00043 / ca)**4.8)'}
        ca = sp.symbols("ca")
        self.varinf = {
            "z": sp.Piecewise(
                (1.0 / (1.0 + (0.00043 / ca) ** 4.8), ca > 1e-7),
                (1.0 / (1.0 + (0.00043 / (ca + 1e-7)) ** 4.8), True),
            )
        }
        self.tauinf = {"z": "1."}  # ms


class ConcDepChan(IonChannel):
    """
    Toy channel with a concentration appearing directly in p_open (not only via state-variable
    kinetics). p_open = m / (1 + ca), where ca is intracellular calcium.

    Used to test first-class support for direct concentration dependencies in p_open.
    """

    def define(self):
        self.ion = "ca"
        self.conc = ["ca"]
        self.p_open = "m / (1 + ca)"
        self.varinf = {"m": "1 / (1 + exp(-(v + 30) / 10))"}
        self.tauinf = {"m": "1."}
        self.e = 50.0


class VoltDepChan(IonChannel):
    """
    Toy channel with voltage appearing directly in p_open (not only through state-variable
    kinetics). p_open = m / (1 + exp(-v / 10)).

    Used to test first-class support for direct voltage dependencies in p_open.
    """

    def define(self):
        self.p_open = "m / (1 + exp(-v / 10))"
        self.varinf = {"m": "1 / (1 + exp(-(v + 30) / 10))"}
        self.tauinf = {"m": "1."}
        self.e = -23.0


class GHKChan(IonChannel):
    """Toy GHK channel: p_open = m, driving force = GHK(ca)."""

    def define(self):
        self.ion = "ca"
        self.conc = ["ca"]
        self.conc_ext = ["ca"]
        self.p_open = "m"
        self.driving_force = ghk_expr("ca")
        self.varinf = {"m": "1 / (1 + exp(-(v + 30) / 10))"}
        self.tauinf = {"m": "1."}
