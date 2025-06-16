# -*- coding: utf-8 -*-
#
# channelcollection_for_tutorial.py
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
