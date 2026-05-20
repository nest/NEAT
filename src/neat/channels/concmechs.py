# -*- coding: utf-8 -*-
#
# concmechs.py
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

from ..factorydefaults import DefaultPhysiology

import sympy as sp
import numpy as np

CFG = DefaultPhysiology()


class ConcMech(object):
    def __init__(self, ion):
        self.ion = ion

    def iteritems(self):
        yield None, None

    def __repr__(self):
        return f"ConcMech(ion={self.ion})"


class ExpConcMech(ConcMech):
    def __init__(self, ion, tau=100.0, gamma=1e-15, inf=None):
        self.tau = tau  # ms
        self.gamma = gamma  # mM / (nA*ms)

        if inf is None:
            self.inf = CFG.conc[ion]
        else:
            self.inf = inf

        super().__init__(ion)

    def __str__(self):
        return f"ExpConcMech(ion={self.ion}, gamma={self.gamma}, tau={self.tau}, inf={self.inf})"

    def iteritems(self):
        yield "gamma", self.gamma
        yield "tau", self.tau
        yield "inf", self.inf

    def items(self):
        return [("gamma", self.gamma), ("tau", self.tau), ("inf", self.inf)]

    def compute_linear(self, freqs):
        return -self.gamma * 1e3 / (freqs + 1e3 / self.tau)

    def compute_lin(self, freqs):
        return -1e3 / (freqs + 1e3 / self.tau)

    def compute_lin_tau_fit(self, freqs):
        return -self.gamma * 1e3 * self.tau, 1e3

    def __repr__(self):
        return f"ExpConcMech(ion={self.ion}, gamma={self.gamma:1.6g}, tau={self.tau:1.6g}, inf={self.inf:1.6g})"

    def write_nestml_blocks(
        self, blocks=["state", "parameters", "equations", "function"], channels=[]
    ):
        ion = self.ion
        ion_channels = [chan for chan in channels if chan.ion == ion]
        read_channels = [chan for chan in channels if ion in chan.conc]
        blocks_dict = {block: "\n" for block in blocks}

        if len(read_channels) == 0:
            return blocks_dict

        if "state" in blocks:
            blocks_dict["state"] += f"        c_{ion} real = {self.inf}\n"

        if "parameters" in blocks:
            blocks_dict["parameters"] += (
                f"        gamma_{ion} real = {self.gamma}\n"
                + f"        tau_{ion} real = {self.tau} ms\n"
                + f"        inf_{ion} real = {self.inf}\n"
            )

        if "equations" in blocks:
            if len(ion_channels) == 0:
                chan_str = "0."
            else:
                chan_str = f"({' + '.join([f'i_{chan.__class__.__name__}' for chan in ion_channels])})"

            blocks_dict["equations"] += (
                f"        c_{ion}' = (inf_{ion} - c_{ion}) / (tau_{ion} * 1s) + "
                + f"gamma_{ion} * {chan_str} @mechanism::concentration\n"
            )

        return blocks_dict
