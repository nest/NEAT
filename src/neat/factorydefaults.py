# -*- coding: utf-8 -*-
#
# factorydefaults.py
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
import numpy.typing as npt

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DefaultPhysiology:
    # default concentrations (same as NEURON)
    conc: Dict[str, float] = field(
        default_factory=lambda: {"na": 10.0, "k": 54.4, "ca": 1e-4}  # mM
    )
    # default extracellular concentrations (same as NEURON)
    conc_ext: Dict[str, float] = field(
        default_factory=lambda: {"ca": 2.0, "na": 140.0, "k": 5.0}  # mM
    )
    # default ion channel reversals
    e_rev: Dict[str, float] = field(
        default_factory=lambda: {"na": 50.0, "k": -85.0, "ca": 50.0}  # mV
    )
    # default temperature
    temp: float = 36.0  # celcius


@dataclass
class FitParams:
    # holding potentials channel fit
    e_hs: np.array = field(
        default_factory=lambda: np.array([-75.0, -55.0, -35.0, -15.0])
    )
    # holding concentrations channel fit
    conc_hs: Dict[str, np.array] = field(
        default_factory=lambda: {"ca": np.array([0.00010, 0.00012, 0.00014, 0.00016])}
    )

    # holding potentials concmech fit
    e_hs_cm: np.array = field(
        default_factory=lambda: np.array(
            [
                -78.22,
                -68.22,
                -58.22,
            ]
        )
    )
    # holding concentrations concmech fit
    conc_hs_cm: Dict[str, np.array] = field(
        default_factory=lambda: {"ca": np.array([0.000100, 0.000105, 0.000110])}
    )

    # eps parameter for comptree
    fit_comptree_eps: float = 1e-2

    # frequency evaluation
    freqs: float = 0.0  # Hz
    freqs_tau: np.array = field(
        default_factory=lambda: np.logspace(0.0, 3.0, 100) * 1j  # Hz
    )

    # # time-points at which to evaluate time domain kernels for capacitance fit
    # t_fit: np.array = field(
    #     default_factory=lambda: np.array([
    #         11., 12., 13., 14., 15.,
    #         16., 17., 18., 19., 20.,
    #         22., 24., 26., 28. ,30.,
    #     ]) # ms
    # )
    # time-points at which to evaluate time domain kernels for capacitance fit
    t_fit: np.array = field(
        default_factory=lambda: np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                22.0,
                24.0,
                26.0,
                28.0,
                30.0,
            ]
        )  # ms
    )
    # # time-points at which to evaluate time domain kernels for capacitance fit
    # t_fit: np.array = field(
    #     default_factory=lambda: np.array([
    #     11., 12., 13., 14., 15.,
    #     16., 17., 18., 19., 20.,
    #     22., 24., 26., 28. ,30.,
    #     32., 34., 36., 38., 40.,
    #     50., 60., 70., 80.
    #     ]) # ms
    # )
    # # time-points at which to evaluate time domain kernels for capacitance fit
    # t_fit: np.array = field(
    #     default_factory=lambda: np.array([
    #     11., 12., 13., 14., 15.,
    #     16., 17., 18., 19., 20.,
    #     ]) # ms
    # )


@dataclass
class MechParams:
    # concentration mechanism default parameters
    exp_conc_mech: Dict[str, float] = field(
        default_factory=lambda: {"gamma": 0.0, "tau": 100.0}  # [tau] = ms
    )
