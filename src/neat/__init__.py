# -*- coding: utf-8 -*-
#
# __init__.py
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
import warnings

havedisplay = "DISPLAY" in os.environ
if not havedisplay:
    exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
    havedisplay = exitval == 0

if not havedisplay:
    import matplotlib

    matplotlib.use("Agg")

from .trees.stree import STree
from .trees.stree import SNode

from .trees.morphtree import MorphTree
from .trees.morphtree import MorphNode
from .trees.morphtree import MorphLoc

from .trees.phystree import PhysTree
from .trees.phystree import PhysNode

from .trees.sovtree import SOVTree
from .trees.sovtree import SOVNode, SomaSOVNode

from .trees.greenstree import GreensTree, GreensTreeTime
from .trees.greenstree import GreensNode, SomaGreensNode

from .trees.netree import NET
from .trees.netree import NETNode

from .trees.compartmenttree import CompartmentTree
from .trees.compartmenttree import CompartmentNode

try:
    from .simulations.neuron.neuronmodel import load_neuron_model
    from .simulations.neuron.neuronmodel import NeuronSimTree
    from .simulations.neuron.neuronmodel import NeuronSimNode
    from .simulations.neuron.neuronmodel import NeuronCompartmentTree
except ModuleNotFoundError:
    warnings.warn("NEURON not available", UserWarning)

try:
    from .simulations.nest.nestmodel import NestCompartmentTree
    from .simulations.nest.nestmodel import NestCompartmentNode
    from .simulations.nest.nestmodel import load_nest_model
except ModuleNotFoundError:
    warnings.warn("NEST not available", UserWarning)

try:
    from .simulations.jaxley.jaxleymodel import load_jaxley_model
    from .simulations.jaxley.jaxleymodel import JaxleySimTree
    from .simulations.jaxley.jaxleymodel import JaxleySimNode
    from .simulations.jaxley.jaxleymodel import JaxleyCompartmentTree
except ModuleNotFoundError:
    warnings.warn("Jaxley not available", UserWarning)

from .tools.kernelextraction import FourierQuadrature
from .tools.kernelextraction import FourierTools
from .tools.kernelextraction import Kernel

from .channels.ionchannels import IonChannel
from .channels.concmechs import ExpConcMech

from .modelreduction.compartmentfitter import CompartmentFitter
from .modelreduction.cachetrees import CachedTree
from .modelreduction.cachetrees import EquilibriumTree
from .modelreduction.cachetrees import CachedGreensTree
from .modelreduction.cachetrees import CachedGreensTreeTime
from .modelreduction.cachetrees import CachedSOVTree

from .factorydefaults import FitParams, MechParams

from .__version__ import __version__
