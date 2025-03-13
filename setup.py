# -*- coding: utf-8 -*-
#
# setup.py
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

import numpy
from setuptools import setup, extension
from Cython.Build import cythonize

ext = extension.Extension(
    name="netsim",
    sources=[
        "src/neat/simulations/net/netsim.pyx",
        "src/neat/simulations/net/Ionchannels.cc",
        "src/neat/simulations/net/netsim.pyx",
        "src/neat/simulations/net/NETC.cc",
        "src/neat/simulations/net/Synapses.cc",
        "src/neat/simulations/net/Tools.cc",
    ],
    language="c++",
    extra_compile_args=["-w", "-O3", "-std=gnu++11"],
    include_dirs=[numpy.get_include()],
)

s_ = setup(
    ext_package="neat",
    ext_modules=cythonize([ext], language_level=3),
)
