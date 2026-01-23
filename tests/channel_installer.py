# -*- coding: utf-8 -*-
#
# channel_installer.py
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

try:
    import nest
    import nest.lib.hl_api_exceptions as nestexceptions
except ImportError as e:
    pass

import os
import subprocess

from neat import load_neuron_model, load_nest_model


def load_or_install_neuron_test_channels():
    """
    neatmodels install multichannel_test -s neuron -p channelcollection_for_tests.py
    """
    channel_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "channelcollection_for_tests.py")
    )
    print("check running load_or_install_neuron_test_channels")
    try:
        # load_neuron_model() calls will raise a RuntimeError is a compiled model
        # is loaded multiple times
        try:
            # raises FileNotFoundError if not compiled
            load_neuron_model("multichannel_test")
            print("loaded multichannel_test neuron model")
        except FileNotFoundError:
            subprocess.call(
                [
                    "neatmodels",
                    "install",
                    "multichannel_test",
                    "-s",
                    "neuron",
                    "-p",
                    channel_file,
                ]
            )
            load_neuron_model("multichannel_test")
    except RuntimeError as e:
        # the neuron model "multichannel_test" has already been loaded
        pass


def load_or_install_nest_test_channels():
    """
    neatmodels install multichannel_test -s nest -p channelcollection_for_tests.py
    """
    channel_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "channelcollection_for_tests.py")
    )
    try:
        # raises FileNotFoundError if not compiled
        load_nest_model("multichannel_test")
    except (nestexceptions.NESTErrors.DynamicModuleManagementError, FileNotFoundError):
        subprocess.call(
            [
                "neatmodels",
                "install",
                "multichannel_test",
                "-s",
                "nest",
                "-p",
                channel_file,
            ]
        )
        load_nest_model("multichannel_test")
