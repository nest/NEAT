# -*- coding: utf-8 -*-
#
# list.py
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
import glob


def _list_models(path_neat, simulators=["nest", "neuron", "jaxley"], pprint=True):
    """
    Get (and print) a dictionary containing all installed model

    Parameters
    ----------
    path_neat:
        the path to the root directory of the import neat package
    simulators: list of str
        the simulators for which to show the model of a given name
    """
    models = {sim: [] for sim in simulators}

    if "nest" in simulators:
        path_nest = os.path.join(path_neat, "simulations/", "nest/tmp/*/")

        for file_path in glob.glob(path_nest):
            file_name = os.path.basename(os.path.normpath(file_path))
            # only append name if directory contains .nestml files
            path_test = os.path.join(file_path, "*.nestml")
            if not len([f for f in glob.glob(path_test)]) == 0:
                models["nest"].append(file_name)

    if "neuron" in simulators:
        path_neuron = os.path.join(path_neat, "simulations/", "neuron/tmp/*/")

        for file_path in glob.glob(path_neuron):
            file_name = os.path.basename(os.path.normpath(file_path))
            models["neuron"].append(file_name)

    if "jaxley" in simulators:
        path_jaxley = os.path.join(path_neat, "simulations/", "jaxley/tmp/*")

        for file_path in glob.glob(path_jaxley):
            file_name = os.path.basename(os.path.normpath(file_path)).replace(".py", "")
            models["jaxley"].append(file_name)

    if pprint:
        print("\n------- installed models --------")
        for simulator, model_list in models.items():
            print(f"> {simulator}")
            for model in model_list:
                print(f"  - {model}")
        print("---------------------------------\n")

    return models
