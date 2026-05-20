# -*- coding: utf-8 -*-
#
# uninstall.py
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
import pathlib
import shutil

from .install import get_local_bin_dir


def _check_model_name(model_name):
    if not len(model_name) > 0:
        raise IOError(
            "No model name [name] argument was provided. " "Nothing to uninstall."
        )

    if "/" in model_name or "." in model_name:
        raise IOError(
            "Model name [name] is a path name (contains '/') or "
            "a file name (contains '.', which is not allowed."
        )


def remove_nrnspecial_wrapper(model_name):
    wrapper_name = pathlib.Path(f"python_with_{model_name}")

    local_bin_dir = get_local_bin_dir()
    wrapper_path = local_bin_dir / wrapper_name

    try:
        os.remove(wrapper_path)
        print(f"> Uninstalled {wrapper_name} from {local_bin_dir}")
    except FileNotFoundError as e:
        print(f"> {wrapper_name} not found in {local_bin_dir}, nothing to uninstall.")


def _uninstall_models(
    *model_names,
    path_neat,
    simulators=["nest", "neuron"],
):
    """
    Uninstall the model with the given name from the provided simulators

    Parameters
    ----------
    model_names: str
        the name of the model to be uninstalled
    path_neat:
        the path to the root directory of the import neat package
    simulators: list of str
        the simulators for which to uninstall the model of a given name
    """
    for model_name in model_names:
        _check_model_name(model_name)

        if "nest" in simulators:
            try:
                path_nest = os.path.join(
                    path_neat, "simulations/", f"nest/tmp/{model_name}/"
                )
                shutil.rmtree(path_nest)
                print(f"> Uninstalled {model_name} from nest")
            except FileNotFoundError as e:
                print(f"> {model_name} not found in nest, nothing to uninstall.")

        if "neuron" in simulators:
            remove_nrnspecial_wrapper(model_name)
            try:
                path_neuron = os.path.join(
                    path_neat, "simulations/", f"neuron/tmp/{model_name}/"
                )
                shutil.rmtree(path_neuron)
                print(f"> Uninstalled {model_name} from neuron")
            except FileNotFoundError as e:
                print(f"> {model_name} not found in neuron, nothing to uninstall.")
