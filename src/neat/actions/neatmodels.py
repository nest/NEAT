# -*- coding: utf-8 -*-
#
# neatmodels.py
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
import argparse

import neat
from neat.actions.list import _list_models
from neat.actions.uninstall import _uninstall_models
from neat.actions.install import _install_models


def parse_cmd_args(path_neat):
    parser = argparse.ArgumentParser(
        description="The program compiles and installs neat ion channels models for "
        "simulators of neuronal dynamics.\n"
        "It can also list installed models and uninstall them.\n"
        "Currently, 'nest' and 'neuron' are supported as simulators"
    )

    parser.add_argument(
        "action",
        help="Whether to install a model, list all installed models or "
        "uninstall a model",
        choices=["install", "list", "uninstall"],
    )
    parser.add_argument(
        "name",
        nargs="*",
        default=[""],
        help="(i) If the [action] is 'install',"
        "specifies the name of the to be compiled model.\n"
        "If not provided, the name of the last element in [--path] will be taken.\n"
        "(ii) If the [action] is 'list', the argument is ignored.\n"
        "(iii) If the [action] is 'uninstall',"
        "specifies the name of the be removed model.\n"
        "If not provided, nothing will be removed. "
        "Should be a pure name, not a path.\n"
        "At most one argument can be provided.",
    )
    parser.add_argument(
        "-s",
        "--simulator",
        nargs="*",
        choices=["neuron", "nest"],
        default=["neuron"],
        help="The simulators to which the action is applied. \n" "Default is 'neuron'",
    )
    parser.add_argument(
        "--neuronresource",
        default=os.path.join(path_neat, "simulations/neuron/mech_storage/"),
        help="Path to directory containing additional .mod-file mechanisms "
        "(e.g. synapses) that will be compiled together with the "
        "generated ion channel mod files.\n"
        "Only used when the [action] is 'install'.",
    )
    parser.add_argument(
        "--nestresource",
        default=os.path.join(path_neat, "simulations/nest/default_syns.nestml"),
        help="Path to directory containing additional .mod-file mechanisms "
        "(e.g. synapses) that will be compiled together with the "
        "generated ion channel mod files.\n"
        "Only used when the [action] is 'install'.",
    )
    parser.add_argument(
        "--codegen_opts",
        default="",
        help="Extra options for code generation, passed as a JSON string. \n",
    )
    parser.add_argument(
        "-p",
        "--path",
        nargs="*",
        default=[""],
        help="Path where the program searches for ion channels to compile, "
        "may be one or more directories, and/or one or more python files. \n\n"
        "For each directory provided, the program will search all python "
        "files in that directory for all subclasses of `neat.IonChannel` \n\n"
        "For each python file, the program will search for all subclasses "
        "of `neat.IonChannel in that file.\n\n"
        "If nothing is provided, the program searches the current working "
        "directory. \n\n"
        "If the argument `default` is provided. The program compiles the "
        "default ion channels that are used for testing. \n\n"
        "Only used when the [action] is 'install'.",
    )
    return parser.parse_args()


def main():
    path_neat = neat.__path__[0]
    # parse the commandline args
    cmd_args = parse_cmd_args(path_neat)

    if len(cmd_args.name) > 1:
        raise IOError("At most one [name] argument can be provided")
    else:
        cmd_args.name = cmd_args.name[0]

    if cmd_args.action == "install":
        _install_models(
            cmd_args.name,
            path_neat=path_neat,
            channel_path_arg=cmd_args.path,
            simulators=cmd_args.simulator,
            path_neuronresource=cmd_args.neuronresource,
            path_nestresource=cmd_args.nestresource,
            codegen_opts=cmd_args.codegen_opts,
        )
    elif cmd_args.action == "list":
        _list_models(path_neat=path_neat, simulators=cmd_args.simulator)
    elif cmd_args.action == "uninstall":
        _uninstall_models(
            cmd_args.name, path_neat=path_neat, simulators=cmd_args.simulator
        )


if __name__ == "__main__":
    main()
