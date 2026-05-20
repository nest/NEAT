# -*- coding: utf-8 -*-
#
# nestml_tools.py
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
import copy
import warnings

PERMITTED_BLOCK_NAMES = [
    "parameters",
    "state",
    "equations",
    "function",
    "internals",
    "input",
    "output",
]


def strip_v_comp(contents):
    ss = "v_comp real"
    idx = None
    for ll, line in enumerate(contents):
        if ss in line:
            idx = ll

    if idx is not None:
        del contents[idx]


def strip_comments(contents):
    for ll, line in enumerate(contents):
        try:
            s0 = line.index("#")
            contents[ll] = line[:s0] + "\n"
        except ValueError as e:
            # do nothing if there is no comment
            pass


def _get_index_of_block(contents, block_name):
    found, kk = False, 0
    while not found and kk < len(contents):
        if block_name in contents[kk]:
            found = True
        else:
            kk += 1

    return kk


def get_block_string(contents, block_name):
    try:
        c0 = _get_index_of_block(contents, block_name + ":")
        s0 = contents[c0].index(block_name) + len(block_name + ":")

        c1 = min(
            [
                _get_index_of_block(contents[c0 + 1 :], block_name + ":") + c0
                for block_name in PERMITTED_BLOCK_NAMES
            ]
        )

        block_str = contents[c0][s0:] + "".join(contents[c0 + 1 : c1]) + "\n"
    except IndexError as e:
        warnings.warn("'%s' block not found in .nestml file")
        block_str = ""

    return block_str


def get_functions_string(contents):
    function_str = ""

    try:
        kk = 0
        while kk < len(contents):
            c0 = _get_index_of_block(contents[kk:], "function") + kk
            s0 = contents[c0].index("function")

            c1 = min(
                [
                    _get_index_of_block(contents[c0 + 1 :], block_name) + c0
                    for block_name in PERMITTED_BLOCK_NAMES
                ]
                + [_get_index_of_block(contents[c0 + 1 :], "function") + c0]
            )

            function_str += (
                "\n    " + contents[c0][s0:] + "".join(contents[c0 + 1 : c1]) + "\n"
            )

            # move further, functions are assumed to at least occupy one line each
            kk = c1 + 1

    except (ValueError, IndexError) as e:
        # we reached the end of the NESTML file and don't have anymore functions
        # to check
        pass

    return function_str


def parse_nestml_file(f_name):
    with open(f_name, "r") as file:
        contents = file.readlines()

    strip_comments(contents)

    blocks = dict(zip(PERMITTED_BLOCK_NAMES, [""] * len(PERMITTED_BLOCK_NAMES)))
    for block_name in blocks:
        if block_name != "output" and block_name != "function":
            blocks[block_name] += get_block_string(contents, block_name)
        elif block_name == "function":
            blocks[block_name] += get_functions_string(contents)

    return blocks


def write_nestml_blocks(
    blocks,
    path_name,
    neuron_name,
    v_comp=0.0,
    write_blocks=["parameters", "state", "equations", "input", "output", "functions"],
):
    for block, blockstr in blocks.items():
        if block != "function" and block != "output":
            blocks[block] = f"\n    {block}:\n{blockstr}"
            if block == "state" and not "v_comp" in blockstr:
                blocks[block] += f"\n        v_comp real = {v_comp}"
            blocks[block] += "\n\n"

        elif block == "output":
            blocks[block] = "\n    output:\n        spike\n\n"

    fname = os.path.join(path_name, neuron_name + ".nestml")

    file = open(fname, "w")
    file.write("\nmodel %s:\n" % neuron_name)
    for blockstr in blocks.values():
        file.write(blockstr)
    file.close()

    return fname
