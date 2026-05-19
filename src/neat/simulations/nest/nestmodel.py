# -*- coding: utf-8 -*-
#
# nestmodel.py
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

from ...trees.compartmenttree import CompartmentTree, CompartmentNode
from ...factorydefaults import DefaultPhysiology

import numpy as np

import warnings
import subprocess

CFG = DefaultPhysiology()


try:
    import nest
except ModuleNotFoundError:
    warnings.warn(
        "NEST not available, importing non-functional nest module only for doc generation",
        UserWarning,
    )

    # universal iterable mock object
    class N(object):
        def __init__(self):
            pass

        def __getattr__(self, attr):
            try:
                return super(N, self).__getattr__(attr)
            except AttributeError:
                return self.__global_handler

        def __global_handler(self, *args, **kwargs):
            return N()

        def __iter__(self):  # make iterable
            return self

        def __next__(self):
            raise StopIteration

        def __mul__(self, other):  # make multipliable
            return 1.0

        def __rmul__(self, other):
            return self * other

        def __call__(self, *args, **kwargs):  # make callable
            return N()

    nest = N()
    np_array = np.array

    def array(*args, **kwargs):
        if isinstance(args[0], N):
            print(args)
            print(kwargs)
            return np.eye(2)
        else:
            return np_array(*args, **kwargs)

    np.array = array


def load_nest_model(name):
    # Currently, we have no way of checking whether the *.so-file
    # associated with the model is in {nest build directory}/lib/nest,
    # so instead we check whether the model is installed in NEAT
    output = str(
        subprocess.check_output(
            [
                "neatmodels",
                "list",
                "multichannel_test",
                "-s",
                "nest",
            ]
        ),
        "utf-8",
    )
    if name not in output:
        raise FileNotFoundError(f"The NEST model '{name}' is not installed")
    nest.Install(name + "_module")


class NestCompartmentNode(CompartmentNode):
    """
    Node class that summarizes the parameters of `neat.CompartmentNode` into
    a parameter dictionary suitable for the NEST model.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def _make_compartment_dict(self, channel_storage=None):

        # channel parameters
        g_dict = {
            f"gbar_{key}": self.currents[key][0] for key in self.currents if key != "L"
        }
        e_dict = {
            f"e_{key}": self.currents[key][1] for key in self.currents if key != "L"
        }

        # concentration mech parameters
        c_dict = {}
        for ion, concmech in self.concmechs.items():
            c_dict.update(
                {
                    f"gamma_{ion}": concmech.gamma,
                    f"tau_{ion}": concmech.tau,
                    f"inf_{ion}": concmech.inf,
                }
            )

        # passive parameters
        p_dict = {
            "g_L": self.currents["L"][0],
            "e_L": self.currents["L"][1],
            "C_m": self.ca * 1e3,  # convert uF to nF
            "g_C": self.g_c,
        }

        # initialization parameter
        i_dict = {"v_comp": self.e_eq}
        for key, (g, e) in self.currents.items():
            if key == "L":
                continue
            channel = channel_storage[key]

            # append asymptotic state variables to initialization dictionary
            svs = channel.compute_varinf(i_dict["v_comp"])
            for sv, val in svs.items():
                i_dict[f"{sv}_{key}"] = val

        # create full compartment initialization dictionary
        p_dict.update(g_dict)
        p_dict.update(e_dict)
        p_dict.update(c_dict)
        p_dict.update(i_dict)

        if self.parent_node is None:
            parent_idx = -1
        else:
            parent_idx = self.parent_node.index

        return {"parent_idx": parent_idx, "params": p_dict}


class NestCompartmentTree(CompartmentTree):
    def __init__(self, arg=None):
        super().__init__(arg)

    def create_corresponding_node(self, index, **kwargs):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
        node_index: int
            index of the new node
        **kwargs
            Parameters of the parent class node initialization function
        """
        return NestCompartmentNode(index, **kwargs)

    def _make_compartments_params_list(self):
        # ensure that the all node indices are equal to the position where
        # they appear in the iteration, so that they also correspond to the NEST
        # model indices
        self.reset_indices()

        return [
            node._make_compartment_dict(channel_storage=self.channel_storage)
            for node in self
        ]

    def init_model(self, model_name, n, suffix="_model", v_th=-20.0, **kwargs):
        """
        Initialize n nest instantiations of the current model.

        Parameters
        ----------
        model_name: str
            name of the model, corresponds to the name given when the nest model
            was installed with `neatmodels install`
        n: int (> 0)
            The number of copies of the model to be instantiated
        v_th: float
            The spike detection threshold, in mV
        **kwargs
            Keyword arguments to the `nest.Create()` function
        """
        models = nest.Create(model_name + suffix, n, **kwargs)
        models.V_th = v_th
        models.compartments = self._make_compartments_params_list()

        return models
