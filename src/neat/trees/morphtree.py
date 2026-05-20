# -*- coding: utf-8 -*-
#
# morphtree.py
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

import matplotlib.patheffects as patheffects
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import copy
import pathlib
import warnings
from typing import Literal
from functools import reduce
from contextlib import contextmanager

from .stree import SNode, STree
from .compartmenttree import CompartmentNode, CompartmentTree
from ..tools.interval import interval


def computational_tree_decorator(fun):
    """
    Decorator that provides the safety that the active tree is the
    computational one inside the functions it decorates. This decorator also
    checks if a computational tree has been defined.

    Raises
    ------
        AttributeError
            If this function is called and no computational tree has been
            defined
    """

    # wrapper to access self
    def wrapped(self, *args, **kwargs):
        if self._computational_root is None:
            raise AttributeError(
                "No computational tree has been defined, "
                "and this function requires one. Use "
                "`MorphTree.set_comp_tree()` or its "
                "overwritten version in one of the derived"
                "classes"
            )
        with self.as_computational_tree:
            res = fun(self, *args, **kwargs)
        return res

    wrapped.__doc__ = fun.__doc__
    return wrapped


def original_tree_decorator(fun):
    """
    Decorator that provides the safety that active tree is the
    original one inside the functions it decorates.
    """

    # wrapper to access self
    def wrapped(self, *args, **kwargs):
        with self.as_original_tree:
            res = fun(self, *args, **kwargs)
        return res

    wrapped.__doc__ = fun.__doc__
    return wrapped


class MorphLoc(object):
    """
    Stores a location on the morphology. The location is initialized starting
    from a node and x-value on the real morphology. The location is also be
    stored in the coordinates of the computational morphology. To toggle between
    coordinates, the class stores a reference to the morphology tree on which
    the location is defined, and returns either the original coordinate or the
    coordinate on the computational tree, depending on which tree is active.

    Initialized based on either a tuple or a dict where one entry specifies the
    node index and the other entry the x-coordinate specifying the location
    between parent node (x=0) or the node indicated by the index (x=1), or on
    a `neat.MorphLoc`.

    Parameters
    ----------
        loc: tuple or dict or `neat.MorphLoc`
            if tuple: (node index, x-value)
            if dict: {'node': node index, 'x': x-value}
        reftree: `neat.MorphTree`
        set_as_comploc: bool
            if True, assumes the paremeters provided in `loc` are coordinates
            on the computational tree. Doing this while no computational tree
            has been initialized in `reftree` will result in an error.
            Defaults to False

    Raises
    ------
        ValueError
            If x-coordinate of location is not in ``[0,1]``
    """

    def __init__(self, loc, reftree, set_as_comploc=False):
        self.reftree = reftree

        if isinstance(loc, tuple):
            x = float(loc[1])
            if x > 1.0 or x < 0.0:
                raise ValueError("x-value should be in [0,1]")
            if set_as_comploc:
                self.comp_loc = {"node": int(loc[0]), "x": x}
                self._set_original_loc()
            else:
                self.loc = {"node": int(loc[0]), "x": x}
        elif isinstance(loc, dict):
            x = float(loc["x"])
            if x > 1.0 or x < 0.0:
                raise ValueError("x-value should be in [0,1]")
            if set_as_comploc:
                self.comp_loc = loc
                self._set_original_loc()
            else:
                self.loc = loc
        elif isinstance(loc, MorphLoc):
            self.loc = loc.loc
            self.reftree = reftree
        else:
            raise TypeError("Not a valid location type, should be tuple or dict")

    def __getitem__(self, key):
        if isinstance(key, int) and key in (0, 1):
            key = "node" if key == 0 else "x"
        if isinstance(key, str):
            if self.reftree.check_computational_tree_active():
                try:
                    return self.comp_loc[key]
                except AttributeError:
                    self._set_computational_loc()
                    return self.comp_loc[key]
            else:
                return self.loc[key]

    def __eq__(self, other_loc):
        loc1 = self.loc
        loc2 = MorphLoc(
            other_loc,
            self.reftree,
            set_as_comploc=self.reftree.check_computational_tree_active(),
        )

        # covering all posible combinations
        if loc1["node"] != 1:
            if loc2["node"] != 1:
                if (loc1["x"] < 1e-8) and ((1.0 - loc2["x"]) < 1e-8):
                    node = self.reftree[loc2["node"]]
                    parent = self.reftree[loc1["node"]].parent_node
                    result = node.index == parent.index

                elif (loc2["x"] < 1e-8) and ((1.0 - loc1["x"]) < 1e-8):
                    node = self.reftree[loc1["node"]]
                    parent = self.reftree[loc2["node"]].parent_node
                    result = node.index == parent.index

                else:
                    result = loc1["node"] == loc2["node"]
                    result *= np.allclose(loc1["x"], loc2["x"])

            else:
                if loc1["x"] < 1e-8:
                    node = self.reftree[loc1["node"]]
                    result = node.parent_node.index == 1

                else:
                    result = False

        else:
            if loc2["node"] != 1:
                if loc2["x"] < 1e-8:
                    node = self.reftree[loc2["node"]]
                    result = node.parent_node.index == 1

                else:
                    result = False

            else:
                result = True

        return result

    def keys(self):
        return ["node", "x"]

    def __iter__(self):
        yield self["node"]
        yield self["x"]

    def __neq__(self, other_loc):
        result = self.__eq__(other_loc)
        if result is NotImplemented:
            return result
        else:
            return not result

    def __copy__(self):
        """
        Customization of the copy function so that `loc` and `comp_loc`
        attributes are deep copied and `reftree` attribute still refers to the
        original tree
        """
        new_loc = type(self)(copy.deepcopy(self.loc), self.reftree)
        if hasattr(self, "comp_loc"):
            new_loc.__dict__.update({"comp_loc": copy.deepcopy(self.comp_loc)})
        return new_loc

    def __str__(self):
        return "{'node': %d, 'x': %.2f }" % (self.loc["node"], self.loc["x"])

    def __repr__(self):
        return str(self)

    def _set_computational_loc(self):
        if self.loc["node"] != 1:
            with self.reftree.as_original_tree:
                node = self.reftree[self.loc["node"]]
                # find the computational nodes that are resp. up and down from the node
                node_start = self.reftree._find_comp_node_to_root(node.parent_node)
                node_stop = self.reftree._find_comp_node_from_root(node)
                # length between loc and parent computational node to compute segment
                # length
                L = self.reftree.path_length(
                    {"node": node_start.index, "x": 1.0}, self.loc
                )
            # get the computational nodes' length
            with self.reftree.as_computational_tree:
                L_cn = self.reftree[node_stop.index].L
            # set the computational loc
            self.comp_loc = {"node": node_stop.index, "x": L / L_cn}
        else:
            self.comp_loc = copy.deepcopy(self.loc)

    def _set_original_loc(self):
        if self.comp_loc["node"] != 1:
            with self.reftree.as_computational_tree:
                compnode = self.reftree[self.comp_loc["node"]]
            with self.reftree.as_original_tree:
                node = self.reftree[self.comp_loc["node"]]
                # find the computational node that is down from the original node
                pcnode = self.reftree._find_comp_node_to_root(node.parent_node)
                # find the node index and x-coordinate of the original location
                path = self.reftree.path_between_nodes(pcnode, node)
                L0 = 0.0
                found = False
                for pathnode in path[1:]:
                    L1 = L0 + pathnode.L
                    Lloc = self.comp_loc["x"] * compnode.L
                    if Lloc == 0.0:
                        Lloc += 1e-7
                    if Lloc > L0 and Lloc <= L1:
                        self.loc = {
                            "node": pathnode.index,
                            "x": (Lloc - L0 - 1e-8) / pathnode.L,
                        }
                    L0 = L1
                if self.loc["x"] > 1.0 or self.loc["x"] < 0.0:
                    raise ValueError("x-value should be in [0,1]")
        else:
            self.loc = copy.deepcopy(self.comp_loc)


class MorphNode(SNode):
    """
    Node associated with `neat.MorphTree`. Stores the geometrical information
    associated with a point on the tree morphology

    Attributes
    ----------
        xyz: numpy.array of floats
            The xyz-coordinates associated with the node (um)
        R: float
            The radius of the node (um)
        swc_type: int
            The type of node, according to the .swc file format convention:
            ``1`` is dendrites, ``2`` is axon, ``3`` is basal dendrite and ``4``
            is apical dendrite.
        L: float
            The length of the node (um)
    """

    def __init__(self, index, p3d=None):
        super().__init__(index)
        if p3d != None:
            self.set_p3d(*p3d)
        else:
            # bogus values, to overwrite
            self.set_p3d(np.array([0.0, 0.0, 0.0]), 1.0, 1)
            self.L = 1.0
            self.R = 1.0

    def set_p3d(self, xyz, R, swc_type):
        """
        Set the 3d parameters of the node

        Parameters
        ----------
            xyz: `np.array`
                3D location (um)
            R: float
                Radius of the segment (um)
            swc_type: int
                Type asscoiated with the segment according to SWC standards
        """
        # morphology parameters
        self.xyz = xyz
        self.R = R
        self.swc_type = swc_type
        # auxiliary variable
        self.used_in_comp_tree = False

    def set_length(self, L):
        """
        Set the length of the segment represented by the node

        Parameters
        ----------
            L: float
                the length of the segment (um)
        """
        self.L = L

    def set_radius(self, R):
        """
        Set the radius of the segment represented by the node

        Parameters
        ----------
            L: float
                the radius of the segment (um)
        """
        self.R = R

    def get_child_nodes(self, skip_inds=(2, 3)):
        """
        Get the `child_nodes` of this node. Indices ``2`` and ``3`` are skipped
        by default (3-point soma convention)

        Parameters
        ----------
        skip_inds: list or tuple of ints
            Node indices of child nodes that are not added to the returned list

        Returns
        -------
        list of `neat.MorphNode`
            The child nodes
        """
        return [cnode for cnode in self._child_nodes if cnode.index not in skip_inds]

    def set_child_nodes(self, cnodes):
        return super().set_child_nodes(cnodes)

    child_nodes = property(get_child_nodes, set_child_nodes)

    def __str__(self, **kwargs):
        node_str = super().__str__(**kwargs)
        node_str += (
            f" --- xyz = [{self.xyz[0]:.3f}, {self.xyz[1]:.3f}, {self.xyz[2]:.3f}] um, "
            f"R = {self.R:.2f} um, swc_type = {self.swc_type}"
        )
        return node_str

    def _get_repr_dict(self):
        repr_dict = super()._get_repr_dict()
        repr_dict.update(
            {
                "xyz": self.xyz,
                "R": f"{self.R:1.6g}",
                "swc_type": self.swc_type,
            }
        )
        return repr_dict

    def __repr__(self):
        return repr(self._get_repr_dict())


class MorphTree(STree):
    """
    Subclass of simple tree that implements neuronal morphologies. Reads in
    trees from '.swc' files (http://neuromorpho.org/).

    Neural morphologies are assumed to follow the three-point soma convention.
    Internally however, the soma is represented as a sphere. Hence nodes with
    indices 2 and 3 do not represent anything and are skipped in iterations and
    getters.

    The `MorphTree` can also store a simplified version of the original tree,
    where only nodes are retained that should hold computational parameters,
    i.e. the root, the bifurcation nodes and the leafs at least, although the
    user can also specify additional nodes. By default, the active tree is the
    original one, where nodes correspond to the underlying '.swc' file. Trough the
    `MorphTree.as_computational_tree` context, the computational tree can be
    accessed.

    For computational efficiency, it is possible to store sets of locations on
    the morphology, under user-specified names. These sets are stored as
    lists of `neat.MorphLoc`, and associated arrays are stored that contain the
    corresponding node indices of the locations, their x-coordinates, their
    distances to the soma and their distances to the nearest bifurcation in the
    in the direction of the soma.

    Parameters
    ----------
    arg: `str` (optional), `neat.MorphNode` or subclass, or `neat.MorphTree` or subclass, or ``None``
        - If `str`, represents the file path of the morphology file.
        Assumed to follow the '.swc' format.
        - If `neat.MorphNode` or derived class, initializes a the tree with the
        provided node as root.
        - If `neat.MorphTree` or derived class, initialized a copy of the provided
        tree.
        - Default is ``None``, which initialized an empty tree
    types: list of int (optional)
        The list of node types to be included. As per the '.swc' convention,
        ``1`` is soma, ``2`` is axon, ``3`` is basal dendrite and ``4`` apical
        dendrite. Default is ``[1,3,4]``.

    Attributes
    ----------
        root: `neat.MorphNode` instance
            The root of the tree.
        locs: dict {str: list of `neat.MorphLoc`}
            Stored sets of locations, key is the user-specified the name of the
            set of locations. Initialized as empty dict.
        nids: dict {str: np.array of int}
            Node indices of locations. Initialized as empty dict.
        xs: dict {str: np.array of float}
            x-coordinates of locations. Initialized as empty dict.
        d2s: dict {str: np.array of float}
            distances to soma of locations. Initialized as empty dict.
        d2b: dict {str: np.array of float}
            distances to nearest bifurcation in the direction of the soma
            of locations. Initialized as empty dict.
    """

    def __init__(self, arg=None, types=[1, 3, 4]):
        # we initialize two root nodes, one for the original tree mimicking the
        # .swc file, and one for the coarse grained tree for computational efficiency
        self._computational_root = None
        self._original_root = None
        # to store sets of locations on the morphology
        self.locs = {}
        self._nids_orig = {}
        self._nids_comp = {}
        self._xs_orig = {}
        self._xs_comp = {}
        self.d2s = {}
        self.d2b = {}
        self.leafinds = {}

        # instantiate the tree structure
        try:
            swc_file_path = pathlib.Path(arg)
            self.read_swc_tree_from_file(swc_file_path, types=types)
        except TypeError:
            super().__init__(arg)
            # STree will always initialize `self._root`, independent of whether the
            # input argument is a node or a tree.
            # When the input argument is a node, it will *only* initialize `self._root`.
            # When the input argument is a tree, it will call `self.__copy__`, which
            # ensures that both the `self._original_root`` and `self._computational_root``
            # (when the comptree is already set, see `self.set_comp_tree()`) are initialized.
            # Therefor, the following code ensures that `self._original_root` is always set
            # to the correct value.
            if not self.check_computational_tree_active():
                self._original_root = self._root

    def __getitem__(self, index, skip_inds=(2, 3)):
        """
        Returns the node with given index, if no such node is in the tree, None
        is returned.

        Parameters
        ----------
            index: int
                the index of the node to be found

        Returns:
            `neat.MorphNode` or None
        """
        return self._find_node(self.root, index, skip_inds=skip_inds)

    def _find_node(self, node, index, skip_inds=(2, 3)):
        """
        Breadth-first/stack iteration to replace the recursive call.
        Traverses the tree until it finds the node you are looking for.
        Returns SNode when found and None when not found

        Parameters
        ----------
            node: :class:`SNode` (optional)
                node where the search is started
            index: int
                the index of the node to be found

        Returns
        -------
            :class:`SNode`
        """
        stack = []
        stack.append(node)
        while len(stack) != 0:
            for cnode in stack:
                if cnode.index == index:
                    return cnode
                else:
                    stack.remove(cnode)
                    stack.extend(cnode.get_child_nodes(skip_inds=skip_inds))
        return None  # Not found!

    def __iter__(self, node=None, skip_inds=(2, 3)):
        """
        Overloaded iterator from parent class that avoids iterating over the
        nodes with index 2 and 3

        Parameters
        ----------
            node: `neat.MorphNode`
                The starting node. Defaults to the root
            skip_inds: tuple of ints
                Indices of the nodes that are skipped by the iterator. Defaults
                to ``(2,3)``, the nodes that contain extra geometrical
                information on the soma.

        Yields
        ------
            `neat.MorphNode`
                Nodes in the tree
        """
        if node is None:
            node = self.root
        if node is not None:
            if node.index not in skip_inds:
                yield node
            for cnode in node.get_child_nodes(skip_inds=skip_inds):
                for inode in self.__iter__(cnode, skip_inds=skip_inds):
                    if node.index not in skip_inds:
                        yield inode

    def reset_indices(self):
        """
        Resets the indices in the order they appear in a depth-first iteration
        """
        for ind, node in enumerate(self):
            node.index = ind + 1

    def get_nodes(self, skip_inds=(2, 3)):
        """
        Overloads the parent function to allow skipping nodes with certain
        indices and to return the nodes associated with the corresponding
        active tree.

        Parameters
        ----------
            skip_inds: tuple of ints
                Indices of the nodes that are skipped by the iterator. Defaults
                to ``(2,3)``, the nodes that contain extra geometrical
                information on the soma.

        Returns
        -------
            list of `neat.MorphNode`
        """
        nodes = []
        self._gather_nodes(self.root, nodes, skip_inds=skip_inds)
        return nodes

    def set_nodes(self, illegal):
        raise AttributeError(
            f"Property 'nodes' of '{self.__class__.__name__}' object has no setter."
            "Add nodes to the tree with `tree.add_node_with_parent(node)`"
        )

    nodes = property(get_nodes, set_nodes)

    def _gather_nodes(self, node, node_list=[], skip_inds=(2, 3)):
        """
        Overloaded gathering function that avoids appending nodes with index 2
        or 3 to the list.

        Parameters
        ----------
            node: `neat.MorphNode`
            node_list: list of `neat.MorphNode`
        """
        if node.index not in skip_inds:
            node_list.append(node)
        for cnode in node.get_child_nodes(skip_inds=skip_inds):
            self._gather_nodes(cnode, node_list=node_list, skip_inds=skip_inds)

    def get_leafs(self):
        """
        Overloads the `get_leafs` of the parent class to return the leaf nodes
        in the corresponding active tree.
        """
        return [node for node in self if self.is_leaf(node)]

    leafs = property(get_leafs)

    def get_nodes_in_basal_subtree(self):
        """
        Return the nodes associated with the basal subtree

        Returns
        -------
            list of `neat.MorphNode`
                List of all nodes in the basal subtree
        """
        return [node for node in self if node.swc_type in [3]]

    basal_nodes = property(get_nodes_in_basal_subtree)

    def get_nodes_in_apical_subtree(self):
        """
        Return the nodes associated with the apical subtree

        Returns
        -------
            list of `neat.MorphNode`
                List of all nodes in the apical subtree
        """
        return [node for node in self if node.swc_type in [4]]

    apical_nodes = property(get_nodes_in_apical_subtree)

    def get_nodes_in_axonal_subtree(self):
        """
        Return the nodes associated with the apical subtree

        Returns
        -------
            list of `neat.MorphNode`
                List of all nodes in the apical subtree
        """
        return [node for node in self if node.swc_type in [2]]

    def _check_computational_root(self):
        if self._computational_root is None:
            raise AttributeError(
                "No computational tree has been defined. Use "
                "`MorphTree.set_comp_tree()` or its overwritten "
                "version in one of the derived classes"
            )

    def set_default_tree(
        self, default: Literal["original", "computational"] = "original"
    ):
        """
        Set either the 'original' or 'computatianal' tree as the default one.
        Note that this function should be used outside
        `as_computational_tree()` and `as_original_tree()` context managers,
        otherwise the default will not be stored.

        Parameters
        ----------
        default : Literal[&#39;original&#39;, &#39;computational&#39;], optional
            The tree to set as default, by default 'original'.

        Raises
        ------
        ValueError
            For incorrect argument
        """
        if default == "original":
            self.root = self._original_root
        elif default == "computational":
            self._check_computational_root()
            self.root = self._computational_root
        else:
            raise ValueError("Default tree can be 'original' or 'computational'.")

    @property
    @contextmanager
    def as_computational_tree(self):
        """
        Context manager that ensures the computational tree is active
        """
        self._check_computational_root()
        treetype = (
            "computational" if self.check_computational_tree_active() else "original"
        )
        self.root = self._computational_root
        try:
            yield self
        except Exception as e:
            raise
        finally:
            self.root = (
                self._computational_root
                if treetype == "computational"
                else self._original_root
            )

    @property
    @contextmanager
    def as_original_tree(self):
        """
        Context manager that ensures the original tree is active
        """
        treetype = (
            "computational" if self.check_computational_tree_active() else "original"
        )
        self.root = self._original_root
        try:
            yield self
        except Exception as e:
            raise
        finally:
            self.root = (
                self._computational_root
                if treetype == "computational"
                else self._original_root
            )

    def check_computational_tree_active(self):
        """
        check whether the computational tree is active

        Returns
        -------
        `bool`:
            ``True`` if the computational tree is active
        """
        return (
            self.root is self._computational_root
            and self._computational_root is not None
        )

    def create_corresponding_node(self, node_index, p3d=None):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return MorphNode(node_index, p3d=p3d)

    def read_swc_tree_from_file(self, file_n, types=[1, 3, 4]):
        """
        Non-specific for a "tree data structure"
        Read and load a morphology from an SWC file and parse it into
        an `neat.MorphTree` object.

        On the NeuroMorpho.org website, 5 types of somadescriptions are
        considered (http://neuromorpho.org/neuroMorpho/SomaFormat.html).
        The "3-point soma" is the standard and most files are converted
        to this format during a curation step. `neat` follows this default
        specification and the *internal structure of `neat` implements
        the 3-point soma*. Additionally multi-cylinder descriptions with more
        than three nodes are also supported, but are converted to the standard
        three point description.

        Additionally, the root node of the tree must have ``index == 1``,
        ``swc_type == 1`` and occur first in the SWC file.

        Parameters
        -----------
        file_n: str
            name of the file to open
        types: list of ints
            NeuroMorpho.org segment types to be loaded

        Examples
        --------
        The three point description is

        .. code-block:: python

             1 1 x y   z r -1
             1 1 x y-r z r 1
             1 1 x y+r z r 1

        with `x,y,z` the coordinates of the soma center and `r` the soma radius

        This is a valid three point desciption

        .. code-block:: python

            # start of file
            1 1 45.3625 18.6775 -50.25 10.1267403895 -1
            2 1 45.3625 8.55075961052 -50.25 10.1267403895 1
            3 1 45.3625 28.8042403895 -50.25 10.1267403895 1
            # dendrite nodes
            4 3 37.76 12.99 -46.08 0.29 1
            5 3 26.7068019951 8.26344199599 -36.9426896493 0.795614809475 4
            # ...

        This is a valid multi-cylinder descirption

        .. code-block:: python

            # start of file
            1 1 1066.38 399.67 157.0 4.9215 -1
            2 1 1071.3 399.67 157.0 4.9215 1
            3 1 1076.22 399.67 157.0 4.9215 2
            4 1 1066.5 402.83 157.0 11.494 2
            5 1 1062.4 405.5 157.0 15.308 4
            6 1 1056.6 410.25 158.0 20.536 5
            7 1 1056.6 410.25 158.0 20.536 6
            8 1 1070.0 427.75 161.0 2.305 7
            # dendrite nodes
            9 3 1070.0 427.75 161.0 0.886 8
            # ...

        Raises
        ------
        ValueError
            If the SWC file is not consistent with the aforementioned conventions

        """
        # check soma-representation: 3-point soma or a non-standard representation
        soma_type = self.determine_soma_type(file_n)

        file = open(file_n, "r")
        all_nodes = dict()
        for line in file:
            if not line.startswith("#"):
                split = line.split()
                index = int(split[0].rstrip())
                swc_type = int(split[1].rstrip())
                x = float(split[2].rstrip())
                y = float(split[3].rstrip())
                z = float(split[4].rstrip())
                radius = float(split[5].rstrip())
                parent_index = int(split[6].rstrip())
                # create the nodes
                if swc_type in types:
                    p3d = (np.array([x, y, z]), radius, swc_type)
                    node = self.create_corresponding_node(index, p3d)
                    all_nodes[index] = (swc_type, node, parent_index)

        # check if node with index 1 is soma node (swc_type == 1)
        if all_nodes[1][0] != 1:
            raise ValueError(
                "Node with index 1 should be soma-type, i.e. swc_type == 1"
            )

        # one point soma representation
        if soma_type == 0:
            for index, (swc_type, node, parent_index) in list(all_nodes.items()):
                if index == 1:
                    self.root = node
                else:
                    parent_node = all_nodes[parent_index][1]
                    self.add_node_with_parent(node, parent_node)

            # we add nodes 2 and 3 to adhere to obtain a 3-point soma representation
            # (http://neuromorpho.org/neuroMorpho/SomaFormat.html)
            # and increment the indices of all nodes that are not the soma by 2
            for node in self.__iter__(skip_inds=[]):
                if self.is_root(node):
                    # create p3ds for the extra soma nodes
                    xyz_2 = copy.copy(node.xyz)
                    xyz_2[1] -= node.R
                    p3d_2 = (xyz_2, node.R, node.swc_type)
                    xyz_3 = copy.copy(node.xyz)
                    xyz_3[1] += node.R
                    p3d_3 = (xyz_3, node.R, node.swc_type)
                else:
                    node.index += 2
            # add extra soma nodes to the tree
            snode_2 = self.create_corresponding_node(2, p3d_2)
            snode_3 = self.create_corresponding_node(3, p3d_3)
            self.add_node_with_parent(snode_2, self[1])
            self.add_node_with_parent(snode_3, self[1])

        # three point soma representation
        if soma_type == 1:
            for index, (swc_type, node, parent_index) in list(all_nodes.items()):
                if index == 1:
                    self.root = node
                elif index in (2, 3):
                    # the 3-point soma representation
                    # (https://neuromorpho.org/SomaFormat.html)
                    somanode = all_nodes[1][1]
                    self.add_node_with_parent(node, somanode)
                else:
                    parent_node = all_nodes[parent_index][1]
                    self.add_node_with_parent(node, parent_node)

            # check if soma follows three point convention
            radius_arr = np.array(
                [
                    all_nodes[1][1].R,
                    all_nodes[2][1].R,
                    all_nodes[3][1].R,
                    np.linalg.norm(all_nodes[2][1].xyz - all_nodes[1][1].xyz),
                    np.linalg.norm(all_nodes[3][1].xyz - all_nodes[1][1].xyz),
                ]
            )
            if not np.allclose(
                np.abs(radius_arr - radius_arr[0]), np.zeros_like(radius_arr), atol=2e-2
            ):
                raise ValueError(
                    "Soma radii not consistent with three-point convention"
                )

        # IF multiple cylinder soma representation
        elif soma_type == 2:
            self.root = all_nodes[1][1]

            # get all soma info
            soma_cylinders = []
            connected_to_root = []
            for index, (swc_type, node, parent_index) in list(all_nodes.items()):
                if swc_type == 1 and not index == 1:
                    soma_cylinders.append((node, parent_index))
                    if index > 1:
                        connected_to_root.append(index)

            # make soma
            s_node_2, s_node_3 = self._make_soma_from_cylinders(
                soma_cylinders, all_nodes
            )

            # add soma
            self.root.R = s_node_2.R
            self.add_node_with_parent(s_node_2, self.root)
            self.add_node_with_parent(s_node_3, self.root)

            # add the other points
            for index, (swc_type, node, parent_index) in list(all_nodes.items()):
                if swc_type == 1:
                    pass
                else:
                    parent_node = all_nodes[parent_index][1]
                    if parent_node.index in connected_to_root:
                        self.add_node_with_parent(node, self.root)
                    else:
                        self.add_node_with_parent(node, parent_node)

        # set the lengths of the nodes
        for node in self:
            if node.parent_node != None:
                L = np.sqrt(np.sum((node.parent_node.xyz - node.xyz) ** 2))
                # if the length is zero we can just delete it
                if L == 0:
                    self.remove_single_node(node)
            else:
                L = 0.0
            node.set_length(L)

        # store the created root as the root of the original tree
        self._original_root = self.root

        return self

    def _make_soma_from_cylinders(self, soma_cylinders, all_nodes):
        """
        Construct 3-point soma
        Step 1: calculate surface of all cylinders
        Step 2: make 3-point representation with the same surface
        """
        total_surf = 0
        xyz_sum = self.root.xyz
        for node, parent_index in soma_cylinders:

            parent = all_nodes[parent_index][1]

            nxyz = node.xyz
            pxyz = parent.xyz

            H = np.sqrt(np.sum((nxyz - pxyz) ** 2))

            surf = 2 * np.pi * parent.R * H
            total_surf += surf

            xyz_sum += node.xyz

        # define apropriate radius
        radius = np.sqrt(total_surf / (4.0 * np.pi))
        rp = xyz_sum / (len(soma_cylinders) + 1.0)
        rp2 = np.array([rp[0], rp[1] - radius, rp[2]])
        rp3 = np.array([rp[0], rp[1] + radius, rp[2]])

        self.root.xyz = rp
        # create the soma nodes
        s_node_2 = self.create_corresponding_node(2, (rp2, radius, 1))
        s_node_3 = self.create_corresponding_node(3, (rp3, radius, 1))

        return s_node_2, s_node_3

    def determine_soma_type(self, file_n):
        """
        Determine the soma type used in the SWC file.
        This method searches the whole file for soma entries.

        Only tbe standard three-point soma type and a multi-cylinder description
        are supported.

        Furthermore, the root node of the tree must have ``index == 1``,
        ``swc_type == 1`` and occur first in the SWC file.

        Parameters
        ----------
        file_n: string
            Name of the file containing the SWC description

        Returns
        -------
        soma_type: int
            Integer indicating one of the su[pported SWC soma formats.
            0: One point soma
            1: Default three-point soma,
            2: multiple cylinder description

        Raises
        ------
        ValueError
            If soma type is not supported (less than three nodes have soma)
        """
        file = open(file_n, "r")
        somas = 0
        for line in file:
            if not line.startswith("#"):
                split = line.split()
                index = int(split[0].rstrip())
                s_type = int(split[1].rstrip())
                if s_type == 1:
                    somas = somas + 1
        file.close()
        if somas == 1:
            return 0
        if somas == 3:
            return 1
        elif somas < 3:
            raise ValueError(
                "Soma description not supported, use 3-point or multi-cylinder description"
            )
        else:
            return 2

    def _evaluate_comp_criteria(self, node, eps=1e-8, rbool=False):
        """
        Return ``True`` if relative difference between node radius and parent
        node raidus is larger than margin ``eps``, or if the node is the root
        or bifurcation node.

        Parameters
        ----------
        node: `neat.MorphNode`
            node that is compared to parent node
        eps: float (optional, default ``1e-8``)
            the margin

        return
        ------
        bool
        """
        if not rbool:
            rbool = node.parent_node == None
        if not rbool:
            rbool = len(node.get_child_nodes()) != 1
        if not rbool:
            cnode = node.child_nodes[0]
            rbool = np.abs(node.R - cnode.R) > eps * np.max([node.R, cnode.R])

        return rbool

    def set_comp_tree(self, compnodes=None, eps=1e-8):
        """
        Sets the nodes that contain computational parameters. This are a priori
        either bifurcations, leafs, the root or nodes where the neurons'
        relevant parameters change.

        Parameters
        ----------
            compnodes: list of ::class::`MorphNode`
                list of nodes that should be retained in the computational tree.
                Note that specifying bifurcations, leafs or the root is
                superfluous, since they are part of the computational tree by
                default.
            eps: float (default ``1e-8``)
                relative margin for parameter change
        """
        self.remove_comp_tree()
        if compnodes is None:
            compnodes = []
        compnodes += [
            node for node in self if self._evaluate_comp_criteria(node, eps=eps)
        ]
        compnode_indices = [node.index for node in compnodes]
        nodes = copy.deepcopy(self.nodes)

        for node in nodes:
            if node.index not in compnode_indices:
                self.remove_single_node(node)
            elif node.parent_node != None:
                orig_node = self[node.index]
                orig_bnode = node.parent_node
                L, R = self.path_length(
                    {"node": orig_bnode.index, "x": 1.0},
                    {"node": orig_node.index, "x": 1.0},
                    compute_radius=1,
                )
                node.set_length(L)
                node.set_radius(R)
                node.used_in_comp_tree = True
                orig_node.used_in_comp_tree = True
            else:
                orig_node = self[node.index]
                node.used_in_comp_tree = True
                orig_node.used_in_comp_tree = True

        self._computational_root = next(node for node in nodes if node.index == 1)
        self._leafs_comp = [node for node in nodes if self.is_leaf(node)]
        self._nodes_comp = []
        self._gather_nodes(self._computational_root, self._nodes_comp)

        # create conversion of all coordinate arrays
        for name in self.locs:
            self._store_comp_locs(name)

    def _find_comp_node_to_root(self, node):
        """
        !!! Computational tree has to be initialized, otherwise may results in
        error !!!

        If the input node is a node of the original tree, finds the first node
        on the path to the root that has an equivalent in the computational tree.
        If the input node has such an equivalent, it is returned itself.

        If the input node is in the computational tree, returns the node itself.

        Parameters
        ----------
            node: `neat.MorphNode` instance
                the input node

        Returns
        -------
            `neat.MorphNode` instance
        """
        if not node.used_in_comp_tree:
            node = self._find_comp_node_to_root(node.parent_node)
        return node

    def _find_comp_node_from_root(self, node):
        """
        !!! Computational tree has to be initialized, otherwise may results in
        error !!!

        If the input node is a node of the original tree, finds the first node
        away from the root that has an equivalent in the computational tree. If
        the input node has such an equivalent, it is returned itself.

        If the input node is in the computational tree, returns the node itself.

        Parameters
        ----------
            node: `neat.MorphNode` instance
                the input node

        Returns
        -------
            `neat.MorphNode` instance
        """
        if not node.used_in_comp_tree:
            node = self._find_comp_node_from_root(node.child_nodes[0])
        return node

    def remove_comp_tree(self):
        """
        Removes the computational tree
        """
        self._computational_root = None
        for node in self:
            node.used_in_comp_tree = False

    def convert_loc_arg_to_locs(self, loc_arg):
        """
        Converts locations argument to list of `neat.MorphLoc`.

        Parameters
        ----------
        loc_arg: list of dictionaries, tuples or `neat.MorphLoc`, or string
            * If list, entries should be valid arguments to initialize a `neat.MorphLoc`
            * If string, should be the name of a list of locations stored in `self`

        Returns
        -------
        list of `neat.MorphLoc`
            List of locations, each referencing the current tree
        """
        if isinstance(loc_arg, list):
            locs = [MorphLoc(loc, self) for loc in loc_arg]
        elif isinstance(loc_arg, str):
            self._try_name(loc_arg)
            locs = self.get_locs(loc_arg)
        else:
            raise IOError("`loc_arg` should be list of locs or string")
        return locs

    def convert_node_arg_to_nodes(self, node_arg):
        """
        Converts a node argument to a list of nodes. Behaviour depends on the
        type of argument.

        Parameters
        ----------
        node_arg: ``None``, `neat.MorphNode`, {'apical', 'basal', 'axonal', 'somatic'} or iterable collection of instances of `neat.MorphNode`
            * `None`: returns all nodes
            * `neat.MorphNode`: returns list of nodes in the subtree of the given node
            * {'apical', 'basal', 'axonal'}: returns list of nodes in the apical, basal or axonal subtree
            * iterable collection of `neat.MorphNode`: returns the same list of nodes
                If an iterable collection of original nodes is given, and the computational
                tree is active, a reduced list is returned where only the corresponding
                computational nodes are included. If an iterable collection of
                computational nodes is given, and the original tree is active, a list of
                corresponding original nodes is given, but the in between nodes are not
                added.


        Returns
        -------
        list of `neat.MorphNode`
        """
        # convert the input argument to a list of nodes
        nodes = []
        if node_arg == None:
            nodes.extend(self.nodes)

        elif isinstance(node_arg, MorphNode):
            if self.check_computational_tree_active():
                # assure that a list of computational nodes is returned
                node_arg = self._find_comp_node_from_root(node_arg)
                node_arg = self[node_arg.index]
            else:
                # assure that a list of original nodes is returned
                node_arg = self[node_arg.index]
            nodes.extend(self.gather_nodes(node_arg))

        elif hasattr(node_arg, "__len__") and len(node_arg) > 0:
            # if node arg is iterable, it may a string, a list of strings
            # or a list of nodes
            check_all_morphnode = True
            for node in node_arg:
                if not isinstance(node, MorphNode):
                    check_all_morphnode = False

            if check_all_morphnode:
                for node in node_arg:
                    if self.check_computational_tree_active():
                        # assure that a list of computational nodes is returned
                        node_ = self._find_comp_node_from_root(node)
                        compnode = self[node_.index]
                        if compnode not in nodes:
                            nodes.append(compnode)
                    else:
                        # assure that a list of original nodes is returned
                        nodes.append(self[node.index])

            else:
                if not isinstance(node_arg, str):
                    for entry in node_arg:
                        if entry not in ["apical", "basal", "axonal", "somatic"]:
                            raise IOError(
                                "input should be (i) `None`, (ii) an instance of "
                                "`neat.MorphNode`, (iii) one of the following 4 strings "
                                "'somatic', 'apical', 'basal' or 'axonal' or (iv) an iterable "
                                "collection of instances of :class:MorphNode"
                            )
                else:
                    if node_arg not in ["apical", "basal", "axonal", "somatic"]:
                        raise IOError(
                            "input should be (i) `None`, (ii) an instance of "
                            "`neat.MorphNode`, (iii) one of the following 3 strings "
                            "'somatic', 'apical', 'basal' or 'axonal' or (iv) an iterable "
                            "collection of instances of :class:MorphNode"
                        )

                if "apical" in node_arg:
                    nodes.extend(self.get_nodes_in_apical_subtree())
                if "basal" in node_arg:
                    nodes.extend(self.get_nodes_in_basal_subtree())
                if "axonal" in node_arg:
                    nodes.extend(self.get_nodes_in_axonal_subtree())
                if "somatic" in node_arg:
                    nodes.extend([self[1]])

        else:
            raise IOError(
                "input should be (i) `None`, (ii) an instance of "
                "`neat.MorphNode`, (iii) one of the following 3 strings "
                "'apical', 'basal' or 'axonal' or (iv) an iterable "
                "collection of instances of :class:MorphNode"
            )

        return nodes

    def path_length(self, loc1, loc2, compute_radius=0):
        """
        Find the length of the direct path between loc1 and loc2

        Parameters
        ----------
            loc1: dict, tuple or `neat.MorphLoc`
                one location
            loc2: dict, tuple or `neat.MorphLoc`
                other location
            compute_radius: bool
                if True, also computes the average weighted radius of the path

        Returns
        -------
        L, R (optional)
            L: float
                length of path, in micron
            R: float
                weighted average radius of path, in micron
        """
        # define location objects
        if type(loc1) == dict or type(loc1) == tuple:
            loc1 = MorphLoc(loc1, self)
        if type(loc2) == dict or type(loc2) == tuple:
            loc2 = MorphLoc(loc2, self)

        # start path length calculation
        if loc1["node"] == loc2["node"]:
            node = self[loc1["node"]]
            if node.index == 1:
                L = 0.0  # soma is spherical and has no lenght
            else:
                L = node.L * np.abs(loc1["x"] - loc2["x"])
            if compute_radius:
                R = node.R
        else:
            node1 = self[loc1["node"]]
            node2 = self[loc2["node"]]
            path1 = self.path_to_root(node1)[::-1]
            path2 = self.path_to_root(node2)[::-1]
            path = path1 if len(path1) < len(path2) else path2
            ind = next(
                (ii for ii in range(len(path)) if path1[ii] != path2[ii]), len(path)
            )
            if path1[ind - 1] == node1:
                L = node1.L * (1.0 - loc1["x"])
                L += sum(node.L for node in path2[ind:-1])
                L += node2.L * loc2["x"]
                if compute_radius:
                    R = node1.R * node1.L * (1.0 - loc1["x"])
                    R += sum(node.R * node.L for node in path2[ind:-1])
                    R += node2.R * node2.L * loc2["x"]
                    R /= L
            elif path2[ind - 1] == node2:
                L = node1.L * loc1["x"]
                L += sum(node.L for node in path1[ind:-1])
                L += node2.L * (1.0 - loc2["x"])
                if compute_radius:
                    R = node1.R * node1.L * loc1["x"]
                    R += sum(node.R * node.L for node in path2[ind:-1])
                    R += node2.R * node2.L * (1.0 - loc2["x"])
                    R /= L
            else:
                L = node1.L * loc1["x"]
                L += sum(node.L for node in path1[ind:-1])
                L += sum(node.L for node in path2[ind:-1])
                L += node2.L * loc2["x"]
                if compute_radius:
                    R = node1.R * node1.L * loc1["x"]
                    R += sum(node.R * node.L for node in path1[ind:-1])
                    R += sum(node.R * node.L for node in path2[ind:-1])
                    R += node2.R * node2.L * loc2["x"]
                    R /= L
        if compute_radius:
            return L, R
        else:
            return L

    @original_tree_decorator
    def store_locs(self, locs, name, warn=True):
        """
        Store locations under a specified name

        Parameters
        ----------
            locs: list of dicts, tuples or `neat.MorphLoc`
                the locations to be stored
            name: string
                name under which these locations are stored
            warn: bool (default ``True``)
                raise a `UserWarning` if two or more locations in `locs` refer
                to the soma. Choose ``False`` if this is desired to remove
                the warning.
        """

        # copy list and store in MorphLoc if necessary
        locs_ = []
        n1 = 0
        for loc in locs:
            locs_.append(MorphLoc(loc, self))
            if locs_[-1]["node"] == 1:
                n1 += 1
        if n1 > 1 and warn:
            warnings.warn(
                "There are multiple locations on the soma in this set "
                + "locations, this can cause issues in certain functions",
                UserWarning,
            )

        self.remove_locs(name)

        self.locs[name] = locs_
        self._nids_orig[name] = np.array([loc["node"] for loc in locs_])
        self._xs_orig[name] = np.array([loc["x"] for loc in locs_])

        if self._computational_root != None:
            self._store_comp_locs(name)

    @computational_tree_decorator
    def _store_comp_locs(self, name):
        self._nids_comp[name] = np.array([loc["node"] for loc in self.locs[name]])
        self._xs_comp[name] = np.array([loc["x"] for loc in self.locs[name]])

    def add_loc(self, loc, name):
        """
        Add location to set of locations of given name

        Parameters
        ----------
        loc: dict, tuple or `neat.MorphLoc`
            the location to be added
        name: str
            the name of the set of locations to which the location is added
        """
        loc = MorphLoc(loc, self)
        self.locs[name].append(loc)
        self._nids_orig[name] = np.concatenate((self._nids_orig[name], [loc["node"]]))
        self._xs_orig[name] = np.concatenate((self._xs_orig[name], [loc["x"]]))
        if self._computational_root != None:
            self._add_comp_loc(loc, name)

    @computational_tree_decorator
    def _add_comp_loc(self, loc, name):
        self._nids_comp[name] = np.concatenate((self._nids_comp[name], [loc["node"]]))
        self._xs_comp[name] = np.concatenate((self._xs_comp[name], [loc["x"]]))

    def clear_locs(self):
        """
        Remove all set of locs stored in the tree
        """
        self.locs = {}
        self._nids_orig = {}
        self._nids_comp = {}
        self._xs_orig = {}
        self._xs_comp = {}
        self.d2s = {}
        self.d2b = {}
        self.leafinds = {}

    def remove_locs(self, name):
        """
        Remove a set of locations of a given name

        Parameters
        ----------
            name: string
                name under which the desired list of locations is stored
        """
        try:
            del self.locs[name]
            del self._nids_orig[name]
            del self._nids_comp[name]
            del self._xs_orig[name]
            del self._xs_comp[name]
        except KeyError:
            pass
        # warnings.warn('Locations of name %s were not defined'%name)
        try:
            del self.d2s[name]
        except KeyError:
            pass
        try:
            del self.d2b[name]
        except KeyError:
            pass
        try:
            del self.leafinds[name]
        except KeyError:
            pass

    def _try_name(self, name):
        """
        Tests if the name is in use. Raises a KeyError when it is not in use and
        prints a list of possible names

        Parameters
        ----------
            name: string
                name of the desired list of locations

        Raises
        ------
            KeyError
                If 'name' does not refer to a set of locations in use
        """
        try:
            self.locs[name]
        except KeyError as err:
            err.args = (
                "'"
                + err.args[0]
                + "' name not in use. Possible names are "
                + str(list(self.locs.keys())),
            )
            raise

    def get_locs(self, name):
        """
        Returns a set of locations of a specified name

        Parameters
        ----------
            name: string
                name under which the desired list of locations is stored

        Returns
        -------
            list of `neat.MorphLoc`
        """
        self._try_name(name)
        return self.locs[name]

    def get_node_indices(self, name):
        """
        Returns an array of node indices of locations of a specified name

        Parameters
        ----------
            name: string
                name under which the desired list of locations is stored

        Returns
        -------
            numpy.array of ints
        """
        self._try_name(name)
        return self.nids[name]

    def get_node_idxs(self):
        if self.check_computational_tree_active():
            return self._nids_comp
        else:
            return self._nids_orig

    def set_node_idxs(self, nids):
        if self.check_computational_tree_active():
            self._nids_comp = nids
        else:
            self._nids_orig = nids

    nids = property(get_node_idxs, set_node_idxs)

    def get_x_coords(self, name):
        """
        Returns an array of x-values of locations of a specified name

        Parameters
        ----------
            name: string
                name under which the desired list of locations is stored
        """
        self._try_name(name)
        return self.xs[name]

    def get_xs(self):
        if self.check_computational_tree_active():
            return self._xs_comp
        else:
            return self._xs_orig

    def set_xs(self, xs):
        if self.check_computational_tree_active():
            self._xs_comp = xs
        else:
            self._xs_orig = xs

    xs = property(get_xs, set_xs)

    def get_loc_idxs_on_node(self, name, node):
        """
        Returns a list of the indices of locations in the list of a given name
        that are on a the input node, ordered for increasing x

        Parameters
        ----------
            name: string
                which list of locations to consider
            node: `neat.MorphNode`
                the node to consider. Should be part of the original
                tree
        Returns
        -------
            list of ints
                indices of locations on the path
        """
        self._try_name(name)
        nids = self.nids[name]
        xs = self.xs[name]
        # get the locinds on the node
        inds = np.where(nids == node.index)[0]
        sortinds = np.argsort(xs[inds])

        return inds[sortinds].tolist()

    def get_loc_idxs_on_nodes(self, name, node_arg):
        """
        Returns a list of the indices of locations in the list of a given name
        that are on one of the nodes specified in the node list. Within each
        node, locations are ordered for increasing x

        Parameters
        ----------
            name: string
                which list of locations to consider
            node_arg:
                see documentation of `MorphTree.convert_node_arg_to_nodes`
        Returns
        -------
            list of ints
                indices of locations on the path
        """
        # find locinds on all nodes
        locinds = []
        for node in self.convert_node_arg_to_nodes(node_arg):
            locinds.extend(self.get_loc_idxs_on_node(name, node))

        return locinds

    def get_loc_idxs_on_path(self, name, node0, node1, xstart=0.0, xstop=1.0):
        """
        Returns a list of the indices of locations in the list of a given name
        that are on the given path. The path is taken to start at the input
        x-start coordinate of the first node in the list and to stop at the
        given x-stop coordinate of the last node in the list

        Parameters
        ----------
            name: string
                which list of locations to consider
            node0: :class:`SNode`
                start node of path
            node1: :class:`SNode`
                stop node of path
            xstart: float (in ``[0,1]``)
                starting coordinate on `node0`
            xstop: float (in ``[0,1]``)
                stopping coordinate on `node1`

        Returns
        -------
            list of ints
                Indices of locations on the path. If path is empty, an empty
                array is returned.
        """
        self._try_name(name)
        locs = self.locs[name]
        xs = self.xs[name]
        # find the path
        path = self.path_between_nodes(node0, node1)
        # find the location indices
        locinds = []
        if len(path) > 1:
            # first node in path
            node = path[0]
            ninds = np.array(self.get_loc_idxs_on_node(name, node)).astype(int)
            if node.parent_node == None:
                locinds.extend(ninds)
            else:
                if node.parent_node == path[1]:
                    # goes runs towards root
                    inds = np.where(xs[ninds] <= xstart)[0]
                    sortinds = np.argsort(xs[ninds][inds])[::-1]
                else:
                    # path goes away from root
                    inds = np.where(xs[ninds] >= xstart)[0]
                    sortinds = np.argsort(xs[ninds][inds])
                locinds.extend(ninds[inds][sortinds])
            # middle nodes in path
            for ii, node in enumerate(path[1:-1]):
                ninds = np.array(self.get_loc_idxs_on_node(name, node)).astype(int)
                if node.parent_node == None:
                    locinds.extend(ninds)
                elif path[ii + 2] == node.parent_node:
                    # path goes towards root
                    sortinds = np.argsort(xs[ninds])
                    locinds.extend(ninds[sortinds[::-1]])
                elif path[ii] == node.parent_node:
                    # path goes away from root
                    sortinds = np.argsort(xs[ninds])
                    locinds.extend(ninds[sortinds])
                else:
                    # turning point (path only goes on this node at x=1)
                    inds = np.where((1.0 - xs[ninds]) < 1e-4)[0]
                    if len(inds) > 0:
                        locinds.extend(ninds[inds])
            # last node in path
            node = path[-1]
            ninds = np.array(self.get_loc_idxs_on_node(name, node)).astype(int)
            if node.parent_node == None:
                locinds.extend(ninds)
            else:
                if node.parent_node == path[-2]:
                    # path goes away from root
                    inds = np.where(xs[ninds] <= xstop)[0]
                    sortinds = np.argsort(xs[ninds][inds])
                else:
                    # path goes towards root
                    inds = np.where(xs[ninds] >= xstop)[0]
                    sortinds = np.argsort(xs[ninds][inds])[::-1]
                locinds.extend(ninds[inds][sortinds])
        elif len(path) == 1:
            node = path[0]
            ninds = np.array(self.get_loc_idxs_on_node(name, node)).astype(int)
            if node.parent_node == None:
                locinds.extend(ninds)
            else:
                if xstart < xstop:
                    inds = np.where(
                        np.logical_and(xs[ninds] >= xstart, xs[ninds] <= xstop)
                    )[0]
                    sortinds = np.argsort(xs[ninds][inds])
                else:
                    inds = np.where(
                        np.logical_and(xs[ninds] >= xstop, xs[ninds] <= xstart)
                    )[0]
                    sortinds = np.argsort(xs[ninds][inds])[::-1]
                locinds.extend(ninds[inds][sortinds])

        return locinds

    def get_nearest_loc_idxs(
        self, locs, name, direction=0, check_siblings=True, pprint=False
    ):
        """
        For each location in the input location list, find the index of the
        closest location in a set of locations stored under a given name. The
        search can go in the either go in the up or down direction or in both
        directions.

        Parameters
        ----------
            locs: list of dicts, tuples or `neat.MorphLoc`
                the locations for which the nearest location index has to be
                found
            name: string
                name under which the reference list is stored
            direction: int
                flag to indicate whether to search in both directions (0), only
                in the direction of the root (1) or in the direction away from
                the root (2).

        Returns
        -------
            loc_idxices: list of ints
                indices of the locations closest to the given locs
        """
        self._try_name(name)
        # create the locs in a desirable format
        locs_ = []
        for loc in locs:
            locs_.append(MorphLoc(loc, self))
        locs = locs_
        # look for the location indices
        loc_idxices = []
        for loc in locs:
            loc_idx1 = None
            loc_idx2 = None
            # find the location indices if necessary
            if direction == 0 or direction == 1:
                loc_idx1 = self._find_locs_to_root(
                    loc, name, check_siblings=check_siblings
                )
            if direction == 0 or direction == 2:
                loc_idx2 = self._find_locs_from_root(loc, name)
            # save the index of the closest location, if it exists and
            # if it is asked for
            if loc_idx1 == None and (direction == 0 or direction == 2):
                loc_idxices.append(loc_idx2)
            elif loc_idx2 == None and (direction == 0 or direction == 1):
                loc_idxices.append(loc_idx1)
            else:
                L1 = self.path_length(loc, self.locs[name][loc_idx1])
                L2 = self.path_length(loc, self.locs[name][loc_idx2])
                if L1 >= L2:
                    loc_idxices.append(loc_idx2)
                else:
                    loc_idxices.append(loc_idx1)
        return loc_idxices

    def _find_locs_from_root(self, loc, name):
        look_further = False
        # look if there are locs on the same node
        n_inds = np.where(loc["node"] == self.nids[name])[0]
        if len(n_inds) > 0:
            if loc["node"] == 1:
                loc_idx = n_inds[0]
            else:
                x_inds = np.where(loc["x"] <= self.xs[name][n_inds])[0]
                if len(x_inds) != 0:
                    ind = np.argmin(self.xs[name][n_inds][x_inds])
                    loc_idx = n_inds[x_inds[ind]]
                else:
                    look_further = True
        else:
            look_further = True
        # if no locs on the same node, then proceed to child nodes
        # else, return the smallest location larger than loc
        if look_further:
            node = self[loc["node"]]
            cnodes = node.get_child_nodes()
            loc_idxs = []
            for cnode in cnodes:
                cloc_idx = self._find_locs_from_root(
                    {"node": cnode.index, "x": 0.0}, name
                )
                if cloc_idx != None:
                    loc_idxs.append(cloc_idx)
            # get the one that is closest, if they exist
            pl_aux = 1e4
            ind_loc = 0
            for i, l_i in enumerate(loc_idxs):
                pl = self.path_length(
                    {"node": loc["node"], "x": 1.0}, self.locs[name][l_i]
                )
                if pl < pl_aux:
                    pl_aux = pl
                    ind_loc = i
            if pl_aux > 0.0 and len(loc_idxs) > 0:
                loc_idx = loc_idxs[ind_loc]
            elif pl_aux == 0.0 and node.index == 1:
                loc_idx = loc_idxs[ind_loc]
            else:
                loc_idx = None
        return loc_idx

    def _find_locs_to_root(self, loc, name, check_siblings=True):
        look_further = False
        # look if there are locs on the same node
        n_inds = np.where(loc["node"] == self.nids[name])[0]
        if len(n_inds) > 0:
            if loc["node"] == 1:
                loc_idx = n_inds[0]
            else:
                x_inds = np.where(loc["x"] >= self.xs[name][n_inds])[0]
                if len(x_inds) != 0:
                    ind = np.argmax(self.xs[name][n_inds][x_inds])
                    loc_idx = n_inds[x_inds[ind]]
                else:
                    look_further = True
        else:
            look_further = True
        if look_further:
            # if no locs on the same node, then proceed to resp. parent and child nodes
            node = self[loc["node"]]
            pnode = node.get_parent_node()
            loc_idxs = []
            # check parent node
            if pnode != None:
                ploc_idx = self._find_locs_to_root(
                    {"node": pnode.index, "x": 1.0}, name, check_siblings=check_siblings
                )
                if ploc_idx != None:
                    loc_idxs.append(ploc_idx)
            # check other child nodes of parent node
            if pnode != None and check_siblings:
                ocnodes = copy.copy(pnode.get_child_nodes())
                ocnodes.remove(node)
            else:
                ocnodes = []
            for cnode in ocnodes:
                cloc_idx = self._find_locs_from_root(
                    {"node": cnode.index, "x": 0.0}, name
                )
                if cloc_idx != None:
                    loc_idxs.append(cloc_idx)
            # get the one that is closest, if they exist
            pl_aux = 1e4
            ind_loc = 0
            for i, l_i in enumerate(loc_idxs):
                pl = self.path_length(
                    {"node": loc["node"], "x": 1.0}, self.locs[name][l_i]
                )
                if pl < pl_aux:
                    pl_aux = pl
                    ind_loc = i
            if pl_aux > 0.0 and len(loc_idxs) > 0:
                loc_idx = loc_idxs[ind_loc]
            else:
                loc_idx = None
        return loc_idx

    def get_nearest_neighbour_loc_idxs(self, loc, loc_arg):
        """
        Search nearest neighbours to `loc` in `loc_arg`.

        Parameters
        ----------
        loc: tuple, dict or `neat.MorphLoc`
            The locations for which nearest neighbours have to be found
        loc_arg: str or list of locs
            See documentation of `MorphTree.convert_loc_arg_to_locs`, the set of locations
            within which to look for nearest neighbours

        Returns
        -------
        list of ints
            Indices of nearest neighbours of `loc` in `loc_arg`
        """
        # preprocess loc_arg
        loc = MorphLoc(loc, self)
        if isinstance(loc_arg, str):
            name = loc_arg
            locs = self.convert_loc_arg_to_locs(loc_arg)
        else:
            name = "nn aux"
            locs = loc_arg
            self.store_locs(locs, name=name)

        nns = []
        # search for nearest neighbours
        node = self[loc["node"]]
        locinds_aux = np.where(node.index == self.nids[name])[0]
        if len(locinds_aux) > 0:
            dx = self.xs[name][locinds_aux] - loc["x"]
            # locs on node in down direction
            inds_down = np.where(dx >= 0)[0]
            if len(inds_down) > 0:
                ind_aux = np.argmin(dx[inds_down])
                nns.append(locinds_aux[inds_down][ind_aux])
            else:
                for c_node in node.child_nodes:
                    self._search_nn_from_root(c_node, nns, name)
            # locs on node in up direction
            inds_up = np.where(dx <= 0)[0]
            if len(inds_up) > 0:
                ind_aux = np.argmax(dx[inds_up])
                nns.append(locinds_aux[inds_up][ind_aux])
            else:
                self._search_nn_to_root(node, nns, name)
        else:
            for c_node in node.child_nodes:
                self._search_nn_from_root(c_node, nns, name)
            self._search_nn_to_root(node, nns, name)

        if name == "nn aux":
            self.remove_locs(name)

        return list(set(nns))

    def _search_nn_to_root(self, node, nns, name):
        p_node = node.parent_node
        if p_node is not None:
            # up direction
            locinds_aux = np.where(p_node.index == self.nids[name])[0]
            xval = 0.0
            if len(locinds_aux) > 0:
                ind_aux = np.argmax(self.xs[name][locinds_aux])
                locind = locinds_aux[ind_aux]
                nns.append(locind)
                xval = self.xs[name][locind]
            else:
                self._search_nn_to_root(p_node, nns, name)
            # down direction
            if xval < 1.0 - 1e-5:
                for c_node in set(p_node.child_nodes) - {node}:
                    self._search_nn_from_root(c_node, nns, name)

    def _search_nn_from_root(self, node, nns, name):
        locinds_aux = np.where(node.index == self.nids[name])[0]
        if len(locinds_aux) > 0:
            ind_aux = np.argmin(self.xs[name][locinds_aux])
            locind = locinds_aux[ind_aux]
            nns.append(locind)
        else:
            for c_node in node.child_nodes:
                self._search_nn_from_root(c_node, nns, name)

    def get_leaf_loc_idxs(self, name, recompute=False):
        """
        Find the indices in the location list specified stored under `name`
        that are 'leafs', i.e. locations for which no other location exist
        that is farther from the root.

        Parameters
        ----------
            name: string
                name of the desired set of locations
            recompute: bool (optional, default ``False``)
                whether or not to force recomputing the distances

        Returns
        -------
            list of inds
                the indices of the 'leaf' locations
        """
        try:
            if recompute:
                raise KeyError
            self.leafinds[name]
        except KeyError:
            self._try_name(name)
            self.leafinds[name] = []
            locs = self.locs[name]
            for ind, loc in enumerate(locs):
                if not self._has_loc_from_root(loc, name):
                    self.leafinds[name].append(ind)
        return self.leafinds[name]

    def _has_loc_from_root(self, loc, name):
        look_further = False
        # look if there are locs on the same node
        if loc["node"] != 1:
            n_inds = np.where(loc["node"] == self.nids[name])[0]
            if len(n_inds) > 0:
                x_inds = np.where(loc["x"] < self.xs[name][n_inds])[0]
                if len(x_inds) > 0:
                    returnbool = True
                else:
                    look_further = True
            else:
                look_further = True
        else:
            look_further = True
        # if no locs on the same node, then proceed to child nodes
        if look_further:
            node = self[loc["node"]]
            cnodes = node.child_nodes
            returnbool = False
            for cnode in cnodes:
                if self._has_loc_from_root({"node": cnode.index, "x": 0.0}, name):
                    returnbool = True
        return returnbool

    def distances_to_soma(self, loc_arg, recompute=False):
        """
        Compute the distance of each location in a given set to the soma

        Parameters
        ----------
        loc_arg: list of locations or string
            if list of locations, specifies the locations, if str,
            specifies the name under which the set of location is stored
            that should be used to create the new tree

        Returns
        -------
            np.array of float
                the distances to the soma of the corresponding locations
            recompute: bool (optional)
                whether or not to force recomputing the distances
        """
        # process input argument
        if isinstance(loc_arg, list):
            locs = [MorphLoc(loc, self) for loc in loc_arg]
            recompute = True
            save = False
        elif isinstance(loc_arg, str):
            name = loc_arg
            self._try_name(name)
            locs = self.get_locs(name)
            recompute = not (name in self.d2s) or recompute
            save = True
        else:
            raise IOError("`loc_arg` should be list of locs or string")

        if recompute:
            d2s = np.array(
                [self.path_length({"node": 1, "x": 0.0}, loc) for loc in locs]
            )
        else:
            d2s = self.d2s[name]

        if save:
            self.d2s[name] = d2s

        return d2s

    def distances_to_bifurcation(self, name, recompute=False):
        """
        Compute the distance of each location to the nearest bifurcation in
        the 'up' direction (towards root)

        Parameters
        ----------
        name: str
            name of the set of locations
        recompute: bool (optional, default ``False``)
            whether or not to force recomputing the distances

        Returns
        -------
        np.array of floats
            the distances to the nearest bifurcation of the corresponding
            locations
        """
        try:
            if recompute:
                raise KeyError
            return self.d2b[name]
        except KeyError:
            self._try_name(name)
            self.d2b[name] = []
            locs = self.locs[name]
            for i, loc in enumerate(locs):
                if loc["node"] != 1:
                    if loc["node"] != locs[i - 1]["node"]:
                        node = self[loc["node"]]
                        bnode, _ = self.find_bifurcation_node_to_root(node)
                    self.d2b[name].append(
                        self.path_length({"node": bnode.index, "x": 1.0}, loc)
                    )
                else:
                    self.d2b[name].append(0.0)
            return self.d2b[name]

    def distribute_locs_at_d2s(self, d2s, node_arg=None, name="dont save"):
        """
        Distributes locs on a given set of nodes at specified distances to the
        soma ('d2s'). If the specified distances are on the specified nodes, the list
        of locations will be empty. The locations are stored if the name is set
        to be something other than 'dont save'. On each node, locations are
        ordered from low to high x-values.

        Parameters
        ----------
            d2s: numpy.array of floats
                the distances from the soma at which to put the locations (micron)
            node_arg:
                see documentation of `MorphTree.convert_node_arg_to_nodes`
            name: string
                the name under which the locations are stored. Defaults to 'dont save'
                which means the locations are not stored

        Returns
        -------
            list of `neat.MorphLoc`
                the list of locations
        """
        # distribute the locations
        locs = []
        for node in self.convert_node_arg_to_nodes(node_arg):
            if node.parent_node != None:
                L0 = self.path_length(
                    {"node": 1, "x": 0.5}, {"node": node.index, "x": 0.0}
                )
                L1 = self.path_length(
                    {"node": 1, "x": 0.5}, {"node": node.index, "x": 1.0}
                )
                inds = np.where(np.logical_and(L0 < d2s, d2s <= L1))[0]
                Ls = np.sort(d2s[inds])
                locs.extend(
                    [
                        MorphLoc((node.index, (L - L0) / (L1 - L0)), self)
                        for L in Ls
                        if L > 1e-12
                    ]
                )
            elif np.any(np.abs(d2s) <= 1e-12):
                # node is soma, append a location on the soma
                locs.append(MorphLoc((node.index, 0.5), self))
        if name != "dont save":
            self.store_locs(locs, name=name)
        return locs

    @computational_tree_decorator
    def distribute_locs_uniform(
        self, dx, node_arg=None, add_bifurcations=False, name="dont save"
    ):
        """
        Distributes locations as uniformly, i.e. for a given distance
        between locations `dx`, locations are distributed equidistantly on each
        given node in the computational tree and their amount is computed
        so that the distance in between them is as close to `dx` as possible.
        Note that if a (computational tree) node has `L < dx / 2`, no locations
        will be distributed on that node.
        Locations in the list follow a depth-first ordering.

        Parameters
        ----------
            dx: float (> 0)
                target distance in micron between the locations
            node_arg:
                see documentation of `MorphTree.convert_node_arg_to_nodes`
            add_bifurcations: bool
                whether to ensure that all bifurcation nodes are added
            name: string
                the name under which the locations are stored. Defaults to 'dont save'
                which means the locations are not stored

        Returns
        -------
            list of `neat.MorphLoc`
                the list of locations
        """
        assert dx > 0
        # distribute the locations
        locs = []
        ii = 0
        for node in self.convert_node_arg_to_nodes(node_arg):
            ii += 1
            if node.parent_node == None:
                locs.append(MorphLoc((node.index, 0.5), self, set_as_comploc=True))
            else:
                Nloc = np.round(node.L / dx)
                if Nloc == 0 and len(node.child_nodes) > 1 and add_bifurcations:
                    Nloc = 1
                xvals = np.arange(1, Nloc + 1) / float(Nloc)
                locs.extend(
                    [
                        MorphLoc((node.index, xv), self, set_as_comploc=True)
                        for xv in xvals
                    ]
                )
        if name != "dont save":
            self.store_locs(locs, name=name)
        return locs

    def distribute_locs_random(
        self,
        num,
        dx=0.001,
        node_arg=None,
        add_soma=True,
        name="dont save",
        seed=None,
    ):
        """
        Returns a list of input locations randomly distributed on the tree.
        Locations are distributed uniformly, with the optional exclusion of an interval
        of size `dx` around each location.

        Parameters
        ----------
        num: int
            number of inputs
        dx: float (optional)
            minimal or given distance between input locations (micron)
        node_arg (optional):
            see documentation of `MorphTree.convert_node_arg_to_nodes`
        add_soma: bool (optional)
            whether or not to include the possibility of adding locations on the
            soma
        name: string (optional)
            the name under which the locations are stored. Defaults to 'dont save'
            which means the locations are not stored
        seed: int (optiona)
            Seed for numpy random number generator

        Returns
        -------
        list of `neat.MorphLoc`
            the locations
        """
        rng = np.random.default_rng(seed=seed)

        # setup, tag all nodes as empty intervals (no space is excluded for sampling)
        # when a location is sampled, we add an interval around that location to the tag,
        # which will exclude that interval from being sampled
        for node in self:
            node.content["tag"] = interval()

        # use the requested subset of nodes
        nodes = [
            node for node in self.convert_node_arg_to_nodes(node_arg) if node.index != 1
        ]
        # initialize the loclist with or without soma
        if add_soma:
            locs = [MorphLoc({"node": 1, "x": 0.0}, self)]
            self.root.content["tag"] = interval([0.0, 1.0])
        else:
            locs = []

        nodes_left = [node.index for node in nodes]
        probs_left = np.array([node.L for node in nodes])
        probs_left /= np.sum(probs_left)

        jj = num
        while jj > 0 and len(nodes_left) > 0:
            index = rng.choice(nodes_left, p=probs_left, replace=True)
            node = self[index]

            # draw an x-value not in the tagged intervals
            x = rng.uniform()
            ii = 0
            while x in node.content["tag"] and ii < 100:
                x = rng.uniform()
                ii += 1

            if ii == 100:
                node.content["tag"] = interval([0.0, 1.0])
            else:
                # add the location
                locs.append(MorphLoc((index, x), self))
                jj -= 1

            # tag new intervals
            self._tag_nodes_from_root(locs[-1], node, dx=dx)
            self._tag_nodes_to_root(locs[-1], node, dx=dx)

            # if node is full, remove it from sampling set
            nodes_left_ = []
            probs_left_ = []
            for nidx, prob in zip(nodes_left, probs_left):
                if interval([0, 1]) not in self[nidx].content["tag"]:
                    nodes_left_.append(nidx)
                    probs_left_.append(prob)
            nodes_left = nodes_left_
            probs_left = np.array(probs_left_) / np.sum(probs_left_)

        self._remove_tags()

        # store the locations
        if name != "dont save":
            self.store_locs(locs, name=name)
        return locs

    def _tag_nodes_from_root(self, start_loc, node, dx=0.001):
        """
        tag intervals within dx from the start_loc, away from root
        """
        if dx <= 0:
            return

        # check if we are on the first node or deeper into the recursion
        if node.index == start_loc["node"]:
            _loc = start_loc
        else:
            _loc = MorphLoc((node.index, 0.0), self)

        d0 = _loc["x"] * node.L
        d1 = d0 + dx

        if d1 > node.L:
            # append the interval to the excluded zone and continue
            node.content["tag"] = node.content["tag"] | interval([_loc["x"], 1.0])

            # leftover part of dx
            dx -= node.L - d0

            for cnode in node.child_nodes:
                self._tag_nodes_from_root(start_loc, cnode, dx=dx)
        else:
            # append the interval to the excluded zone
            node.content["tag"] = node.content["tag"] | interval(
                [_loc["x"], d1 / node.L]
            )

    def _tag_nodes_to_root(self, start_loc, node, cnode=None, dx=0.001):
        """
        tag intervals within dx from the start_loc, towards from root
        """
        # check if we are on the first node or deeper into the recursion
        if node.index == start_loc["node"]:
            _loc = start_loc
        else:
            _loc = MorphLoc((node.index, 1.0), self)

        d0 = _loc["x"] * node.L
        d1 = d0 - dx

        if d1 < 0.0:
            node.content["tag"] = node.content["tag"] | interval([0.0, _loc["x"]])

            dx -= d0

            # if we are deeper in the recursion, we also have to assess
            # sibling branches at bifurcations
            if _loc != start_loc:
                cnodes = node.child_nodes
                if len(cnodes) > 1:
                    if cnode != None:
                        cnodes = list(set(cnodes) - set([cnode]))
                    for cn in cnodes:
                        self._tag_nodes_from_root(start_loc, cn, dx=dx)

            pnode = node.get_parent_node()
            if pnode != None:
                self._tag_nodes_to_root(start_loc, pnode, node, dx=dx)

        else:
            node.content["tag"] = node.content["tag"] | interval(
                [d1 / node.L, start_loc["x"]]
            )

    def _remove_tags(self):
        for node in self:
            if "tag" in node.content:
                del node.content["tag"]

    def distribute_locs_finite_diff(
        self, dx_max=15.0, node_arg=None, name="dont store"
    ):
        """
        Distribute locs in such a way that they correspond to the compartment
        locations under NEAT's finite difference approximation.

        Parameters
        ----------
        dx_max: float
            Maximum distance step between compartments (in [um]). By default,
            each node of this tree will correspond to at least one compartment,
            and thus one node in the comparment tree. If the length of a node
            exceeds `dx_max`, there will be the smallest possible number of
            equally spaced comparments so that the distance between them does
            not exceed `dx_max`. Note that if the computational tree is active,
            the computational nodes will be taken as a reference for placing
            the compartment locations.
        node_arg (optional):
            see documentation of `MorphTree.convert_node_arg_to_nodes`
        name: string (optional)
            the name under which the locations are stored. Defaults to 'dont save'
            which means the locations are not stored

        Returns
        -------
        list of `neat.MorphLoc`
            the locations
        """
        set_as_comploc = self.check_computational_tree_active()

        locs = []
        for node in self.convert_node_arg_to_nodes(node_arg):
            if self.is_root(node):
                locs.append(
                    MorphLoc((node.index, 0.5), self, set_as_comploc=set_as_comploc)
                )
            else:
                n_comp = np.ceil(node.L / dx_max).astype(int)

                for cc in range(1, n_comp + 1):
                    new_loc = MorphLoc(
                        (node.index, cc / n_comp), self, set_as_comploc=set_as_comploc
                    )
                    locs.append(new_loc)

        if name != "dont store":
            self.store_locs(locs, name)

        return locs

    def extend_with_bifurcation_locs(self, loc_arg, name="dont save"):
        """
        Extends input loc_arg with the intermediate bifurcations. They are
        appended to the end of the list

        Parameters
        ----------
        loc_arg: list of `neat.MorphLoc` or string
            the locations
        name: string (optional)
            The name under which the list of bifurcation locs will be stored.
            Defaults to 'dont save' which means they are not stored.

        Returns
        -------
        list of `neat.MorphLoc`
            the extended location list
        """
        locs = self.convert_loc_arg_to_locs(loc_arg)
        # get the bifurcation locs
        nodes = [self[loc["node"]] for loc in locs]
        bnodes = self.find_in_between_bifurcation_nodes(nodes)
        blocs = [MorphLoc((bnode.index, 1.0), self) for bnode in bnodes]
        # retain unique locs
        all_locs = self.unique_locs(locs + blocs)
        # store the locations
        if name != "dont save":
            self.store_locs(all_locs, name=name)
        return all_locs

    def unique_locs(self, loc_arg, name="dont save"):
        """
        Gets the unique locations in the provided locs

        Parameters
        ----------
        loc_arg: list of `neat.MorphLoc` or string
            the locations
        name: string (optional)
            The name under which the list of bifurcation locs will be stored.
            Defaults to 'dont save' which means they are not stored.

        Returns
        -------
        list of `neat.MorphLoc`
            the bifurcation locs
        """
        locs = self.convert_loc_arg_to_locs(loc_arg)
        locs_ = reduce(lambda l, x: l.append(x) or l if x not in l else l, locs, [])

        if name != "dont save":
            self.store_locs(locs_, name=name)
        return locs_

    def make_x_axis(self, dx=10.0, node_arg=None, loc_arg=None):
        """
        Create a set of locs suitable for serving as the x-axis for 1D plotting.
        The neurons is put on a 1D axis with a depth-first ordering.

        Parameters
        ----------
        dx: float
            target separation between the plot points (micron)
        node_arg:
            see documentation of `MorphTree.convert_node_arg_to_nodes`
            The nodes on which the locations for the x-axis are distributed.
            When this is given as a list of nodes, assumes a depth first
            ordering.
        loc_arg: list of locs or string
            if list of locs, these locs will be used as x-axis, if string, name
            of set of locs on the morphology that will be used as x-axis

        """
        if loc_arg is None:
            # if comptree has not been set, create a basic one for plotting
            if self._computational_root == None:
                self.set_comp_tree()
            # distribute the x-axis locations
            self.distribute_locs_uniform(dx, node_arg=node_arg, name="xaxis")
            # get the root node
            nodes = self.convert_node_arg_to_nodes(node_arg)
            # check that first node is root
            for node in nodes:
                if nodes[0] in node.child_nodes:
                    raise ValueError(
                        "Input `node_arg` is not a depth-first ordered"
                        " list of nodes."
                    )
            # set the node colors for original and computational trees
            tempnode_orig = self._find_comp_node_from_root(nodes[0])
            self.set_node_colors(tempnode_orig)
            with self.as_computational_tree:
                tempnode_comp = self[tempnode_orig.index]
                self.set_node_colors(tempnode_comp)

        else:
            if isinstance(loc_arg, list):
                self.store_locs(loc_arg, name="xaxis")
            elif isinstance(loc_arg, str):
                self.store_locs(self.get_locs(loc_arg), name="xaxis")
            else:
                raise IOError("`loc_org` should be string or list of locs")
        # compute the x-axis 1D array
        pinds = self.get_leaf_loc_idxs("xaxis")
        d2s = self.distances_to_soma("xaxis")
        xaxis = d2s[0 : pinds[0] + 1].tolist()
        d_add = d2s[pinds[0]]
        for ii in range(0, len(pinds) - 1):
            xaxis.extend(
                (
                    d_add + d2s[pinds[ii] + 1 : pinds[ii + 1] + 1] - d2s[pinds[ii] + 1]
                ).tolist()
            )
            d_add += d2s[pinds[ii + 1]] - d2s[pinds[ii] + 1]
        self.xaxis = np.array(xaxis)

    def set_node_colors(self, startnode=None):
        """
        Set the color code for the nodes for 1D plotting

        Parameters
        ----------
            node: int or `neat.MorphNode`
                index of the node or node whose subtree will be colored. Defaults
                to the root
        """
        if startnode == None:
            startnode = self.root
        for node in self:
            node.content["color"] = 0.0
        self.node_color = [0.0]  # trick to pass the pointer and not the number itself
        self._set_node_colors_from_root(startnode)

    def _set_node_colors_from_root(self, node):
        node.content["color"] = self.node_color[0]
        if self.is_leaf(node):
            self.node_color[0] += 1.0
        for cnode in node.child_nodes:
            self._set_node_colors_from_root(cnode)

    def get_x_values(self, locs):
        """
        Get the corresponding location on the x-axis of the input locations

        Parameters
        ----------
            locs: list of tuples, dicts or `neat.MorphLoc`
                list of the locations
        """
        locinds = np.array(self.get_nearest_loc_idxs(locs, "xaxis")).astype(int)
        return self.xaxis[locinds]

    def plot_1d(self, ax, parr, *args, **kwargs):
        """
        Plot an array where each element corresponds to the matching location on
        the x-axis with a depth-first ordering on a 1D plot

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object on which the plot will be made
            parr: numpy.array of floats
                the array that will be plotted
            args, kwargs:
                arguments for `matplotlib.pyplot.plot`

        Returns
        -------
            lines: list of `matplotlib.lines.Line2D` instances
                the line segments corresponding to the value of the plotted array
                in each branch

        Raises
        ------
            AssertionError
                When the number of elements in the data array in not equal to
                the number of elements on the x-axis
        """
        assert len(parr) == len(self.locs["xaxis"])
        pinds = self.get_leaf_loc_idxs("xaxis")
        d2s = self.distances_to_soma("xaxis")
        # make the plot
        lines = []

        line = ax.plot(
            self.xaxis[0 : pinds[0] + 1], parr[0 : pinds[0] + 1], *args, **kwargs
        )
        lines.append(line[0])
        if "label" in list(kwargs.keys()):
            kwargs = copy.deepcopy(kwargs)
            del kwargs["label"]
        for ii in range(0, len(pinds) - 1):
            line = ax.plot(
                self.xaxis[pinds[ii] + 1 : pinds[ii + 1] + 1],
                parr[pinds[ii] + 1 : pinds[ii + 1] + 1],
                *args,
                **kwargs,
            )
            lines.append(line[0])
        return lines

    def set_line_data(self, lines, parr):
        """
        Update the line objects with new data

        Parameters
        ----------
            lines: list of `matplotlib.lines.Line2D` instance
                the line segments of which the data has to be updated
            parr: numpy.array of floats
                the array that will be put in the line segments

        Raises
        ------
            AssertionError
                When the number of elements in the data array in not equal to
                the number of elements on the x-axis
        """
        assert len(parr) == len(self.locs["xaxis"])
        pinds = self.get_leaf_loc_idxs("xaxis")
        d2s = self.distances_to_soma("xaxis")
        lines[0].set_data(self.xaxis[0 : pinds[0] + 1], parr[0 : pinds[0] + 1])
        for ii in range(0, len(pinds) - 1):
            ll = ii + 1
            lines[ll].set_data(
                self.xaxis[pinds[ii] + 1 : pinds[ii + 1] + 1],
                parr[pinds[ii] + 1 : pinds[ii + 1] + 1],
            )

    def plot_true_d2s(self, ax, parr, cmap=None, **kwargs):
        """
        Plot an array where each element corresponds to the matching location in
        the x-axis location list. Now all locations are plotted at their true
        distance from the soma.

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object on which the plot will be made
            parr: numpy.array of floats
                the array that will be plotted
            cmap: `matplotlib.colors.Colormap` instance
                If provided, the lines will be colored according to the branch
                to which they belong, in colors specified by the colormap
            kwargs:
                keyword arguments for `matplotlib.pyplot.plot`

        Returns
        -------
            lines
            lines: list of `matplotlib.lines.Line2D`
                the line segments corresponding to the value of the plotted array
                in each branch

        Raises
        ------
            AssertionError
                When the number of elements in the data array in not equal to
                the number of elements on the x-axis
        """
        assert len(parr) == len(self.locs["xaxis"])
        locs = self.locs["xaxis"]
        pinds = self.get_leaf_loc_idxs("xaxis")
        d2s = self.distances_to_soma("xaxis")
        # list of colors for plotting
        cs = {node.index: node.content["color"] for node in self}
        cplot = [cs[loc["node"]] for loc in locs]
        max_cs = max(cplot)
        min_cs = min(cplot)
        if np.abs(max_cs - min_cs) < 1e-12:
            norm_cs = max_cs + 1e-2
        else:
            norm_cs = (max_cs - min_cs) * (1.0 + 1.0 / 100.0)
        # create the truespace plot
        lines = []
        if cmap != None:
            kwargs["c"] = cmap((cplot[0] - min_cs) / norm_cs)
            if "color" in kwargs:
                del kwargs["color"]
        line = ax.plot(d2s[0 : pinds[0] + 1], parr[0 : pinds[0] + 1], **kwargs)
        lines.append(line[0])
        if "label" in kwargs:
            del kwargs["label"]
        for ii in range(0, len(pinds) - 1):
            if cmap != None:
                kwargs["c"] = cmap((cs[locs[pinds[ii] + 1]["node"]] - min_cs) / norm_cs)
            line = ax.plot(
                d2s[pinds[ii] + 1 : pinds[ii + 1] + 1],
                parr[pinds[ii] + 1 : pinds[ii + 1] + 1],
                **kwargs,
            )
            lines.append(line[0])
        return lines

    def _add_scalebar(
        self,
        ax,
        xlabel=f" $\mu$m",
        fstr_xlabel=r"%.0f ",
        lx_offset=0.1,
        ly_offset=0.1,
        bx_offset=0.05,
        by_offset=0.05,
        bc_offset=0.0,
        sb_width=4.0,
        text_kwargs_x=dict(size=15.0, rotation=0, va="center"),
    ):
        xticks = ax.get_xticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        transf = ax.transData.inverted() + ax.transAxes
        p0, p1 = transf.transform((0.0, 0.0)), transf.transform((0.0, 1.0))
        bcx_offset = p1[0] - p0[0]
        bcy_offset = p1[1] - p0[1]
        p0, p1 = transf.transform((0.0, 0.0)), transf.transform((by_offset, -bx_offset))
        bx_offset = p1[1] - p0[1]
        by_offset = p1[0] - p0[0]
        p0, p1 = transf.transform((0.0, 0.0)), transf.transform((ly_offset, -lx_offset))
        lx_offset = p1[1] - p0[1]
        ly_offset = p1[0] - p0[0]

        # position and length
        sblen = xticks[-1] - xticks[-2]
        xpos = (xlim[0] + xlim[1]) / 2.0 + bcx_offset
        ypos = ylim[0] + bx_offset

        px = (xpos, ypos)
        xbar = ((xpos - sblen / 2.0, xpos + sblen / 2.0), (ypos, ypos))

        # draw the scale bar
        ax.plot(*xbar, "k-", lw=sb_width, clip_on=False)
        ax.annotate(
            fstr_xlabel % sblen + xlabel,
            xy=px,
            xytext=(px[0], px[1] + lx_offset),
            annotation_clip=False,
            transform=ax.transData,
            ha="center",
            **text_kwargs_x,
        )

        ax.tick_params(axis="x", which="both", length=0, color="none")

        ax.set_xlim(xlim)

    def color_x_axis(self, ax, cmap, add_scalebar=1):
        """
        Color the x-axis of a plot according to the morphology.

        !!! Has to be called after all lines are plotted !!!

        Furthermor, node colors have to be set first. This can be done with
        `MorphTree.set_node_colors()` or manually by adding a 'color' entry
        to the ``MorphNode.content`` dictionary

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object of which the x-axis will be colored
            cmap: `matplotlib.colors.Colormap` instance
                Colormap that determines the color of each branch
            sizex: float
                Size of scalebar (in micron). If set to None, no scalebar is
                plotted.
        """
        locs = self.locs["xaxis"]
        # list of colors for plotting
        cs = {node.index: node.content["color"] for node in self}
        cplot = [cs[loc["node"]] for loc in locs]
        max_cs = max(cplot)
        min_cs = min(cplot)
        if np.abs(max_cs - min_cs) < 1e-12:
            norm_cs = max_cs + 1e-2
        else:
            norm_cs = (max_cs - min_cs) * (1.0 + 1.0 / 100.0)
        # necessary distance arrays
        pinds = self.get_leaf_loc_idxs("xaxis")
        assert len(pinds) > 0
        d2s = self.distances_to_soma("xaxis")
        # plot colored xaxis
        ylim = np.array(ax.get_ylim())
        ax.plot(
            self.xaxis[0 : pinds[0] + 1],
            [ylim[0] + 1e-9 for _ in d2s[0 : pinds[0] + 1]],
            c=cmap((cplot[0] - min_cs) / norm_cs),
            lw=10,
        )
        for ii in range(0, len(pinds) - 1):
            if locs[pinds[ii] + 1]["node"] in list(cs.keys()):
                ax.plot(
                    self.xaxis[pinds[ii] + 1 : pinds[ii + 1] + 1],
                    [ylim[0] + 1e-9 for _ in d2s[pinds[ii] + 1 : pinds[ii + 1] + 1]],
                    c=cmap((cs[locs[pinds[ii] + 1]["node"]] - min_cs) / norm_cs),
                    lw=10,
                )
            else:
                ax.plot(
                    self.xaxis[pinds[ii] + 1 : pinds[ii + 1] + 1],
                    [ylim[0] + 1e-9 for _ in d2s[pinds[ii] + 1 : pinds[ii + 1] + 1]],
                    c="k",
                    lw=10,
                )
        ax.set_ylim((ylim[0], ylim[1]))
        # add scalebar
        if add_scalebar:
            self._add_scalebar(ax)
        ax.axes.get_xaxis().set_visible(False)

    def plot_2d_morphology(
        self,
        ax,
        node_arg=None,
        cs=None,
        cminmax=None,
        cmap=None,
        use_radius=1,
        draw_soma_circle=1,
        plotargs={},
        textargs={},
        marklocs=[],
        loc_args={},
        marklabels={},
        labelargs={},
        cb_draw=0,
        cb_orientation="vertical",
        cb_label="",
        sb_draw=1,
        sb_scale=100,
        sb_width=5.0,
        set_lims=True,
        lims_margin=0.1,
        soma_pos=None,
    ):
        """
        Plot the morphology projected on the x,y-plane

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object on which the plot will be drawn
            node_arg:
                see documentation of `MorphTree.convert_node_arg_to_nodes`
            cs: dict {int: float}, None or 'node_color'
                If dict, node indices are keys and the float value will
                correspond to the plotted color. If None, the color of the tree
                will be the one specified in ``plotargs``. Note that the dict
                does not have to contain all node indices. The ones that are not
                featured in the dict are plot in the color specified in ``plotargs``.
                If 'node_color', colors will be those stored on the nodes. Note
                that choosing this option when there are nodes without 'color'
                as an entry in ``node.content`` will result in an error.
            cminmax: (float, float) or None (default)
                The min and max values of the color scale (if cs is provided).
                If None, the min and max values of cs are used.
            cmap: `matplotlib.colors.Colormap` instance
                colormap fram which colors in ``cs`` are taken
            use_radius: bool
                If ``True``, uses the swc radius for the width of the line
                segments
            draw_soma_circle: bool
                If ``True``, draws the soma as a circle, otherwise doesn't draw
                soma
            plotargs: dict
                `kwargs` for `matplotlib.pyplot.plot`. 'c'- or 'color'-
                argument will be overwritten when cs is defined. 'lw'- or
                'linewidth' argument will be multiplied with the swc radius of
                the node if `use_radius` is ``True``.
            textargs: dict
                text properties for various labels in the plot
            marklocs: list of tuples, dicts or instances of `neat.MorphLoc`
                Location that will be plotted on the morphology
            loc_args: dict or list of dict
                `kwargs` for `matplotlib.pyplot.plot` for the location.
                Use only point markers and no lines! When it is a single dict
                all location will have the same marker. When it is a list it
                should have the same length as `marklocs`.
            marklabels: dict {int: string}
                Keys are indices of locations in `marklocs`, values are strings
                that are used to annotate the corresponding locations
            labelargs: dict
                text properties for the location annotation
            cb_draw: bool
                Whether or not to draw a `matplotlib.pyplot.colorbar()`
                instance.
            cb_orientation: string, 'vertical' or 'horizontal'
                The colorbars' orientation
            cb_label: string
                The label of the colorbar
            sb_draw: bool
                Whether or not to draw a scale bar
            sb_scale: float
                Lenght of the scale bar (micron)
            sb_width: float
                Width of the scale bar
            set_lims: bool (optional, default ``True``)
                set ``ax`` limits based on the morphology
            lims_margin: float
                the margin, as fraction of total width and height of tree, at
                which the limits are placed
            soma_pos: np.ndarray of float (shape=(3,))
                translate the soma position the the given coordinates
        """
        # apply the translation
        if soma_pos is None:
            soma_pos = np.zeros((3,))
        for node in self:
            node.xyz += soma_pos
        # default cmap
        if cmap is None:
            cmap = cm.get_cmap("jet")
        # ensure color is indicated by the 'c'-parameter in `plotargs`
        if "color" in plotargs:
            plotargs["c"] = plotargs["color"]
            del plotargs["color"]
        elif "c" not in plotargs:
            plotargs["c"] = "k"
        # define a norm for the colors, if defined
        if cs == "node_color":
            cs = {node.index: node.content["color"] for node in self}
        if cs is not None:
            if cminmax is None:
                if len(cs) > 0:
                    max_cs = cs[max(cs, key=cs.__getitem__)]  # works for dict and list
                    min_cs = cs[min(cs, key=cs.__getitem__)]  # works for dict and list
                else:
                    min_cs, max_cs = 0.0, 1.0
            else:
                min_cs = cminmax[0]
                max_cs = cminmax[1]
            norm = pl.Normalize(vmin=min_cs, vmax=max_cs)
        # ensure linewidth is indicated as 'lw' in plotargs
        if "linewidth" in plotargs:
            plotargs["lw"] = plotargs["linewidth"]
            del plotargs["linewidth"]
        elif "lw" not in plotargs:
            plotargs["lw"] = 1.0
        plotargs_orig = copy.deepcopy(plotargs)
        # loc_args can be dictionary, so that the same properties hold for every
        # markloc, or can be list with the same size as marklocs, so that every
        # marker has different properties. `zorder` of the markers is also set
        # very high so that they are always in the foreground
        self.store_locs(marklocs, "plotlocs")
        xs = self.xs["plotlocs"]
        if isinstance(loc_args, dict):
            if "zorder" not in loc_args:
                loc_args["zorder"] = 1e4
            loc_args = [loc_args for _ in marklocs]
        else:
            assert len(loc_args) == len(marklocs)
            for loc_arg in loc_args:
                if "zorder" not in loc_arg:
                    loc_arg["zorder"] = 1e4
        # `marklabels` is a dictionary with as keys the index of the loc in
        # `marklocs` to which the label belongs. `labelargs` is the same for
        # every label
        for ind in marklabels:
            assert ind < len(marklocs)
        # plot the tree
        xlim = [0.0, 0.0]
        ylim = [0.0, 0.0]
        for node in self.convert_node_arg_to_nodes(node_arg):
            if node.xyz[0] < xlim[0]:
                xlim[0] = node.xyz[0]
            if node.xyz[0] > xlim[1]:
                xlim[1] = node.xyz[0]
            if node.xyz[1] < ylim[0]:
                ylim[0] = node.xyz[1]
            if node.xyz[1] > ylim[1]:
                ylim[1] = node.xyz[1]
            # find the locations that are on the current node
            inds = self.get_loc_idxs_on_node("plotlocs", node)
            if node.parent_node is None:
                # node is soma, draw as circle if necessary
                if draw_soma_circle:
                    if cs is not None and node.index in cs:
                        plotargs["c"] = cmap(norm(cs[node.index]))
                    else:
                        plotargs["c"] = plotargs_orig["c"]
                    circ = patches.Circle(node.xyz[0:2], node.R, color=plotargs["c"])
                    ax.add_patch(circ)
                for ind in inds:
                    self._plot_loc(
                        ax,
                        ind,
                        node.xyz[0],
                        node.xyz[1],
                        loc_args,
                        marklabels,
                        labelargs,
                    )
            else:
                # plot line segment associated with node
                nxyz = node.xyz
                pxyz = node.parent_node.xyz
                if cs is not None and node.index in cs:
                    plotargs["c"] = cmap(norm(cs[node.index]))
                else:
                    plotargs["c"] = plotargs_orig["c"]
                if use_radius:
                    plotargs["lw"] = plotargs_orig["lw"] * node.R
                ax.plot([pxyz[0], nxyz[0]], [pxyz[1], nxyz[1]], **plotargs)
                # plot the locations
                for ind in inds:
                    locxyz = pxyz + (nxyz - pxyz) * xs[ind]
                    self._plot_loc(
                        ax, ind, locxyz[0], locxyz[1], loc_args, marklabels, labelargs
                    )
        # margins
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        xlim[0] -= dx * lims_margin
        xlim[1] += dx * lims_margin
        ylim[0] -= dy * lims_margin
        ylim[1] += dy * lims_margin
        # draw a scale bar
        if sb_draw:
            scale = sb_scale
            dy = ylim[1] - ylim[0]
            dx = xlim[1] - xlim[0]
            ax.plot(
                [xlim[0], xlim[0] + scale],
                [ylim[0], ylim[0]],
                "k",
                linewidth=sb_width,
                zorder=1e5,
            )
            txt = ax.annotate(
                r"" + str(scale) + " μm",
                xy=(xlim[0] + scale / 2.0, ylim[0]),
                xycoords="data",
                xytext=(xlim[0] + scale / 2.0, ylim[0] - dy / 200.0),
                ha="center",
                va="top",
                **textargs,
            )
            # textcoords='offset points', **textargs)
            txt.set_path_effects([patheffects.withStroke(foreground="w", linewidth=2)])
        if set_lims:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", "datalim")
        if cs != None and cb_draw:
            # create colorbar ax
            divider = make_axes_locatable(ax)
            if cb_orientation == "horizontal":
                cax = divider.append_axes("bottom", "5%", pad="3%")
            else:
                cax = divider.append_axes("right", "5%", pad="3%")
            # create a mappable
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []  # fake array for scalar mappable
            # create the colorbar
            cb = pl.colorbar(sm, cax=cax, orientation=cb_orientation)
            ticks_cb = np.round(np.linspace(min_cs, max_cs, 7), decimals=1)
            cb.set_ticks(ticks_cb)
            if cb_orientation == "horizontal":
                cb.ax.xaxis.set_ticks_position("bottom")
            else:
                cb.ax.yaxis.set_ticks_position("right")
            cb.set_label(cb_label, **textargs)
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.draw_frame = False

        ax.set_xticks([])
        ax.set_yticks([])

        # remove the translation
        for node in self:
            node.xyz -= soma_pos

    def _plot_loc(self, ax, ind, xval, yval, loc_args, marklabels, labelargs):
        """
        plot a location on the morphology together with its annotation
        """
        ax.plot(xval, yval, **loc_args[ind])
        if ind in marklabels:
            txt = ax.annotate(
                marklabels[ind],
                xy=(xval, yval),
                xycoords="data",
                xytext=(5, 5),
                textcoords="offset points",
                **labelargs,
            )
            txt.set_path_effects([patheffects.withStroke(foreground="w", linewidth=2)])

    def plot_morphology_interactive(
        self,
        node_arg=None,
        use_radius=1,
        draw_soma_circle=1,
        plotargs={"c": "k", "lw": 1.0},
        project3d=False,
    ):
        """
        Show the morphology either in 3d or projected on the x,y-plane. When
        a line segment is clicked, the associated node is printed.

        Parameters
        ----------
            ax: `matplotlib.axes.Axes` instance
                the ax object on which the plot will be drawn
            node_arg:
                see documentation of `MorphTree.convert_node_arg_to_nodes`
            use_radius: bool
                If ``True``, uses the swc radius for the width of the line
                segments
            draw_soma_circle: bool
                If ``True``, draws the soma as a circle, otherwise doesn't draw
                soma
        """
        fig = pl.figure("Morphology interactive")
        ax = pl.gca(projection="3d") if project3d else pl.gca()
        # ax = pl.gca()
        if "c" not in plotargs:
            plotargs.update({"c": "k"})
        if "linewidth" in plotargs:
            plotargs["lw"] = plotargs["linewidth"]
            del plotargs["linewidth"]
        if "lw" not in plotargs:
            plotargs.update({"lw": "k"})
        plotargs_orig = copy.deepcopy(plotargs)
        # plot the tree
        node_line_associators = {}
        for ii, node in enumerate(self.convert_node_arg_to_nodes(node_arg)):
            if node.parent_node is not None:
                # plot line segment associated with node
                nxyz = node.xyz
                pxyz = node.parent_node.xyz
                if use_radius:
                    plotargs["lw"] = plotargs_orig["lw"] * node.R
                if project3d:
                    line = ax.plot(
                        [pxyz[0], nxyz[0]],
                        [pxyz[1], nxyz[1]],
                        [pxyz[2], nxyz[2]],
                        label=str(ii),
                        picker=2.0,
                        **plotargs,
                    )
                else:
                    line = ax.plot(
                        [pxyz[0], nxyz[0]],
                        [pxyz[1], nxyz[1]],
                        label=str(ii),
                        picker=2.0,
                        **plotargs,
                    )
                node_line_associators.update({str(ii): node})
            else:
                # node is soma, draw as circle if necessary
                if draw_soma_circle:
                    circ = patches.Circle(node.xyz[0:2], node.R, color=plotargs["c"])
                    ax.add_patch(circ)
        ax.axes.get_xaxis().set_visible(0)
        ax.axes.get_yaxis().set_visible(0)
        ax.axison = 0

        # define the clickevent action
        def onPick(event):
            line = event.artist
            node = node_line_associators[line.get_label()]
            # print the associated node
            print(
                "\n>>> line segment at "
                + str(node)
                + ", distance to soma (um) = "
                + str(
                    self.path_length(
                        {"node": node.index, "x": 1}, {"node": 1, "x": 0.0}
                    )
                )
            )

        # show morphology
        cid = fig.canvas.mpl_connect("pick_event", onPick)
        pl.show()

    def find_common_root(self, name):
        self._try_name(name)
        # get the node indices of nodes
        node_inds = self.get_node_indices(name)
        # find the paths to the root
        paths = [set(self.path_to_root(self[node_ind])) for node_ind in node_inds]
        # possible roots
        roots = list(set.intersection(*paths))
        # return the node of highest order
        rootind = np.argmax([self.order_of_node(node) for node in roots])
        return roots[rootind]

    def create_new_tree(
        self,
        loc_arg,
        name="new tree",
        fake_soma=False,
        new_tree=None,
    ):
        """
        Creates a new tree where the provided location in `loc_arg` are now the nodes.
        Note that if the soma is not in the list of locations, a common root location
        might be added if necessary.

        Distance relations between locations are maintained (note that this
        relation is stored in `L` attribute of `neat.MorphNode`, the `p3d`
        attribute containing the 3d coordinates does not maintain distances)

        The radius of a node is taken as the average radius between the location
        associated with the node and the location associated with the parent node,
        weighted by the lengths of all individual nodes.

        Parameters
        ----------
        loc_arg: list of `neat.MorphLoc` or string
            the locations.
        name: str (default 'new tree')
            The name under which the locations associated to the tree are stored.
        fake_soma: bool (default `False`)
            if `True`, finds the common root of the set of locations and
            uses that as the soma of the new tree. If `False`, the real soma
            is used.
        new_tree: `None` or instance of subclass of `neat.MorphTree`
            The new tree instance.

        Returns
        -------
        `neat.MorphTree`
            The new tree.

        Raises
        ------
        `ValueError`
            If `new_tree` is not a subclass of `self.__class__`
        """
        locs = self.convert_loc_arg_to_locs(loc_arg)
        self.store_locs(locs, name)
        if fake_soma:
            # find the common root of the set of locations
            snode = self.find_common_root(name)
        else:
            # use the soma as root
            snode = self[1]
        # add the common root to the set of locations, if necessary
        new_locs = self.unique_locs([MorphLoc((snode.index, 1.0), self)] + locs)
        self.store_locs(new_locs, name)

        # create new tree
        if new_tree is None:
            new_tree = self.__class__()
        elif not issubclass(type(new_tree), MorphTree):
            raise ValueError(
                f"`new_tree` is an instance of {new_tree.__class__}, "
                f"but should be a subclass of <class 'neat.MorphTree'>."
            )

        # create the root node
        p3d = (snode.xyz, snode.R, snode.swc_type)
        new_snode = new_tree.create_corresponding_node(1, p3d)
        new_snode.L = snode.L
        new_tree.set_root(new_snode)
        new_nodes = [new_snode]
        # set the index of the location associated with the current node
        nids = self.get_node_indices(name)
        new_snode.content["loc idx"] = (
            None if 1 not in nids else np.where(nids == 1)[0][0]
        )
        # make two other soma nodes
        if fake_soma:
            for index in [2, 3]:
                new_cnode = new_tree.create_corresponding_node(index, p3d)
                new_tree.add_node_with_parent(new_cnode, new_snode)
                new_nodes.append(new_cnode)
        else:
            for cnode in snode.get_child_nodes(skip_inds=[]):
                if cnode.index in [2, 3]:
                    p3d = (cnode.xyz, cnode.R, cnode.swc_type)
                    new_cnode = new_tree.create_corresponding_node(cnode.index, p3d)
                    new_tree.add_node_with_parent(new_cnode, new_snode)
                    new_nodes.append(new_cnode)
        # make rest of tree
        for cnode in snode.child_nodes:
            self._add_nodes_to_tree(
                cnode,
                new_snode,
                new_tree,
                new_nodes,
                name,
            )
        # set the lengths of the nodes
        for new_node in new_tree:
            if new_node.parent_node != None:
                L, R = self.path_length(
                    new_locs[new_node.parent_node.content["loc idx"]],
                    new_locs[new_node.content["loc idx"]],
                    compute_radius=True,
                )
            else:
                L = 0.0
                R = self[1].R
            new_node.L = L
            new_node.R = R

        # store the created root as the root of the original tree
        new_tree._original_root = new_tree.root

        return new_tree

    def _add_nodes_to_tree(self, node, new_pnode, new_tree, new_nodes, name):
        # get the specified locs
        xs = self.xs[name]
        # check which locinds are on the branch
        ninds = self.get_loc_idxs_on_node(name, node)
        order_inds = np.argsort(xs[ninds])
        for ind in np.array(ninds)[order_inds]:
            index = len(new_nodes) + 1
            # new coordinates
            new_xyz = node.parent_node.xyz * (1.0 - xs[ind]) + node.xyz * xs[ind]
            # set a fake radius, it's overwritten later
            fake_radius = -1.0
            # make new node
            p3d = (new_xyz, fake_radius, node.swc_type)
            new_node = new_tree.create_corresponding_node(index, p3d)
            new_node.content["loc idx"] = ind
            # add new node
            new_tree.add_node_with_parent(new_node, new_pnode)
            new_nodes.append(new_node)
            # set new node as next parent node
            new_pnode = new_node
        # continue with the children
        for cnode in node.child_nodes:
            self._add_nodes_to_tree(
                cnode,
                new_pnode,
                new_tree,
                new_nodes,
                name,
            )

    def create_compartment_tree(self, loc_arg):
        """
        Creates a new compartment tree where the provided set of locations
        correspond to the nodes.

        Parameters
        ----------
        loc_arg: list of locations or str
            if list of locations, specifies the locations, if str,
            specifies the name under which the set of location is stored
            that should be used to create the new tree

        Returns
        -------
            `neat.CompartmentTree`
                The compartment tree.
        """
        # process input argument
        if isinstance(loc_arg, list):
            locs = [MorphLoc(loc, self) for loc in loc_arg]
            name = "comp_locs"
            self.store_locs(locs, name=name)
        elif isinstance(loc_arg, str):
            name = loc_arg
            self._try_name(name)
        else:
            raise IOError("`loc_arg` should be list of locs or string")
        nids = self.nids[name]
        xs = self.xs[name]
        # create new tree
        new_tree = CompartmentTree()
        # find the common root of the set of locations
        snode = self.find_common_root(name)
        # check if that root is in set of locations
        possible_loc_idxs = self.get_loc_idxs_on_node(name, snode)
        if len(possible_loc_idxs) > 0:
            # create the new root node
            new_pnode = CompartmentNode(0, loc_idx=int(possible_loc_idxs[0]))
            new_tree.set_root(new_pnode)
            new_nodes = [new_pnode]
            # create other nodes
            for loc_idx in possible_loc_idxs[1:]:
                index = len(new_nodes)
                # make new node
                new_node = CompartmentNode(index, loc_idx=int(loc_idx))
                # add new node
                new_tree.add_node_with_parent(new_node, new_pnode)
                new_nodes.append(new_node)
                # set new node as next parent node
                new_pnode = new_node
        else:
            warnings.warn(
                "Locations of name `"
                + name
                + "` do not define a root - "
                + "adding root to set of locations"
            )
            locs = self.get_locs(name)
            locs = [(snode.index, 1.0)] + locs
            self.store_locs(locs, name=name)
            # create the new root node
            new_pnode = CompartmentNode(0, loc_idx=0)
            new_tree.set_root(new_pnode)
            new_nodes = [new_pnode]
        # make rest of tree
        for cnode in snode.child_nodes:
            self._add_compartment_nodes_to_tree(
                cnode, new_pnode, new_tree, new_nodes, name
            )
        return new_tree

    def _add_compartment_nodes_to_tree(
        self, node, new_pnode, new_tree, new_nodes, name
    ):
        # get the specified locs
        xs = self.xs[name]
        # check which locinds are on the branch
        ninds = self.get_loc_idxs_on_node(name, node)
        order_inds = np.argsort(xs[ninds])
        for loc_idx in np.array(ninds)[order_inds]:
            index = len(new_nodes)
            # make new node
            new_node = CompartmentNode(index, loc_idx=int(loc_idx))
            # add new node
            new_tree.add_node_with_parent(new_node, new_pnode)
            new_nodes.append(new_node)
            # set new node as next parent node
            new_pnode = new_node
        # continue with the children
        for cnode in node.child_nodes:
            self._add_compartment_nodes_to_tree(
                cnode, new_pnode, new_tree, new_nodes, name
            )

    @original_tree_decorator
    def __copy__(self, new_tree=None):
        """
        Fill the ``new_tree`` with it's corresponding nodes in the same
        structure as ``self``, and copies all node variables that both tree
        classes have in common

        Parameters
        ----------
        new_tree: :class:`STree` or derived class (default is ``None``)
            the tree class in which the ``self`` is copied. If ``None``,
            returns a copy of ``self``.

        Returns
        -------
        The new tree instance
        """
        if new_tree is None:
            new_tree = self.__class__()

        super().__copy__(new_tree=new_tree)
        new_tree._original_root = new_tree.root
        try:
            with self.as_computational_tree:
                # set the computational tree
                new_node = new_tree.create_corresponding_node(self.root.index)
                self.root.__copy__(new_node=new_node)
                new_tree._computational_root = new_node
                with new_tree.as_computational_tree:
                    self._recurse_copy(self.root, new_tree)
        except AttributeError:
            # computational tree is not defined
            pass

        return new_tree
