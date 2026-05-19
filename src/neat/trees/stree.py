# -*- coding: utf-8 -*-
#
# stree.py
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

import warnings
import copy
import hashlib
from collections import Counter
from functools import reduce


class SNode(object):
    """
    Simple Node for use with a simple Tree (`neat.STree`)

    Parameters
    ----------
    index: int
        index of the node

    Attributes
    ----------
    index: int
        index of the node
    parent_node: `neat.SNode` or ``None``
        parent of node, ``None`` means node is root
    child_nodes: list of `neat.SNode`
        child nodes of ``self``, empty list means node is leaf
    content: dict
        arbitrary items can be stored at the node
    """

    def __init__(self, index):
        self.index = index
        self._parent_node = None
        self._child_nodes = []
        self._content = {}

    def get_parent_node(self):
        return self._parent_node

    def set_parent_node(self, parent_node):
        self._parent_node = parent_node

    parent_node = property(get_parent_node, set_parent_node)

    def get_child_nodes(self, **kwargs):
        return self._child_nodes

    def set_child_nodes(self, cnodes):
        self._child_nodes = cnodes

    child_nodes = property(get_child_nodes, set_child_nodes)

    def add_child(self, child_node):
        self._child_nodes.append(child_node)

    def get_content(self):
        return self._content

    def set_content(self, content):
        if isinstance(content, dict):
            self._content = content
        else:
            raise Exception("SNode.set_content must receive a dict")

    content = property(get_content, set_content)

    def make_empty(self):
        """
        Remove content and references to parent and child nodes
        """
        self._parent_node = None
        self._content = None
        self._child_nodes = []

    def remove_child(self, child_node):
        """
        Remove a single child node

        Parameters
        ----------
        child_node: `neat.SNode`
            child node to be removed
        """
        self._child_nodes.remove(child_node)

    def __getitem__(self, key):
        """
        Get an item from `self.content`

        Parameters
        ----------
        key : hashable
            dictonary key for the `self.content` dictionary

        Returns
        -------
        value store in `self.content`
        """
        return self._content[key]

    def __setitem__(self, key, value):
        """
        Store an item in the `self.content` dictinary

        Parameters
        ----------
        key : hashable
            dictionary key for the `self.content` dictionary
        value : any
            value to be stored in the dictionary
        """
        self._content[key] = value

    def __str__(self, with_parent=True):
        parent_idx = self.parent_node.index if self.parent_node is not None else -1
        node_str = f"{self.__class__.__name__} {self.index}"
        if with_parent:
            pstr = "None" if self.parent_node is None else str(self.parent_node.index)
            node_str += f", Parent: {pstr}"
        return node_str

    def _get_repr_dict(self):
        """
        Note that dictionaries maintain insertion order from Python 3.7 onwards
        """
        parent_idx = self.parent_node.index if self.parent_node is not None else -1
        return {
            "node index": self.index,
            "parent index": parent_idx,
            "content": repr(self.content),
        }

    def __repr__(self):
        return repr(self._get_repr_dict())

    def __copy__(self, new_node=None):
        """
        experimental, untested
        """
        if new_node is None:
            new_node = self.__class__(self.index)
        orig_keys = set(self.__dict__.keys()) - {"_parent_node", "_child_nodes"}
        copy_keys = orig_keys.intersection(set(new_node.__dict__.keys()))
        for key in copy_keys:
            new_node.__dict__[key] = copy.deepcopy(self.__dict__[key])
        return new_node


class STree(object):
    """
    A simple tree for use with a simple Node (`neat.SNode`).

    Generic implementation of a tree structure as a linked list extended with
    some convenience functions

    Parameters
    ----------
    arg: `neat.SNode` or subclass, `neat.STree` or subclass, or ``None``
        When arg is a `neat.SNode`, it specifies the root of the tree.
        When arg is a `neat.STree`, it constructs a deep copy of the provided tree.
        Default is ``None``, which creates an empty tree.

    Attributes
    ----------
    root: `neat.SNode`
        The root of the tree
    nodes: list of `neat.SNode`
        The nodes in the tree
    leafs: list of `neat.SNode`
        The leaf nodes in the tree
    """

    def __init__(self, arg=None):
        """
        Initialize an empty tree by default
        """
        self._root = None

        if issubclass(type(arg), STree):
            # copy the provided tree into self
            arg.__copy__(new_tree=self)
        elif issubclass(type(arg), SNode) or arg is None:
            self.root = arg
        else:
            raise ValueError(
                f"`arg` should be a node (the root node of the tree), "
                f"a tree, or ``None``. Provided `arg` is {type(arg)}"
            )

    def __getitem__(self, index, **kwargs):
        """
        Returns the node with given index, if no such node is in the tree, None
        is returned.

        Parameters
        ----------
            index: int
                the index of the node to be found

        Returns
        -------
            `neat.SNode` or None
        """
        return self._find_node(self.root, index)

    def _find_node(self, node, index):
        """
        Breadth-first/stack iteration to replace the recursive call.
        Traverses the tree until it finds the node you are looking for.
        Returns SNode when found and None when not found

        Parameters
        ----------
            node: `neat.SNode` (optional)
                node where the search is started
            index: int
                the index of the node to be found

        Returns
        -------
            `neat.SNode`
        """
        stack = []
        stack.append(node)
        while len(stack) != 0:
            for cnode in stack:
                if cnode.index == index:
                    return cnode
                else:
                    stack.remove(cnode)
                    stack.extend(cnode.get_child_nodes())
        return None  # Not found!

    def __len__(self, node=None):
        """
        Return the number of nodes in the tree. If an input node is specified,
        the number of nodes in the subtree of the input node is returned

        Parameters
        ----------
            node: `neat.SNode` (optional)
                The starting node. Defaults to root

        Returns
        -------
            int
        """
        self._node_count = 0
        for node in self.__iter__(node=node):
            self._node_count += 1
        return self._node_count

    def __iter__(self, node=None, **kwargs):
        """
        Iterate over the nodes in the subtree of the given node.

        Beware, if the given node is not in the tree, it will simply iterate
        over the subtree of the given node.

        Parameters
        ----------
            node: `neat.SNode` (optional)
                The starting node. Defaults to the root
        """
        if node is None:
            node = self.root
        if node is not None:
            yield node
            if node is not None:
                for cnode in node.get_child_nodes():
                    for inode in self.__iter__(cnode):
                        yield inode

    def __str__(self, node=None):
        """
        Generate a string of the subtree of the given node.

        Beware, if the given node is not in the tree, it will simply iterate
        over the subtree of the given node.

        Parameters
        ----------
            node: `neat.SNode` (optional)
                The starting node. Defaults to the root
        """
        if node is None:
            node = self.root
        tree_string = f">>> {self.__class__.__name__}"
        for iternode in self.__iter__(node):
            tree_string += "\n    " + iternode.__str__(with_parent=True)
        return tree_string

    def __repr__(self, node=None):
        """
        Generate a representation string of the subtree of the given node.

        Beware, if the given node is not in the tree, it will simply iterate
        over the subtree of the given node.

        Parameters
        ----------
            node: `neat.SNode` (optional)
                The starting node. Defaults to the root
        """
        if node is None:
            node = self.root
        repr_list = [self.__class__.__name__]
        for iternode in self.__iter__(node):
            repr_list.append(repr(iternode))
        return repr(repr_list)

    def __hash__(self):
        """
        Generates an integer hash with that standard python `hash` function
        applied to the representation string

        Returns
        -------
            `int`: the hash
        """
        return hash(repr(self))

    def unique_hash(self):
        """
        Generates a hexadecimal hash based on the hashlib sha25 algorithm
        applied to the representation string

        Returns
        -------
            `str`: the hash string
        """
        h = hashlib.new("sha256")
        h.update(repr(self).encode())

        return h.hexdigest()

    def check_ordered(self):
        """
        Check if the indices of the tree are number in the same order as they
        appear in the iterator
        """
        return list(range(len(self))) == [node.index for node in self]

    def get_nodes(self):
        """
        Build a list of all the nodes in the tree

        Parameters
        ----------
            recompute_flag: bool
                whether or not to re-evaluate the node list

        Returns
        -------
            list of :class:`Snode`
        """
        return self.gather_nodes(self.root)

    def set_nodes(self, illegal):
        raise AttributeError("`nodes` is a read-only attribute")

    nodes = property(get_nodes, set_nodes)

    def gather_nodes(self, node):
        """
        Build a list of all the nodes in the subtree of the provided node

        Parameters
        ----------
            node: :class:`Snode`
                starting point node

        Returns
        -------
            list of :class:`Snode`
        """
        nodes = []
        self._gather_nodes(node, nodes)
        return nodes

    def _gather_nodes(self, node, node_list=[]):
        """
        Append node to list and recurse to its child nodes

        Parameters
        ----------
            node: `neat.SNode`
            node_list: list of `neat.SNode`
        """
        node_list.append(node)
        for cnode in node.child_nodes:
            self._gather_nodes(cnode, node_list=node_list)

    def get_leafs(self):
        """
        Get all leaf nodes in the tree.

        Parameters
        ----------
            recompute_flag: bool
                Whether to force recomputing the leaf list. Defaults to 1.
        """
        return [node for node in self if self.is_leaf(node)]

    def set_leafs(self, illegal):
        raise AttributeError("`leafs` is a read-only attribute")

    leafs = property(get_leafs, set_leafs)

    def set_root(self, node):
        """
        Set the root node of the tree

        Parameters
        ----------
            node: `neat.SNode`
                node to be set as root of the tree
        """
        if node is not None:
            assert issubclass(type(node), SNode)
            node.parent_node = None
        self._root = node

    def get_root(self):
        return self._root

    root = property(get_root, set_root)

    def is_root(self, node):
        """
        Check if input node is root of the tree.

        Parameters
        ----------
            node: `neat.SNode`
        """
        return node.get_parent_node() is None

    def is_leaf(self, node):
        """
        Check if input node is a leaf of the tree

        Parameters
        ----------
            node: `neat.SNode`
        """
        return len(node.get_child_nodes()) == 0

    def create_corresponding_node(self, node_index):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return SNode(node_index)

    def add_node_with_parent_from_index(self, node_index, pnode, *args, **kwargs):
        """
        Create a node with the given index and add it to the tree under a
        specific parent node.

        Parameters
        ----------
            node_index: int
                index of the new node
            pnode: `neat.SNode`
                parent node of the newly added node

        Raises
        ------
            ValueError
                if ``node_index`` is already in the tree
        """
        if self[node_index] == None:
            node = self.create_corresponding_node(node_index, *args, **kwargs)
            self.add_node_with_parent(node, pnode)
        else:
            raise ValueError("Index %d is already exists in the tree.")

    def add_node_with_parent(self, node, pnode):
        """
        Add a node to the tree under a specific parent node

        Parameters
        ----------
            node: `neat.SNode`
                node to be added
            pnode: `neat.SNode`
                parent node of the newly added node
        """
        if pnode is not None:
            node.set_parent_node(pnode)
            pnode.add_child(node)
        else:
            warnings.warn("`pnode` was `None`, did nothing.")

    def soft_remove_node(self, node):
        """
        Remove a node and its subtree from the tree by deleting the reference
        to it in its parent. Internally, the node and its linked subtree are not
        changed

        Parameters
        ----------
            node: `neat.SNode`
                node to be removed
        """
        node.get_parent_node().remove_child(node)

    def remove_node(self, node):
        """
        Remove a node as well as its subtree from the tree

        Parameters
        ----------
            node: `neat.SNode`
                node to be removed
        """
        node.get_parent_node().remove_child(node)
        self._deep_remove(node)

    def _deep_remove(self, node):
        cnodes = node.get_child_nodes()
        node.make_empty()
        for cnode in cnodes:
            self._deep_remove(cnode)

    def remove_single_node(self, node):
        """
        Remove a single node from the tree. The nodes' children become the
        children of the nodes' parent.

        Parameters
        ----------
            node: `neat.SNode`
                node to be removed
        """
        if node == self.root:
            raise ValueError("Removing root is forbidden")
        cnodes = node.get_child_nodes()
        pnode = node.get_parent_node()
        pnode.remove_child(node)
        for cnode in cnodes:
            cnode.set_parent_node(pnode)
            pnode.add_child(cnode)

    def insert_node(self, node, pnode, pcnodes=[]):
        """
        Insert a node in the tree as a child of a specified parent. The
        original children of the parent that will become children of the node
        are specified in the ``pcnodes`` list

        Parameters
        ----------
            node: `neat.SNode`
                the node that is to be inserted
            pnode: `neat.SNode`
                the node that will become parent of the node that is to be
                inserted
            pcnodes: list of `neat.SNode`
                the current children of the pnode that will become children of
                the node
        """
        if pnode != None:
            cnodes = pnode.get_child_nodes()
            for pcnode in pcnodes:
                if pcnode in cnodes:
                    ind = cnodes.index(pcnode)
                    del cnodes[ind]
                    node.add_child(pcnode)
                    pcnode.set_parent_node(node)
                else:
                    warnings.warn(
                        str(pcnode)
                        + " is not a child of "
                        + str(pnode)
                        + ", ignoring it"
                    )
            node.set_parent_node(pnode)
            pnode.add_child(node)
        if pnode == None:
            cnode = self.root
            cnode.set_parent_node(node)
            node.set_parent_node(None)
            node.add_child(cnode)
            self.root = node

    def reset_indices(self, n=0):
        """
        Resets the indices in the order they appear in a depth-first iteration
        """
        for ind, node in enumerate(self):
            node.index = ind + n

    def get_sub_tree(self, node, new_tree=None):
        """
        Get the subtree of the specified node. The root of the subtree is a new
        node with the same children as the original node, but None instead of a
        parent.

        Parameters
        ----------
            node: `neat.SNode`
                root of the sub tree
            new_tree: `neat.STree` or derived class
                the type of tree in which the nodes of the subtree are to be
                copied

        Returns
        -------
        `neat.STree`
            Subtree of with ``node`` as root
        """
        if new_tree is None:
            new_tree = self.__class__()

        new_node = new_tree.create_corresponding_node(node.index)
        node.__copy__(new_node=new_node)
        new_node.set_parent_node(None)
        new_tree.set_root(new_node)

        self._recurse_copy(node, new_tree)

        return new_tree

    def depth_of_node(self, node):
        """
        compute the depth of the node (number of edges between node and root)

        Parameters
        ----------
        node: `neat.SNode`

        Returns
        -------
        int
            depth of the node
        """
        return len(self.path_to_root(node)) - 1

    def degree_of_node(self, node):
        """
        Compute the degree (number of leafs in its subtree) of a node.

        Parameters
        ----------
            node: `neat.SNode`
        """
        return len([node for node in self.__iter__(node) if self.is_leaf(node)])

    def order_of_node(self, node):
        """
        Compute the order (number of bifurcations from the root) of a node.

        Parameters
        ----------
            node: `neat.SNode`
        """
        ptr = self.path_to_root(node)
        order = 0
        for node in ptr:
            if len(node.child_nodes) > 1:
                order += 1
        # order is on [0,max_order] thus subtract 1 from this calculation
        return order - 1

    def path_to_root(self, node):
        """
        Return the path from a given node to the root

        Parameters:
            node: `neat.SNode`

        Returns
        -------
            list of `neat.SNode`
                List of nodes from ``node`` to root. First node is the input node
                and last node is the root
        """
        nodes = []
        self._go_to_root_from(node, nodes)
        return nodes

    def _go_to_root_from(self, node, nodes):
        nodes.append(node)
        pnode = node.get_parent_node()
        if pnode != None:
            self._go_to_root_from(pnode, nodes)

    def path_between_nodes(self, from_node, to_node):
        """
        Inclusive path from ``from_node`` to ``to_node``.

        Parameters
        ----------
            from_node: `neat.SNode`
            to_node: `neat.SNode`

        Returns
        -------
            list of `neat.SNode`
                List of nodes representing the direct path between ``from_node``
                and ``to_node``, which are respectively the first and last nodes
                in the list.
        """
        path1 = self.path_to_root(from_node)[::-1]
        path2 = self.path_to_root(to_node)[::-1]
        path = path1 if len(path1) < len(path2) else path2
        ind = next((ii for ii in range(len(path)) if path1[ii] != path2[ii]), len(path))
        return path1[ind:][::-1] + path2[ind - 1 :]

    def path_between_nodes_depth_first(self, from_node, to_node):
        """
        Inclusive path from ``from_node`` to ``to_node``, ginven in a depth-
        first ordering.

        Parameters
        ----------
            from_node: `neat.SNode`
            to_node: `neat.SNode`

        Returns
        -------
            list of `neat.SNode`
                List of nodes representing the direct path between ``from_node``
                and ``to_node``, which are respectively the first and last nodes
                in the list.
        """
        path1 = self.path_to_root(from_node)[::-1]
        path2 = self.path_to_root(to_node)[::-1]
        path = path1 if len(path1) < len(path2) else path2
        ind = next((ii for ii in range(len(path)) if path1[ii] != path2[ii]), len(path))
        return path1[ind - 1 :] + path2[ind:]

    def get_nodes_in_subtree(self, ref_node, subtree_root=None):
        """
        Returns the nodes in the subtree that contains the given reference nodes
        and has the given subtree root as root. If the subtree root is not
        provided, the subtree of the first child node of the root on the path to
        the reference node is given (plus the root)

        Parameters
        ----------
            ref_node: `neat.SNode`
                the reference node that is in the subtree
            subtree_root: `neat.SNode`
                what is to be the root of the subtree. If this node is not on
                the path from reference node to root, a ValueError is raised

        Returns
        -------
            list of `neat.SNode`
                List of all nodes in the subtree. It's root is in the first
                position
        """
        if subtree_root == None:
            subtree_root = self.root
        ref_path = self.path_between_nodes(ref_node, subtree_root)
        if subtree_root in ref_path:
            if len(ref_path) > 1:
                subtree_nodes = [subtree_root] + self.gather_nodes(ref_path[-2])
            else:
                subtree_nodes = [ref_node]  # both input nodes are the same
        else:
            raise ValueError("|subtree_root| not in path from |ref_node| \
                                root")
        return subtree_nodes

    def sister_leafs(self, node):
        """
        Find the leafs that are in the subtree of the nearest bifurcation node
        on the path from input node to root.

        Parameters
        ----------
            node: `neat.SNode`
                Starting node for search

        Returns
        -------
        node: `neat.SNode`
            the bifurcation node
        sister_leafs: list of `neat.SNode`
            The first element is the input node. The others are the leafs
            of the subtree emanating from the bifurcation node that are not
            in the subtree from the input node.
        corresponding_children: list of `neat.SNode`
            The children of the bifurcation node. If the number of leafs
            ``sister_leafs`` is the same as the number of
            ``corresponding_children``, the subtree of each element of
            ``corresponding_children`` has exactly one leaf, the corresponding
            element in ``sister_leafs``
        """
        sleafs = [node]
        cchildren = []
        snode = self._go_to_root_until(node, None, sl=sleafs, cc=cchildren)
        return snode, sleafs, cchildren

    def _go_to_root_until(self, node, c_node, sl=[], cc=[]):
        c_nodes = node.get_child_nodes()
        if (c_node != None and len(c_nodes) > 1) or self.is_root(node):
            cc.append(c_node)
            for c_node_ in set(c_nodes) - {c_node}:
                self._go_from_root_until(c_node_, sl=sl)
                cc.append(c_node_)
        else:
            p_node = node.get_parent_node()
            node = self._go_to_root_until(p_node, node, sl=sl, cc=cc)
        return node

    def _go_from_root_until(self, node, sl=[]):
        c_nodes = node.get_child_nodes()
        if len(c_nodes) == 0:
            sl.append(node)
        else:
            for c_node in c_nodes:
                self._go_from_root_until(c_node, sl=sl)

    def find_bifurcation_node_to_root(self, node, cnode=None):
        """
        Find the nearest bifurcation node towards root from the input node.

        Parameters
        ----------
        node: `neat.SNode`
            Starting node for search
        cnode: `neat.SNode`
            For recursion, don't change default

        Returns
        -------
        node: `neat.SNode`
            the bifurcation node
        cnode: `neat.SNode`
            The bifurcation node's child on the path to the input node.

        """
        if cnode == None or len(node.get_child_nodes()) <= 1:
            pnode = node.get_parent_node()
            if pnode != None:
                node, cnode = self.find_bifurcation_node_to_root(pnode, cnode=node)
        return node, cnode

    def find_bifurcation_node_from_root(self, node):
        """
        Find the nearest bifurcation node towards leaf from the input node.

        Parameters
        ----------
        node: `neat.SNode`
            Starting node for search

        Returns
        -------
        node: `neat.SNode`
            the bifurcation node
        """
        if len(node.child_nodes) > 1:
            return node
        elif len(node.child_nodes) == 0:
            return None
        else:
            return self.find_bifurcation_node_from_root(node.child_nodes[0])

    def find_in_between_bifurcation_nodes(self, nodes):
        """
        Get the bifurcation nodes in bewteen the provided input nodes

        Parameters
        ----------
        nodes: list of `neat.SNode`
            the input nodes

        Returns
        -------
        list of `neat.SNode`
            the bifurcation nodes
        """
        # unique nodes
        nodes = reduce(lambda l, x: l.append(x) or l if x not in l else l, nodes, [])
        # tag all nodes
        for node in self:
            node["tag"] = 0
        # find the 'leafs' within the list of nodes (i.e. most centripetal nodes)
        pnodes = []
        for node in nodes:
            pnodes.extend([n for n in self.path_to_root(node)])
        pcount = Counter(pnodes)
        nodes = [node for node in nodes if pcount[node] == 1]
        # for each node, find nearest bifurcation towards root
        pnodes = []
        for node in nodes:
            pnodes.extend([n for n in self.path_to_root(node)])
        for pathnode in pnodes:
            pathnode["tag"] += 1
        # find the bifurcation nodes
        bifur_nodes = [self.root]
        for node in self:
            # !!! only works for bifurcations with two children
            # TODO: extend for all bifurcations
            if len(node.child_nodes) > 1 and not np.any(
                [cn["tag"] == 0 for cn in node.child_nodes]
            ):
                bifur_nodes.append(node)
        # remove the tags
        for node in self:
            del node.content["tag"]
        # add root if necessary
        if self.root not in bifur_nodes:
            bifur_nodes = [self.root] + bifur_nodes

        return bifur_nodes

    def get_nearest_neighbours(self, node, nodes):
        """
        Find the nearest neighbours of ``node`` in ``nodes``. If ``nodes`` contains
        ``node``, it is excluded from the search.

        When a node in the up-direction is a bifurcation node and in ``nodes``, nodes
        in its other subtree are excluded from the search

        !!! Untested

        Parameters
        ----------
        node: `neat.SNode`
            node for which the nearest neighbours are sought
        nodes: list of `neat.SNode`
            list in which nearest neighbours of ``node`` are sought
        """
        nns = []
        self._search_nn_to_root(node, nodes, nns)
        self._search_nn_from_root(node, nodes, nns)
        return nns

    def _search_nn_to_root(self, node, nodes, nns):
        p_node = node.parent_node
        if p_node is not None:
            # up direction
            if p_node in nodes:
                nns.append(p_node)
            else:
                self._search_nn_to_root(p_node, nodes, nns)
                # down direction
                for c_node in set(p_node.child_nodes) - {node}:
                    self._search_nn_from_root(c_node, nodes, nns)

    def _search_nn_from_root(self, node, nodes, nns):
        for c_node in node.child_nodes:
            if c_node in nodes:
                nns.append(c_node)
            else:
                self._search_nn_from_root(c_node, nodes, nns)

    def __copy__(self, new_tree=None):
        """
        Fill the ``new_tree`` with it's corresponding nodes in the same
        structure as ``self``, and copies all node variables that both tree
        classes have in common

        Parameters
        ----------
        new_tree: `neat.STree` or derived class (default is ``None``)
            the tree class in which the ``self`` is copied. If ``None``,
            returns a copy of ``self``.

        Returns
        -------
        The new tree instance
        """
        if new_tree is None:
            new_tree = self.__class__()

        # copy all attributes not related to tree structure
        orig_keys = set(self.__dict__.keys())
        copy_keys = orig_keys.intersection(set(new_tree.__dict__.keys()))
        for key in copy_keys:
            if key not in ["root", "_root", "_computational_root", "_original_root"]:
                new_tree.__dict__[key] = copy.deepcopy(self.__dict__[key])

        # copy the tree structure
        if self.root is not None:
            new_node = new_tree.create_corresponding_node(self.root.index)
            self.root.__copy__(new_node=new_node)
            new_tree.set_root(new_node)
            self._recurse_copy(self.root, new_tree)

        return new_tree

    def _recurse_copy(self, pnode, new_tree):
        for node in pnode.get_child_nodes(skip_inds=[]):
            new_node = new_tree.create_corresponding_node(node.index)
            new_node = node.__copy__(new_node=new_node)
            new_tree.add_node_with_parent(
                new_node, new_tree.__getitem__(pnode.index, skip_inds=[])
            )
            self._recurse_copy(node, new_tree)
