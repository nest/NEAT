"""
File contains:

    - :class:`Kernel`
    - :class:`NETNode`
    - :class:`NET`

Author: W. Wybo
"""


import numpy as np
import matplotlib.pyplot as pl

from .stree import STree, SNode

import copy
import warnings


class Kernel(object):
    """
    Implements a kernel as a superposition of exponentials:

    .. math:: k(t) = \sum_n c_n e^{ - a_n t}

    Kernels can be added and subtracted, as this class overloads the __add__
    and __subtract__ functions.

    They can be evaluated as a function of time by calling the object with a
    time array.

    They can be evaluated in the Fourrier domain with `Kernel.ft`

    Parameters
    ----------
    kernel: dict, float, neat.Kernel, tuple or list
        If dict, has the form {'a': `np.array`, 'c': `np.array`}.
        If float, sets `c` single exponential prefactor and assumes `a` is 1 kHz.
        If `neat.Kernel`, copies the object.
        If tuple or list, sets 'a' as first element and 'c' as last element.

    Attributes
    ----------
    a: np.array of float or complex
        The exponential coefficients (kHz)
    c: np.array of float or complex
        The exponential prefactors
    k_bar: float
        The total surface area under the kernel
    """
    def __init__(self, kernel):
        # set kernel time scales and exponential prefactors
        if isinstance(kernel, dict):
            self.a = copy.deepcopy(kernel['a'])
            self.c = copy.deepcopy(kernel['c'])
        elif isinstance(kernel, float) or isinstance(kernel, int):
            self.a = np.array([1.])
            self.c = np.array([kernel]).astype(float)
        elif isinstance(kernel, Kernel):
            self.a = copy.deepcopy(kernel.a)
            self.c = copy.deepcopy(kernel.c)
        else:
            self.a = copy.deepcopy(kernel[0])
            self.c = copy.deepcopy(kernel[1])
        if isinstance(self.a, float):
            self.a = np.array([self.a])
        elif not isinstance(self.a, np.ndarray):
            self.a = np.array(self.a)
        if isinstance(self.c, float):
            self.c = np.array([self.c])
        elif not isinstance(self.c, np.ndarray):
            self.c = np.array(self.c)

    def __getitem__(self, ind):
        if ind == 0: return self.a
        elif ind == 1: return self.c
        elif ind == 'a': return self.a
        elif ind == 'c': return self.c
        elif ind == 'alphas': return self.a
        elif ind == 'gammas': return self.c
        else: raise IndexError('Index should be \'0\' or \'1\'')

    def __call__(self, t_arr):
        return np.dot(np.exp(-t_arr[:,np.newaxis] * self.a[np.newaxis,:]), \
                      self.c[:,np.newaxis]).flatten().real

    def __add__(self, kernel):
        if kernel.a.shape[0] == self.a.shape[0] and \
           np.allclose(kernel.a, self.a):
            a = copy.copy(self.a)
            c = kernel.c + self.c
        else:
            a = np.concatenate((self.a, kernel.a))
            c = np.concatenate((self.c, kernel.c))
        return Kernel((a, c))

    def __sub__(self, kernel):
        if kernel.a.shape[0] == self.a.shape[0] and \
           np.allclose(kernel.a, self.a):
            a = copy.copy(self.a)
            c = self.c - kernel.c
        else:
            a = np.concatenate((self.a, kernel.a))
            c = np.concatenate((self.c, -kernel.c))
        return Kernel((a, c))

    def get_k_bar(self):
        """
        The total surface under the kernel
        """
        return np.sum(self.c / self.a).real

    def set_k_bar(self, kk):
        raise AttributeError('`k_bar` is a read-only attribute, adjust attribute `c` ' + \
                             'by multiplying with a factor to change `k_bar`')
    k_bar = property(get_k_bar, set_k_bar)

    def __str__(self, as_timescale=False):
        if as_timescale:
            return 't = ' + np.array2string(1./self.a, precision=4, max_line_width=1000) + '\n' + \
                   'c = ' + np.array2string(self.c, precision=4, max_line_width=1000)
        else:
            return 'a = ' + np.array2string(self.a, precision=4, max_line_width=1000) + '\n' + \
                   'c = ' + np.array2string(self.c, precision=4, max_line_width=1000)

    def __repr__(self):
        return repr({'a': self.a, 'c': self.c})

    def t(self, t_arr):
        """
        Evaluates the kernel in the time domain

        Parameters
        ----------
        t_arr: `np.array` of `float`
            the time array in ``ms`` at which the kernel is evaluated

        Returns
        -------
        np.array of float
            the temporal kernel
        """
        return self(t_arr)

    def diff(self, t_arr=None):
        """
        Computes the time derivative of the kernel. If a time array is provided,
        returns an array of corresponding kernel values. If nothing is provided,
        returns a kernel representing the time derivative.

        Parameters
        ----------
        t_arr: `np.array` (optional)
            the time array

        Returns
        -------
        `np.array` or `neat.Kernel`
            the differentiated kernel
        """
        if t_arr is None:
            return Kernel({
                'a': self.a,
                'c': -self.a * self.c,
            })
        else:
            return np.dot(
                -self.a[np.newaxis,:] * np.exp(-t_arr[:,np.newaxis] * self.a[np.newaxis,:]), \
                self.c[:,np.newaxis]
            ).flatten().real

    def ft(self, s_arr):
        """
        Evaluates the kernel in the Fourrier domain

        Parameters
        ----------
        s_arr: np.array of complex
            The frequencies in ``Hz`` at which the kernel is to be evaluated

        Returns
        -------
        np.array of complex
            The Fourrier transform of the kernel

        """
        return np.sum(self.c[:,None]*1e3 / (self.a[:,None]*1e3 + s_arr[None,:]), 0)

    def fit_c(self, t_arr, func_arr, w=None):
        """
        Perform a linear least squares fit of the exponential prefactors in the
        time domain

        Parameters
        ----------
        t_arr: `np.array` of float
            the time array in ``ms`` at which the kernel is evaluated
        k_arr: `np.array` of float
            the to be fitted kernel array

        Returns
        -------
        `np.ndarray` of float (`ndim = 2`)
            The feature matrix
        """
        if w is None:
            w = np.ones_like(t_arr)
        A = np.exp(-t_arr[:,None] * self.a[None,:])

        self.c = np.linalg.lstsq(w[:,None]*A, w*func_arr, rcond=None)[0]

class NETNode(SNode):
    """
    Node associated with `neat.NET`.

    Attributes
    ----------
    loc_idxs: list of int
        The inidices of locations which the node integrates
    newloc_idxs: list of int
        The locations for which the node is the most local component to integrate
        them
    z_kernel: `neat.Kernel`
        The impedance kernel with which the node integrates inputs
    z_bar: float
        The steady state impedance associated with the impedance kernel
    """
    def __init__(self, index, loc_idxs, newloc_idxs=[], z_kernel=None):
        super().__init__(index)
        # location indices that node integrates
        self.loc_idxs = loc_idxs
        self.newloc_idxs = newloc_idxs
        # kernel associated with node
        self.z_kernel = z_kernel

    def __str__(self):
        if self.parent_node is not None:
            return 'NETNode ' + str(self.index) + \
                    ', loc inds: ' + str(self.loc_idxs) + \
                    ', newloc inds: ' + str(self.newloc_idxs) + \
                    ', parent: ' + str(self.parent_node.index) + \
                    ', z_bar (MOhm) = ' + str(self.z_bar)
        else:
            return 'NETNode ' + str(self.index) + \
                    ', loc inds: ' + str(self.loc_idxs) + \
                    ', newloc inds: ' + str(self.newloc_idxs) + \
                    ', parent: None' \
                    ', z_bar (MOhm) = ' + str(self.z_bar)

    def set_z_kernel(self, z_kernel):
        self._z_kernel = Kernel(z_kernel)

    def get_z_kernel(self):
        return self._z_kernel

    def get_z(self):
        return self._z_kernel.k_bar

    z_kernel = property(get_z_kernel, set_z_kernel)
    z_bar = property(get_z, set_z_kernel)

    def __contains__(self, loc_idx):
        return loc_idx in self.loc_idxs

    def _set_compartment_data(self, node_list, z_root_list, z_comp_list, Iz=5.):
        node_inds  = [node.index for node in node_list if node != None]
        z_root = np.array(z_root_list)
        z_comp = np.array(z_comp_list)
        comp_inds = np.where(z_comp / z_root > Iz)[0]
        # store the relevant quantities
        self._z_root = z_root[comp_inds]
        self._z_comp = z_comp[comp_inds]
        self._node_inds = [node_inds[ind] for ind in comp_inds]

    def _set_tentative_compartments(self, comps):
        self._comps = comps

    def _set_shared_root_idx(self, ind):
        self._root_ind = self._node_inds.index(ind)

    def __str__(self, with_parent=True, with_morph_info=False):
        node_str = super().__str__(with_parent=with_parent)

        node_str += f' --- ' \
            f' loc inds: {str(self.loc_idxs)}' \
            f', newloc inds: {str(self.newloc_idxs)}' \
            f', z_bar = {self.z_bar} MOhm'

        return node_str

    def _get_repr_dict(self):
        repr_dict = super()._get_repr_dict()
        repr_dict.update({
            "loc_idxs": self.loc_idxs,
            "newloc_idxs": self.newloc_idxs,
            "z_kernel": repr(self.z_kernel)
        })
        return repr_dict

    def __repr__(self):
        return repr(self._get_repr_dict())


class NET(STree):
    """
    Abstract tree class that implements the Neural Evaluation Tree
    (Wybo et al., 2019), representing the spatial voltage as a number of voltage
    components present at different spatial scales.
    """
    def __init__(self, root=None):
        super().__init__(root)

    def create_corresponding_node(self, node_index):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return NETNode(node_index, [])

    def get_loc_idxs(self, sroot=None):
        """
        Get the indices of the locations a subtree integrates

        Parameters
        ----------
        sroot: `neat.NETNode`, int or None
            Root of the subtree, or index of the root. If ``None``, subtree is
            the whole tree.

        Returns
        -------
        loc_idxs: indices of locations
        """
        if isinstance(sroot, int):
            sroot = self[sroot]
        elif sroot is None:
            sroot = self.root
        return sroot.loc_idxs

    def get_leaf_loc_node(self, loc_idx):
        """
        Get the node for which ``loc_idx`` is a new location

        Parameters
        ----------
        loc_idx: int
            index of the location

        Returns
        -------
        :obj:`NETNode`
        """
        for node in self:
            if loc_idx in node.newloc_idxs:
                return node

    def set_new_loc_idxs(self):
        """
        Set the new location indices in a tree
        """
        for node in self:
            cloc_idxs = set()
            for cnode in node.child_nodes:
                cloc_idxs = cloc_idxs.union(set(cnode.loc_idxs))
            node.newloc_idxs = list(set(node.loc_idxs) - cloc_idxs)

    def get_reduced_tree(self, loc_idxs, indexing='NET eval'):
        """
        Construct a reduced tree where only the locations index by ``loc_idxs''
        are retained

        Parameters
        ----------
        loc_idxs : iterable of ints
            the indices of the locations that are to be retained
        indexing : 'NET eval' or 'locs'
            if 'NET eval', indexing of ``NETNode.loc_idxs`` will be taken to be the
            indices of locations for which the full NET is evaluated. Otherwise
            will be indices of the input ``loc_idxs``
        """
        loc_idxs_newtree = list({loc_idx for loc_idx in loc_idxs \
                                         if loc_idx in self.root})
        if loc_idxs_newtree:
            new_root = NETNode(0, loc_idxs_newtree,
                                z_kernel=self.root.z_kernel)
            new_tree = NET(new_root)
            for cnode in self.root.child_nodes:
                if cnode is not None:
                    self._construct_reduced_tree(cnode, loc_idxs_newtree,
                                                new_root, new_tree)
            new_tree.set_new_loc_idxs()
            if indexing == 'NET eval':
                return new_tree
            else:
                for node in new_tree:
                    # node.loc_idxs = [np.where(loc_idxs == ind)[0][0] for ind in node.loc_idxs]
                    # node.loc_idxs = sum([np.where(loc_idxs == ind)[0].tolist() for ind in set(node.loc_idxs)], [])
                    node.loc_idxs = sum([np.where(loc_idxs == ind)[0].tolist() for ind in node.loc_idxs], [])
                new_tree.set_new_loc_idxs()
                return new_tree
        else:
            return None

    def _construct_reduced_tree(self, node, loc_idxs, node_newtree, new_tree):
        loc_idxs_subtree = list({loc_idx for loc_idx in loc_idxs \
                                         if loc_idx in node})
        if len(loc_idxs_subtree) > 0:
            if loc_idxs_subtree == loc_idxs:
                node_newtree.z_kernel += node.z_kernel
            else:
                newnode_newtree = NETNode(len(new_tree), loc_idxs_subtree,
                                            z_kernel=node.z_kernel)
                new_tree.add_node_with_parent(newnode_newtree, node_newtree)
                node_newtree = newnode_newtree
            for cnode in node.child_nodes:
                if cnode is not None:
                    self._construct_reduced_tree(cnode, loc_idxs_subtree,
                                                node_newtree, new_tree)

    # def matchInputImpedance(self, z_input):
    #     assert imp_mat.shape[0] == imp_mat.shape[1]
    #     assert imp_mat.shape[0] == len(self.root.loc_idxs)
    #     for node in self:
    #         if self.is_leaf(node):
    #             if len(node.loc_idxs) == 1:
    #                 p_imp = self.calc_total_impedance(node.parent_node)
    #                 node.z_kernel.c *= (z_input[node.locs_inds[0]] - p_imp) / node.z_kernel.k_bar
    #             else:
    #                 for loc_idx in node.loc_idxs:
    #                     new_node = NETNode(len(tree), [loc_idx])
    #                     self.add_node_with_parent

    def calc_total_impedance(self, node):
        """
        Compute the total impedance associated with a node. I.e. the sum of all
        impedances on the path from node to root

        Parameters
        ----------
        node : :class:`SNode`

        Returns
        -------
        float
            total impedance
        """
        return np.sum([node_.z_bar for node_ in self.path_to_root(node)])

    def calc_total_kernel(self, node):
        """
        Compute the total impedance kernel associated with a node. I.e. the sum
        of all impedance kernels on the path from node to root

        Parameters
        ----------
        node : :class:`SNode`

        Returns
        -------
        :class:`Kernel`
        """
        z_k = copy.deepcopy(node.z_kernel)
        if node.parent_node is not None:
            for pn in self.path_to_root(node.parent_node):
                z_k += pn.z_kernel
        return z_k

    def calc_i_z(self, loc_idxs):
        """
        compute I_Z between any pair of locations in ``loc_idxs``

        Parameters
        ----------
        loc_idxs : iterable of ints
            the indices of locations between which I_Z has to be evaluated

        Returns
        -------
        float or dict of tuple : float
            Returns a float if the number of location indices is two, otherwise
            a dictionary with location pairs (smallest is listed first) as keys
            and I_Z values as values
        """
        Iz_dict = {}
        for ii, loc_idx0 in enumerate(loc_idxs):
            for jj, loc_idx1 in enumerate(loc_idxs):
                if jj < ii:
                    net_red = self.get_reduced_tree([loc_idx0, loc_idx1])
                    key = (loc_idx0, loc_idx1) if loc_idx0 < loc_idx1 \
                                               else (loc_idx1, loc_idx0)
                    n0 = net_red.get_leaf_loc_node(loc_idx0)
                    z0 = n0.z_bar if n0 != net_red.root else 0.
                    n1 = net_red.get_leaf_loc_node(loc_idx1)
                    z1 = n1.z_bar if n1 != net_red.root else 0.
                    Iz_dict[key] = (z0 + z1) / (2. * net_red.root.z_bar)
                else:
                    break
        if len(loc_idxs) == 2:
            return list(Iz_dict.values())[0]
        else:
            return Iz_dict

    def calc_i_z_matrix(self):
        """
        compute the Iz matrix for all locations present in the tree

        Returns
        -------
        np.ndarray of float
            The Iz matrix
        """
        z_mat = self.calc_impedance_matrix()
        z_in = np.diag(z_mat)
        return (z_in[:,np.newaxis] + z_in[np.newaxis,:]) / (2. * z_mat) - 1.

    def calc_impedance_matrix(self):
        """
        Compute the impedance matrix approximation associated with the NET

        Returns
        -------
        np.ndarray (ndim = 2)
            the impedance matrix approximation
        """
        n_loc = len(self.root.loc_idxs)
        loc_map = {loc_idx: map_ind for map_ind, loc_idx in enumerate(self.root.loc_idxs)}
        z_mat = np.zeros((n_loc, n_loc))
        self._add_node_to_imp_mat(self.root, z_mat, loc_map)
        return z_mat

    def _add_node_to_imp_mat(self, node, z_mat, loc_map):
        inds = np.array([loc_map[loc_idx] for loc_idx in node.loc_idxs])
        z_mat[np.tile(inds, len(inds)), np.repeat(inds, len(inds))] += node.z_bar
        for cnode in node.child_nodes:
            self._add_node_to_imp_mat(cnode, z_mat, loc_map)

    def calc_compartmentalization(self, Iz, returntype='node index'):
        """
        Returns a compartmentalization for the NET tree where each pair of
        compartments is separated by an Iz of at least ``Iz``. The
        compartmentalization is coded as a list of list, each sublist representing
        a the nodes closest to the root associated with the compartment.

        Parameters
        ----------
        Iz : float
            the minimum Iz separating the compartments
        returntype: str ('node index', 'node')
            either returns the node indices or the node objects

        Returns
        -------
        list of lists
            the compartments
        """
        self._compute_tentative_compartments(Iz=Iz)
        # determine the nodes that contain the eventual compartments and
        # remove the rest
        net = copy.deepcopy(self)
        self._remove_non_compartments(net.leafs, net=net)
        # get the compartment nodes
        comp_nodes = self._set_compartments_leafbased(net.leafs, net)
        if returntype == 'node index':
            comp_inds  = []
            for node in comp_nodes:
                inds = node._comps[node._root_ind]
                comp_inds.append(inds)
            return comp_inds
        elif returntype == 'node':
            comp_nodes_ = []
            for node in comp_nodes:
                inds = node._comps[node.rootind]
                comp_nodes_.append([self[ind] for ind in inds])
            return comp_nodes_

    def _set_compartments_leafbased(self, leafs, net):
        comp_nodes = []
        for ii, leaf in enumerate(leafs):
            root, _, _ = net.sister_leafs(leaf)
            new_leaf = leaf
            comp_bool = False
            while root.index in new_leaf._node_inds:
                comp_bool = True
                old_leaf = new_leaf
                new_leaf = old_leaf.parent_node
            if comp_bool:
                # mark the old_leaf as the compartment indexing node
                old_leaf._set_shared_root_idx(root.index)
                comp_nodes.append(old_leaf)
        return comp_nodes

    def _remove_non_compartments(self, leafs, net=None, n_count=0):
        if net is None:
            warnings.warn('Modifying original tree')
            net = self
        # count number of leafs
        n_leaf = len(leafs)
        leaf = leafs[0]
        # shuffle list
        del leafs[0]
        leafs = leafs + [leaf]
        # leaf is not highest order
        common_root, sister_leafs, corresponding_children = net.sister_leafs(leaf)
        if common_root.index == 0:
            pass
        if len(sister_leafs) == len(corresponding_children):
            # find the compartments with maximal size and closest to common root
            sleafs_comp = []
            sinds_comp = []
            for ii, leaf in enumerate(sister_leafs):
                newleaf = leaf
                comp_bool = False
                while common_root.index in newleaf._node_inds:
                    comp_bool = True
                    oldleaf = newleaf
                    newleaf = oldleaf.parent_node
                if comp_bool:
                    sinds_comp.append(ii)
                    sleafs_comp.append(oldleaf)
            # delete the leafs that are not in compartments
            if len(sleafs_comp) <= 1 and not net.is_root(common_root):
                # if at most one is compartment, we retain only the largest one
                ind = np.argmax([self.calc_total_impedance(node) \
                                 for node in sister_leafs])
                newleaf = sister_leafs[ind]
                for ii, cnode in enumerate(corresponding_children):
                    if ii != ind:
                        net.soft_remove_node(cnode)
                        leafs.remove(sister_leafs[ii])
            else:
                # if more can be compartments, we retain all those
                for ii, cnode in enumerate(corresponding_children):
                    if not ii in sinds_comp:
                        net.soft_remove_node(cnode)
                        leafs.remove(sister_leafs[ii])
            if n_leaf != len(leafs) and len(leafs) > 0:
                self._remove_non_compartments(leafs, net=net, n_count=0)
            elif n_count < len(leafs):
                self._remove_non_compartments(leafs, net=net, n_count=n_count+1)
        elif n_count < len(leafs) and len(leafs) > 0:
            self._remove_non_compartments(leafs, net=net, n_count=n_count+1)

    def _compute_tentative_compartments(self, Iz=5.):
        # set the prerequisite impedances
        self._set_compartment_info(Iz=Iz)
        # set the tentative compartments
        for node in self:
            self._set_compartments_relative(node)

    def _set_compartment_info(self, Iz=5., node=None, z_p=0.,
                        node_list=[], z_root_list=[], z_comp_list=[]):
        if node != None:
            # list of dependent impedances
            try:
                z_root_list.append(z_root_list[-1] + z_p )
            except IndexError:
                z_root_list.append(z_p)
            # list of independent impedances
            z_comp_list.append(0.)
            z_comp_list = [node.z_bar + z_c for z_c in z_comp_list]
            # list or nodes
            node_list.append(node.parent_node)
            # store the compartment information
            node._set_compartment_data(node_list, z_root_list, z_comp_list, Iz=Iz)
        else:
            node = self.root
            # compute node impedance
            self.root._set_compartment_data([], [], [], Iz=0.)
        # recurse to child nodes
        for cnode in node.child_nodes:
            self._set_compartment_info(Iz=Iz, node=cnode, z_p=node.z_bar,
                                node_list=copy.copy(node_list),
                                z_root_list=copy.copy(z_root_list),
                                z_comp_list=copy.copy(z_comp_list))

    def _set_compartments_relative(self, node):
        z_target = node._z_comp
        node_comps = []
        for z_t in z_target:
            comp = [node.index]
            node_comps.append(comp)
        node._set_tentative_compartments(node_comps)

    def compute_cond_rescale(self, gs):
        assert len(gs) == len(self.root.loc_idxs)
        # array for storing shunt factors
        sfs = np.ones_like(gs)
        # counter for recursion algorithm
        for node in self:
            node.counter = 0
        # recursive algorithm to compute shunt factors
        self._sweep(self.leafs[0], self.leafs[1:], sfs, gs)
        # clean
        for node in self:
            node.counter = 0

        return sfs

    def _sweep(self, node, leafs, sfs, gs):
        node.counter += 1
        if node.counter >= len(node.child_nodes):
            if not self.is_root(node):
                # compute the rescaled shunt factors
                denom = 1. + node.z_bar * np.sum(sfs[node.loc_idxs] * gs[node.loc_idxs])
                sfs[node.loc_idxs] = sfs[node.loc_idxs] / denom
                # further recursion
                self._sweep(node.parent_node, leafs, sfs, gs)
        else:
            self._sweep(leafs[0], leafs[1:], sfs, gs)

    def improve_input_resistance(self, z_mat):
        nmaxind = np.max([n.index for n in self])
        for node in self.get_nodes():
            if len(node.loc_idxs) == 1:
                ind = node.loc_idxs[0]
                # recompute the kernel of this single loc layer
                if node.parent_node is not None:
                    p_k = self.calc_total_kernel(node.parent_node)
                else:
                    p_k = Kernel((node.z_kernel.a, np.zeros_like(node.z_kernel.a)))
                f_z = (z_mat[ind,ind] - p_k.k_bar) / node.z_bar
                node.z_kernel.c *= f_z
            elif len(node.newloc_idxs) > 0:
                z_k_approx = self.calc_total_kernel(node)
                # add new input nodes for the nodes that don't have one
                tbr_inds = []
                for ind in node.newloc_idxs:
                    nmaxind += 1
                    f_z = (z_mat[ind,ind] - z_k_approx.k_bar)
                    if np.abs(f_z) > 1e-7:
                        f_z /= node.z_bar
                        z_k_real = Kernel(dict(a=node.z_kernel.a, c=node.z_kernel.c*f_z))
                        # add node
                        newnode = NETNode(nmaxind, [ind], z_kernel=z_k_real)
                        newnode.newloc_idxs = [ind]
                        self.add_node_with_parent(newnode, node)
                        tbr_inds.append(ind)
                for ind in tbr_inds: node.newloc_idxs.remove(ind)
                # empty the new indices
                node.newloc_idxs = []

        self.set_new_loc_idxs()

    def plot_dendrogram(self, ax,
                        plotargs={}, labelargs={}, textargs={},
                        incolors={},
                        inlabels={}, nodelabels={},
                        cs_comp={}, cmap=None,
                        z_max=None, add_scalebar=True):
        """
        Generate a dendrogram of the NET

        Parameters
        ----------
            ax: :class:`matplotlib.axes`
                the axes object in which the plot will be made
            plotargs : dict (string : value)
                keyword args for the matplotlib plot function, specifies the
                line properties of the dendrogram
            labelargs : dict (string : value)
                keyword args for the matplotlib plot function, specifies the
                marker properties for the node points. Or dict with keys node
                indices, and with values dicts with keyword args for the
                matplotlib function that specify the marker properties for
                specific node points. The entry under key -1 specifies the
                properties for all nodes not explicitly in the keys.
            textargs : dict (string : value)
                keyword args for matplotlib textproperties
            incolors : dict (int : string)
                dict with locinds as keys and colors as values
            inlabels : dict (int : string)
                dict with locinds as keys and label strings as values
            nodelabels: dict (int: string) or None
                labels of the nodes. If None, nodes are named by default
                according to their location indices. If empty dict, no labels
                are added.
            cs_comp : dict (int : float)
                dict with node inds as keys and compartment colors as values
            z_max: float or None
                specifies the y-scale. If None, the scale is computed from
                ``self``
            add_scalebar: bool
                whether or not to add a scale bar
        """
        if cs_comp:
            # compute the compartmental colormap if necessary
            arr = np.array([list(cs_comp.values())])
            max_cs = np.max(arr)
            min_cs = np.min(arr)
            norm_cs = (max_cs - min_cs) * (1. + 1./100.)
            for key, val in cs_comp.items():
                cs_comp[key] = (cs_comp[key] - min_cs) / norm_cs
            if cmap is None: cmap = pl.get_cmap('jet')
            cs_comp['cm'] = cmap
            Z = [[0,0],[0,0]]
            levels = np.linspace(min_cs, max_cs, 100)
            CS3 = pl.contourf(Z, levels, cmap=cmap)
        # get the number of leafs to determine the dendrogram spacing
        rnode    = self.root
        n_branch  = self.degree_of_node(rnode)
        l_spacing = np.linspace(0., 1., n_branch+1)
        # determine input inpedances to fix the y scale
        if z_max == None:
            z_dict = {}
            for node in self.nodes:
                for ind in node.loc_idxs:
                    try:
                        z_dict[ind] += node.z_bar
                    except KeyError:
                        z_dict[ind] = node.z_bar
            z_max = max(z_dict.values())
        # plot the dendrogram
        self._expand_dendrogram(rnode, 0.5, 0.,
                    l_spacing, z_max, ax,
                    plotargs=plotargs, labelargs=labelargs, textargs=textargs,
                    incolors=incolors,
                    inlabels=inlabels, nodelabels=nodelabels,
                    cs_comp=cs_comp)
        # limits
        ax.set_ylim((0.0, 1.2*z_max))
        ax.set_xlim((0.,1.))
        # scalebar
        if add_scalebar:
            sblength = np.around(z_max // 5, -2)
            if sblength < .1: sblength += np.around(z_max % 5, -1)
            if sblength < .1: sblength += np.around(z_max // 5, 0)
            sbwidth = plotargs['lw']*3 if 'lw' in plotargs else 3.
            sbtsize = textargs['size'] if 'size' in textargs else 'small'
            ax.plot([0.,0.], [0., sblength], 'k-', lw=sbwidth)
            ax.annotate(r'%.0f M$\Omega$'%sblength,
                            xy=(0., sblength/2.), xytext=(-0.04, sblength/2.),
                            size=sbtsize, rotation=90, ha='center', va='center')

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axison = False

        return z_max

    def _expand_dendrogram(self, node, x0, y0,
                                        l_spacing, z_max, ax,
                                        plotargs={}, labelargs={}, textargs={},
                                        incolors={},
                                        inlabels={}, nodelabels={},
                                        cs_comp={}):
        # check if part of compartment
        if cs_comp:
            if node.index in list(cs_comp.keys()):
                plotargs = copy.deepcopy(plotargs)
                plotargs['color'] = cs_comp['cm'](cs_comp[node.index])
        # impedance of layer
        ynew = y0 + node.z_bar
        # plot vertical connection line
        ax.vlines(x0, y0, ynew, **plotargs)
        # get the child nodes for recursion
        l0 = 0
        for i, cnode in enumerate(node.child_nodes):
            # attribute space on xaxis
            deg = self.degree_of_node(cnode)
            l1 = l0 + deg
            # new quantities
            xnew = (l_spacing[l0] + l_spacing[l1]) / 2.
            # horizontal connection line limits
            if i == 0:
                xnew0 = xnew
            if i == len(node.child_nodes)-1:
                xnew1 = xnew
            # recursion
            self._expand_dendrogram(cnode, xnew, ynew,
                    l_spacing[l0:l1+1], z_max, ax,
                    plotargs=plotargs, labelargs=labelargs, textargs=textargs,
                    incolors=incolors,
                    inlabels=inlabels, nodelabels=nodelabels,
                    cs_comp=cs_comp)
            # next index
            l0 = l1
        # plot horizontal connection line
        if l0 > 0:
            ax.hlines(ynew, xnew0, xnew1, **plotargs)
        # add label and maybe text annotation to node
        if node.index in labelargs:
            ax.plot([x0], [ynew], **labelargs[node.index])
        elif -1 in labelargs:
            ax.plot([x0], [ynew], **labelargs[-1])
        else:
            try:
                ax.plot([x0], [ynew], **labelargs)
            except TypeError as e:
                pass
        if textargs:
            if nodelabels != None:
                if node.index in nodelabels:
                    if labelargs == {}:
                        ax.plot([x0], [ynew], **nodelabels[node.index][1])
                        ax.annotate(nodelabels[node.index][0],
                                    xy=(x0, ynew), xytext=(x0+0.04, ynew+z_max*0.04),
                                    bbox=dict(boxstyle='round', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)),
                                    **textargs)
                    else:
                        ax.annotate(nodelabels[node.index],
                                    xy=(x0, ynew), xytext=(x0+0.04, ynew+z_max*0.04),
                                    bbox=dict(boxstyle='round', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)),
                                    **textargs)
            else:
                ax.annotate(r'$N='+''.join([str(ind) for ind in node.loc_idxs])+'$',
                                 xy=(x0, ynew), xytext=(x0+0.04, ynew+z_max*0.04),
                                 bbox=dict(boxstyle='round', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)),
                                 **textargs)
            # add input label
            if self.is_leaf(node):
                if inlabels != None:
                    lwidth = plotargs['lw'] if 'lw' in plotargs else 1.
                    ax.vlines(x0, ynew+z_max*0.04, z_max*1.1, lw=lwidth, linestyle=':', color='k')
                    if node.loc_idxs[0] in incolors:
                        bboxdict = dict(boxstyle='round', ec=incolors[node.loc_idxs[0]], fc=incolors[node.loc_idxs[0]], alpha=0.5)
                    else:
                        bboxdict = dict(boxstyle='round', ec=(0.5, 0.5, 1.), fc=(0.8, 0.8, 1.))
                    if node.loc_idxs[0] in inlabels:
                        textstr = inlabels[node.loc_idxs[0]]
                    else:
                        textstr = r'$'+str(node.loc_idxs[0])+'$'
                    ax.annotate(textstr,
                                        xy=(x0, z_max*1.1), xytext=(x0, z_max*1.14), ha='center',
                                        bbox=bboxdict,
                                        **textargs)
