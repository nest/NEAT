# -*- coding: utf-8 -*-
#
# sovtree.py
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

import itertools
import copy

from . import morphtree
from .stree import STree
from .morphtree import MorphLoc, MorphNode
from .phystree import PhysNode, PhysTree
from .netree import NETNode, NET, Kernel

from ..tools.fittools import zerofinding as zf
from ..tools.fittools import histogramsegmentation as hs


def _consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


class SOVNode(PhysNode):
    """
    Node that defines functions and stores quantities to implement separation
    of variables calculation (Major, 1993)
    """

    def __init__(self, index, p3d=None):
        super().__init__(index, p3d)

    def _set_sov(self, channel_storage, tau_0=0.02):
        self.counter = 0
        # segment parameters
        self.g_m = self.calc_g_tot(channel_storage)  # uS/cm^2
        # parameters for SOV approach
        self.R_sov = self.R * 1e-4  # convert um to cm
        self.L_sov = self.L * 1e-4  # convert um to cm
        self.tau_m = self.c_m / self.g_m  # s
        self.eps_m = self.tau_m / tau_0
        self.lambda_m = np.sqrt(self.R_sov / (2.0 * self.g_m * self.r_a))  # cm
        self.tau_0 = tau_0  # s
        self.z_a = self.r_a / (np.pi * self.R_sov**2)  # MOhm/cm
        self.g_inf_m = 1.0 / (self.z_a * self.lambda_m)  # uS
        # # segment amplitude information
        self.kappa_m = np.nan
        self.mu_vals_m = np.nan
        self.q_vals_m = np.nan

    def __str__(self, with_parent=True, with_morph_info=False):
        if with_morph_info:
            node_str = super(PhysNode, self).__str__(with_parent=with_parent)
        else:
            node_str = super(MorphNode, self).__str__(with_parent=with_parent)

        if hasattr(self, "R_sov"):
            node_str += (
                f" --- "
                f"(g_m = {self.g_m:.8f} uS/cm^2, "
                f"tau_m = {self.tau_m:.8f} s, "
                f"eps_m = {self.eps_m:.8f}, "
                f"R_sov = {self.R_sov:.8f} cm, "
                f"L_sov = {self.L_sov:.8f} cm)"
            )
        return node_str

    def q_m(self, x):
        return np.sqrt(self.eps_m * x**2 - 1.0)

    def dq_dp_m(self, x):
        return -self.tau_m / (2.0 * self.q_m(x))

    def mu_m(self, x):
        cns = self.child_nodes
        if len(cns) == 0:
            return self.z_a * self.lambda_m / self.q_m(x) * self.g_shunt
        else:
            x = zf._to_complex(x)
            mu_ds = [cn.mu_m(x) for cn in cns]
            q_ds = [cn.q_m(x) for cn in cns]
            return (
                self.g_shunt
                - np.sum(
                    [
                        cn.g_inf_m
                        * q_ds[i]
                        * (1.0 - mu_ds[i] / np.tan(q_ds[i] * cn.L_sov / cn.lambda_m))
                        / (1.0 / np.tan(q_ds[i] * cn.L_sov / cn.lambda_m) + mu_ds[i])
                        for i, cn in enumerate(cns)
                    ],
                    0,
                )
            ) / (self.g_inf_m * self.q_m(x))

    def dmu_dp_m(self, x):
        cns = self.child_nodes
        if len(cns) == 0:
            return -self.dq_dp_m(x) * self.mu_m(x) / self.q_m(x)
        else:
            x = zf._to_complex(x)
            mu_ds = [cn.mu_m(x) for cn in cns]
            dmu_dp_ds = [cn.dmu_dp_m(x) for cn in cns]
            q_ds = [cn.q_m(x) for cn in cns]
            dq_dp_ds = [cn.dq_dp_m(x) for cn in cns]
            return (
                -self.dq_dp_m(x) * self.mu_m(x)
                - np.sum(
                    [
                        cn.g_inf_m
                        * (
                            dq_dp_ds[i]
                            * (
                                1.0
                                - mu_ds[i] / np.tan(q_ds[i] * cn.L_sov / cn.lambda_m)
                            )
                            / (
                                1.0 / np.tan(q_ds[i] * cn.L_sov / cn.lambda_m)
                                + mu_ds[i]
                            )
                            + q_ds[i]
                            * (
                                (1.0 + mu_ds[i] ** 2)
                                * dq_dp_ds[i]
                                * cn.L_sov
                                / cn.lambda_m
                                - dmu_dp_ds[i]
                            )
                            / (
                                np.cos(q_ds[i] * cn.L_sov / cn.lambda_m)
                                + mu_ds[i] * np.sin(q_ds[i] * cn.L_sov / cn.lambda_m)
                            )
                            ** 2
                        )
                        for i, cn in enumerate(cns)
                    ],
                    0,
                )
                / self.g_inf_m
            ) / self.q_m(x)

    def _set_kappa_factors(self, xzeros):
        xzeros = zf._to_complex(xzeros)
        self.kappa_m = self.parent_node.kappa_m / (
            np.cos(self.q_m(xzeros) * self.L_sov / self.lambda_m)
            + self.mu_m(xzeros) * np.sin(self.q_m(xzeros) * self.L_sov / self.lambda_m)
        )

    def _set_mu_vals(self, xzeros):
        xzeros = zf._to_complex(xzeros)
        self.mu_vals_m = self.mu_m(xzeros)

    def _set_q_vals(self, xzeros):
        xzeros = zf._to_complex(xzeros)
        self.q_vals_m = self.q_m(xzeros)

    def _find_local_poles(self, maxspace_freq=500):
        poles = []
        pmultiplicities = []
        n = 0
        val = 0.0
        while val < maxspace_freq:
            poles.append(np.sqrt((1.0 + val**2) / self.eps_m))
            if val == 0:
                pmultiplicities.append(0.5)
            else:
                pmultiplicities.append(1.0)
            n += 1
            val = n * np.pi * self.lambda_m / self.L_sov
        return poles, pmultiplicities

    def _set_zeros_poles(self, maxspace_freq=500, pprint=False):
        cns = self.child_nodes
        # find the poles of (1 + mu*cot(qL/l))/(cot(qL/l) + mu)
        lpoles, lpmultiplicities = self._find_local_poles(maxspace_freq)
        for cn in cns:
            lpoles.extend(cn.poles)
            lpmultiplicities.extend(cn.pmultiplicities)
        inds = np.argsort(lpoles)
        lpoles = np.array(lpoles)[inds]
        lpmultiplicities = np.array(lpmultiplicities)[inds]
        # construct the function cot(qL/l) + mu
        f = lambda x: 1.0 / np.tan(
            self.q_m(x) * self.L_sov / self.lambda_m
        ) + self.mu_m(x)
        dfdx = (
            lambda x: -2.0
            * x
            * (
                -(self.L_sov / self.lambda_m)
                / np.sin(self.q_m(x) * self.L_sov / self.lambda_m) ** 2
                * self.dq_dp_m(x)
                + self.dmu_dp_m(x)
            )
            / self.tau_0
        )
        # find its zeros, this are the poles of the next level
        xval = 1.5 / np.sqrt(self.eps_m)
        for cn in cns:
            c_eps_m = cn.eps_m
            xval_ = 1.5 / np.sqrt(c_eps_m)
            if xval_ > xval:
                xval = xval_
        if np.abs(f(xval)) < 1e-20:
            xval = (xval + lpoles[1]) / 2.0
        if pprint:
            print("")
            print("xval: ", xval)
        # find zeros larger than xval
        if pprint:
            print("finding real poles")
        PF = zf.poleFinder(
            fun=f,
            dfun=dfdx,
            global_poles={"poles": lpoles, "pmultiplicities": lpmultiplicities},
        )
        poles, pmultiplicities = PF.find_real_zeros(vmin=xval)
        # find the first zero
        if pprint:
            print("finding first pole")
        p1 = []
        pm1 = []
        zf.find_zeros_on_segment(
            p1, pm1, 0.0, xval, f, dfdx, lpoles, lpmultiplicities, pprint=pprint
        )
        self.poles = np.concatenate((p1, poles)).real
        self.pmultiplicities = np.concatenate((pm1, pmultiplicities)).real


class SomaSOVNode(SOVNode):
    """
    Subclass of SOVNode to threat the special case of the soma

    The following member functions are not supposed to work properly,
    calling them may result in errors:
    `neat.SOVNode._set_kappa_factors()`
    `neat.SOVNode._set_mu_vals()`
    `neat.SOVNode._set_q_vals()`
    `neat.SOVNode._find_local_poles()`
    """

    def __init__(self, index, p3d=None):
        super().__init__(index, p3d)

    def _set_sov(self, channel_storage, tau_0=0.02):
        self.counter = 0
        # convert to cm
        self.R_sov = self.R * 1e-4  # convert um to cm
        self.L_sov = self.L * 1e-4  # convert um to cm
        # surface
        self.A = 4.0 * np.pi * self.R_sov**2  # cm^2
        # total conductance
        self.g_m = self.calc_g_tot(channel_storage=channel_storage)  # uS/cm^2
        # parameters for the SOV approach
        self.tau_m = self.c_m / self.g_m  # s
        self.eps_m = self.tau_m / tau_0  # ns
        self.g_s = self.g_m * self.A + self.g_shunt  # uS
        self.c_s = self.c_m * self.A  # uF
        self.tau_0 = tau_0  # s
        # segment amplitude factors
        self.kappa_m = 1.0

    def f_transc(self, x):
        cns = self.child_nodes
        x = zf._to_complex(x)
        mu_ds = [cn.mu_m(x) for cn in cns]
        q_ds = [cn.q_m(x) for cn in cns]
        return self.g_s * (1.0 - self.eps_m * x**2) - np.sum(
            [
                cn.g_inf_m
                * q_ds[i]
                * (1.0 - mu_ds[i] / np.tan(q_ds[i] * cn.L_sov / cn.lambda_m))
                / (1.0 / np.tan(q_ds[i] * cn.L_sov / cn.lambda_m) + mu_ds[i])
                for i, cn in enumerate(cns)
            ],
            0,
        )

    def dN_dp(self, x):
        cns = self.child_nodes
        x = zf._to_complex(x)
        mu_ds = [cn.mu_m(x) for cn in cns]
        dmu_dp_ds = [cn.dmu_dp_m(x) for cn in cns]
        q_ds = [cn.q_m(x) for cn in cns]
        dq_dp_ds = [cn.dq_dp_m(x) for cn in cns]
        return self.c_s - np.sum(
            [
                cn.g_inf_m
                * (
                    dq_dp_ds[i]
                    * (1.0 - mu_ds[i] / np.tan(q_ds[i] * cn.L_sov / cn.lambda_m))
                    / (1.0 / np.tan(q_ds[i] * cn.L_sov / cn.lambda_m) + mu_ds[i])
                    + q_ds[i]
                    * (
                        (1.0 + mu_ds[i] ** 2) * dq_dp_ds[i] * cn.L_sov / cn.lambda_m
                        - dmu_dp_ds[i]
                    )
                    / (
                        np.cos(q_ds[i] * cn.L_sov / cn.lambda_m)
                        + mu_ds[i] * np.sin(q_ds[i] * cn.L_sov / cn.lambda_m)
                    )
                    ** 2
                )
                for i, cn in enumerate(cns)
            ],
            0,
        )

    def _set_zeros_poles(self, maxspace_freq=500, pprint=False):
        # find the poles of cot(qL/l) + mu
        lpoles = []
        lpmultiplicities = []
        for cn in self.child_nodes:
            lpoles.extend(cn.poles)
            lpmultiplicities.extend(cn.pmultiplicities)
        inds = np.argsort(lpoles)
        lpoles = np.array(lpoles)[inds]
        lpmultiplicities = np.array(lpmultiplicities)[inds]
        # construct the function cot(qL/l) + mu
        f = lambda x: self.f_transc(x)
        dfdx = lambda x: -2.0 * x * self.dN_dp(x) / self.tau_0
        # find its zeros, this are the inverse timescales of the model
        xval = 1.5 / np.sqrt(self.eps_m)
        for cn in self.child_nodes:
            c_eps_m = cn.eps_m
            xval_ = 1.5 / np.sqrt(c_eps_m)
            if xval_ > xval:
                xval = xval_
        if np.abs(f(xval)) < 1e-20:
            xval = (xval + lpoles[1]) / 2.0
        if pprint:
            print("xval: ", xval)
        # find zeros larger than xval
        PF = zf.poleFinder(
            fun=f,
            dfun=dfdx,
            global_poles={"poles": lpoles, "pmultiplicities": lpmultiplicities},
        )
        zeros, multiplicities = PF.find_real_zeros(vmin=xval)
        # find the first zero
        z1 = []
        zm1 = []
        zf.find_zeros_on_segment(
            z1, zm1, 0.0, xval, f, dfdx, lpoles, lpmultiplicities, pprint=pprint
        )
        self.zeros = np.concatenate((z1, zeros)).real
        self.zmultiplicities = np.concatenate((zm1, multiplicities)).real
        self.prefactors = self.dN_dp(self.zeros).real


class SOVTree(PhysTree):
    """
    Class that computes the separation of variables time scales and spatial
    mode functions for a given morphology and electrical parameter set. Employs
    the algorithm by (Major, 1994). This three defines a special
    `neat.SomaSOVNode` on as a derived class from `neat.SOVNode` as some
    functions required for SOV calculation are different and thus overwritten.

    The SOV calculation proceeds on the computational tree (see docstring of
    `neat.MorphNode`). Thus it makes no sense to look for sov quantities in the
    original tree.
    """

    def __init__(self, *args, **kwargs):
        self.maxspace_freq = None
        super().__init__(*args, **kwargs)

    def _get_repr_dict(self):
        repr_dict = super()._get_repr_dict()
        repr_dict.update({"maxspace_freq": f"{self.maxspace_freq:1.6g}"})
        return repr_dict

    def __repr__(self):
        repr_str = STree.__repr__(self)
        return repr_str + repr(self._get_repr_dict())

    def create_corresponding_node(self, node_index, p3d=None):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        if node_index == 1:
            return SomaSOVNode(node_index, p3d=p3d)
        else:
            return SOVNode(node_index, p3d=p3d)

    @morphtree.computational_tree_decorator
    def get_sov_matrices(self, loc_arg):
        """
        returns the alphas, the reciprocals of the mode time scales [1/ms]
        as well as the spatial functions evaluated at ``locs``

        Parameters
        ----------
            loc_arg: see :func:`neat.MorphTree.convert_loc_arg_to_locs()`
                the locations at which to evaluate the SOV matrices

        Returns
        -------
            alphas: np.ndarray of complex (ndim = 1)
                the reciprocals of mode time-scales (kHz)
            gammas: np.ndarray of complex (ndim = 2)
                the spatial function associated with each mode, evaluated at
                each locations. Dimension 0 is number of modes and dimension 1
                number of locations
        """
        locs = self.convert_loc_arg_to_locs(loc_arg)
        if len(self) > 1:
            # set up the matrices
            zeros = self.root.zeros
            prefactors = self.root.prefactors
            alphas = zeros**2 / (self.tau_0 * 1e3)
            gammas = np.zeros((len(alphas), len(locs)), dtype=complex)
            # fill the matrix of prefactors
            for ii, loc in enumerate(locs):
                if loc["node"] == 1:
                    x = 0.0
                    node = self.root.child_nodes[0]
                else:
                    x = loc["x"]
                    node = self[loc["node"]]
                # fill a column of the matrix, corresponding to current loc
                gammas[:, ii] = (
                    node.kappa_m
                    * (
                        np.cos(node.q_vals_m * (1.0 - x) * node.L_sov / node.lambda_m)
                        + node.mu_vals_m
                        * np.sin(node.q_vals_m * (1.0 - x) * node.L_sov / node.lambda_m)
                    )
                    / np.sqrt(prefactors * 1e3)
                )
        else:
            alphas = np.array([1e-3 / self.root.tau_m])
            gammas = np.array([[np.sqrt(alphas[0] / self.root.g_s)]])

        # return the matrices
        return alphas, gammas

    @morphtree.computational_tree_decorator
    def calc_sov_equations(self, maxspace_freq=500.0, pprint=False):
        """
        Calculate the timescales and spatial functions of the separation of
        variables approach, using the algorithm by (Major, 1993).

        The (reciprocals) of the timescales (i.e. the roots of the transcendental
        equation) are stored in the somanode.
        The spatial factors are stored in each (computational) node.

        Parameters
        ----------
        maxspace_freq: float (default is 500)
            roughly corresponds to the maximal spatial frequency of the
            smallest time-scale mode
        """
        self.maxspace_freq = maxspace_freq

        self.tau_0 = np.pi  # 1.
        for node in self:
            node._set_sov(self.channel_storage, tau_0=self.tau_0)
        if len(self) > 1:
            # start the recursion through the tree
            self._sov_from_leaf(
                self.leafs[0],
                self.leafs[1:],
                maxspace_freq=maxspace_freq,
                pprint=pprint,
            )
            # zeros are now found, set the kappa factors
            zeros = self.root.zeros
            self._sov_from_root(self.root, zeros)
            # clean
            for node in self:
                node.counter = 0
        else:
            self[1]._set_sov(self.channel_storage, tau_0=self.tau_0)

    def _sov_from_leaf(self, node, leafs, count=0, maxspace_freq=500.0, pprint=False):
        if pprint:
            print("Forward sweep: " + str(node))
        pnode = node.parent_node
        # log how many times recursion has passed at node
        if not self.is_leaf(node):
            node.counter += 1
        # if the number of childnodes of node is equal to the amount of times
        # the recursion has passed node, the mu functions can be set. Otherwise
        # we start a new recursion at another leaf.
        if node.counter == len(node.child_nodes):
            node._set_zeros_poles(maxspace_freq=maxspace_freq)
            if not self.is_root(node):
                self._sov_from_leaf(
                    pnode,
                    leafs,
                    count=count + 1,
                    maxspace_freq=maxspace_freq,
                    pprint=pprint,
                )
        elif len(leafs) > 0:
            self._sov_from_leaf(
                leafs[0],
                leafs[1:],
                count=count + 1,
                maxspace_freq=maxspace_freq,
                pprint=pprint,
            )

    def _sov_from_root(self, node, zeros):
        for cnode in node.child_nodes:
            cnode._set_kappa_factors(zeros)
            cnode._set_mu_vals(zeros)
            cnode._set_q_vals(zeros)
            self._sov_from_root(cnode, zeros)

    def get_mode_importance(
        self, loc_arg=None, sov_data=None, importance_type="simple"
    ):
        """
        Gives the overal importance of the SOV modes for a certain set of
        locations

        Parameters
        ----------
            loc_arg: None or list of locations
            sov_data: None or tuple of mode matrices
                One of the keyword arguments ``loc_arg`` or ``sov_data``
                must not be ``None``. If ``loc_arg`` is not ``None``, the importance
                is evaluated at these locations (see
                :func:`neat.MorphTree.convert_loc_arg_to_locs`).
                If ``sov_data`` is not ``None``, it is a tuple of a vector of
                the reciprocals of the mode timescales and a matrix with the
                corresponding spatial mode functions.
            importance_type: string ('relative' or 'absolute')
                when 'absolute', returns an absolute measure of the importance,
                when 'relative', normalizes so that maximum importance is one.
                Defaults to 'relative'.

        Returns
        -------
            np.ndarray (ndim = 1)
                the importances associated with each mode for the provided set
                of locations
        """
        if loc_arg is not None:
            locs = self.convert_loc_arg_to_locs(loc_arg)
            alphas, gammas = self.get_sov_matrices(locs)
        elif sov_data is not None:
            alphas = sov_data[0]
            gammas = sov_data[1]
        else:
            raise IOError(
                "One of the kwargs `loc_arg` or `sov_data` must not be ``None``"
            )

        if importance_type == "simple":
            absolute_importance = np.sum(np.abs(gammas), 1) / np.abs(alphas)
        elif importance_type == "full":
            absolute_importance = np.zeros(len(alphas))
            for kk, (alpha, phivec) in enumerate(zip(alphas, gammas)):
                absolute_importance[kk] = np.sqrt(
                    np.sum(np.abs(np.dot(phivec[:, None], phivec[None, :])))
                    / np.abs(alpha)
                )
        else:
            raise ValueError("`importance_type` argument can be 'simple' or \
                              'full'")

        return absolute_importance / np.max(absolute_importance)

    def get_important_modes(
        self,
        loc_arg=None,
        sov_data=None,
        eps=1e-4,
        sort_type="timescale",
        return_importance=False,
    ):
        """
        Returns the most importand eigenmodes (those whose importance is above
        the threshold defined by `eps`)

        Parameters
        ----------
            loc_arg: None or list of locations
            sov_data: None or tuple of mode matrices
                One of the keyword arguments ``loc_arg`` or ``sov_data``
                must not be ``None``. If ``loc_arg`` is not ``None``, the importance
                is evaluated at these locations (see
                :func:`neat.MorphTree.convert_loc_arg_to_locs`).
                If ``sov_data`` is not ``None``, it is a tuple of a vector of
                the reciprocals of the mode timescales and a matrix with the
                corresponding spatial mode functions.
            eps: float
                the cutoff threshold in relative importance below which modes
                are truncated
            sort_type: string ('timescale' or 'importance')
                specifies in which order the modes are returned. If 'timescale',
                modes are sorted in order of decreasing time-scale, if
                'importance', modes are sorted in order of decreasing importance.
            return_importance: bool
                if ``True``, returns the importance metric associated with each
                mode

        Returns
        -------
            alphas: np.ndarray of complex (ndim = 1)
                the reciprocals of mode time-scales ``[kHz]``
            gammas: np.ndarray of complex (ndim = 2)
                the spatial function associated with each mode, evaluated at
                each locations. Dimension 0 is number of modes and dimension 1
                number of locations
            importance: np.ndarray (`shape` matches `alphas`, only if `return_importance` is ``True``)
                value of importance metric for each mode
        """
        if loc_arg is not None:
            locs = self.convert_loc_arg_to_locs(loc_arg)
            alphas, gammas = self.get_sov_matrices(locs)
        elif sov_data is not None:
            alphas = sov_data[0]
            gammas = sov_data[1]
        else:
            raise IOError(
                "One of the kwargs `loc_arg` or `sov_data` must not be ``None``"
            )
        importance = self.get_mode_importance(
            sov_data=(alphas, gammas), importance_type="simple"
        )
        inds = np.where(importance > eps)[0]
        # only modes above importance cutoff
        alphas, gammas, importance = alphas[inds], gammas[inds, :], importance[inds]
        if sort_type == "timescale":
            inds_sort = np.argsort(np.abs(alphas))
        elif sort_type == "importance":
            inds_sort = np.argsort(importance)[::-1]
        else:
            raise ValueError("`sort_type` argument can be 'timescale' or \
                              'importance'")
        if return_importance:
            return alphas[inds_sort], gammas[inds_sort, :], importance[inds_sort]
        else:
            return alphas[inds_sort], gammas[inds_sort, :]

    def get_kernels(
        self,
        loc_arg=None,
        sov_data=None,
        eps=0.0,
    ):
        """
        Returns the impulse response kernels as a double nested list of "neat.Kernel".
        The element at the position i,j represents the transfer impedance kernel
        between compartments i and j.

        Parameters
        ----------
        loc_arg: None or list of locations
        sov_data: None or tuple of mode matrices
            One of the keyword arguments ``loc_arg`` or ``sov_data``
            must not be ``None``. If ``loc_arg`` is not ``None``, the importance
            is evaluated at these locations (see
            :func:`neat.MorphTree.convert_loc_arg_to_locs`).
            If ``sov_data`` is not ``None``, it is a tuple of a vector of
            the reciprocals of the mode timescales and a matrix with the
            corresponding spatial mode functions.
        eps: float
            the cutoff threshold in relative importance below which modes
            are truncated

        Returns
        -------
        list of list of `neat.Kernel`
            The kernels of the model
        """
        if loc_arg is not None:
            locs = self.convert_loc_arg_to_locs(loc_arg)
            alphas, gammas = self.get_important_modes(locs, eps=eps)
        elif sov_data is not None:
            alphas = sov_data[0]
            gammas = sov_data[1]
        else:
            raise IOError(
                "One of the kwargs `loc_arg` or `sov_data` must not be ``None``"
            )
        # some renaming
        nn = gammas.shape[1]
        aa = alphas
        cc = np.einsum("ik,kj->kij", gammas.T, gammas)
        return [[Kernel((aa, cc[:, ii, jj])) for ii in range(nn)] for jj in range(nn)]

    def calc_zf(
        self,
        loc0,
        loc1,
        freqs=None,
        eps=0.0,
    ):
        """
        Computes the impedance between two locations for the provided frequencies

        Parameters
        ----------
        loc1: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer impedance is computed
        loc2: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer impedance is computed
        freqs: np.ndarray of complex or None (default)
            if ``None``, returns the steady state impedance matrix, if
            a array of complex numbers, returns the impedance matrix for
            each Fourrier frequency in the array
        eps: float
            the cutoff threshold in relative importance below which modes
            are truncated

        Returns
        ------
        `np.ndarray` (``ndim=3``)
            the matrix of impulse responses, first dimension corresponds to the
            time axis, second and third dimensions contain the impulse response
            in ``[MOhm/ms]`` at that time point
        """
        if freqs is None:
            freqs = np.array([0.0])
        kernel = self.get_kernels(loc_arg=[loc0, loc1], eps=eps)[0][1]
        return kernel.ft(freqs)

    def calc_zt(
        self,
        loc0,
        loc1,
        times=None,
        eps=0.0,
    ):
        """
        Computes the impulse response kernel between two locations for the provided frequencies

        Parameters
        ----------
        loc1: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer impedance is computed
        loc2: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer impedance is computed
        times: np.array
            The time-points at which to evaluate the kernels. If not provided,
            evaluates `t = np.linspace(0.1,100.,1000)`
        eps: float
            the cutoff threshold in relative importance below which modes
            are truncated

        Returns
        ------
        `np.ndarray` (``ndim=3``)
            the matrix of impulse responses, first dimension corresponds to the
            time axis, second and third dimensions contain the impulse response
            in ``[MOhm/ms]`` at that time point
        """
        if times is None:
            times = np.linspace(0.1, 100.0, 1000)
        kernel = self.get_kernels(loc_arg=[loc0, loc1], eps=eps)[0][1]
        return kernel.t(times)

    def calc_impulse_response_matrix(
        self,
        times=None,
        loc_arg=None,
        sov_data=None,
        eps=0.0,
    ):
        """
        Computes the matrix of impulse response kernels at a given set of
        locations

        Parameters
        ----------
        times: np.array
            The time-points at which to evaluate the kernels. If not provided,
            evaluates `t = np.linspace(0.1,100.,1000)`
        loc_arg: None or list of locations
        sov_data: None or tuple of mode matrices
            One of the keyword arguments ``loc_arg`` or ``sov_data``
            must not be ``None``. If ``loc_arg`` is not ``None``, the importance
            is evaluated at these locations (see
            :func:`neat.MorphTree.convert_loc_arg_to_locs`).
            If ``sov_data`` is not ``None``, it is a tuple of a vector of
            the reciprocals of the mode timescales and a matrix with the
            corresponding spatial mode functions.
        eps: float
            the cutoff threshold in relative importance below which modes
            are truncated

        Returns
        ------
        `np.ndarray` (``ndim=3``)
            the matrix of impulse responses, first dimension corresponds to the
            time axis, second and third dimensions contain the impulse response
            in ``[MOhm/ms]`` at that time point
        """
        if times is None:
            times = np.linspace(0.1, 100.0, 1000)
        kernels = self.get_kernels(loc_arg=loc_arg, sov_data=sov_data, eps=eps)
        zt_mat = np.array([[zk.t(times) for zk in row] for row in kernels])
        return np.transpose(zt_mat, axes=(2, 0, 1))

    def calc_impedance_matrix(
        self,
        freqs=None,
        loc_arg=None,
        sov_data=None,
        eps=0.0,
        mem_limit=500,
    ):
        """
        Compute the impedance matrix for a set of locations

        Parameters
        ----------
        freqs: np.ndarray of complex or None (default)
            if ``None``, returns the steady state impedance matrix, if
            a array of complex numbers, returns the impedance matrix for
            each Fourrier frequency in the array
        loc_arg: None or list of locations
        sov_data: None or tuple of mode matrices
            One of the keyword arguments ``loc_arg`` or ``sov_data``
            must not be ``None``. If ``loc_arg`` is not ``None``, the importance
            is evaluated at these locations (see
            :func:`neat.MorphTree.convert_loc_arg_to_locs`).
            If ``sov_data`` is not ``None``, it is a tuple of a vector of
            the reciprocals of the mode timescales and a matrix with the
            corresponding spatial mode functions.
        eps: float
            the cutoff threshold in relative importance below which modes
            are truncated
        mem_limit: int
            parameter governs whether the fast (but memory intense) method
            or the slow method is used

        Returns
        -------
        np.ndarray of floats (ndim = 2 or 3)
            the impedance matrix, steady state if `freqs` is ``None``, the
            frequency dependent impedance matrix if `freqs` is given, with
            the frequency dependence at the first dimension ``[MOhm ]``
        """
        # TODO: this shortened formulation causes some tests to fail, have to figure out why
        # remove_dim = False
        # if freqs is None:
        #     remove_dim = True
        #     freqs = np.array([0.])

        # kernels = self.get_kernels(loc_arg=loc_arg, sov_data=sov_data, eps=eps)
        # zf_mat = np.array([[zk.ft(freqs) for zk in row] for row in kernels])

        # if remove_dim:
        #     zf_mat[:,:,0]
        # return np.transpose(zf_mat, axes=(2,0,1))
        if loc_arg is not None:
            locs = self.convert_loc_arg_to_locs(loc_arg)
            alphas, gammas = self.get_sov_matrices(locs)
        elif sov_data is not None:
            alphas = sov_data[0]
            gammas = sov_data[1]
        else:
            raise IOError(
                "One of the kwargs `loc_arg` or `sov_data` must not be ``None``"
            )
        n_loc = gammas.shape[1]
        if freqs is None:
            # construct the 2d steady state matrix
            y_activation = 1.0 / alphas
            # compute the matrix, methods depends on memory limit
            if gammas.shape[1] < mem_limit and gammas.shape[0] < int(mem_limit / 2.0):
                z_mat = np.sum(
                    gammas[:, :, np.newaxis]
                    * gammas[:, np.newaxis, :]
                    * y_activation[:, np.newaxis, np.newaxis],
                    0,
                ).real
            else:
                z_mat = np.zeros((n_loc, n_loc))
                for ii, jj in itertools.product(range(n_loc), range(n_loc)):
                    z_mat[ii, jj] = np.sum(
                        gammas[:, ii] * gammas[:, jj] * y_activation
                    ).real
        else:
            # construct the 3d fourrier matrix
            y_activation = 1e3 / (alphas[np.newaxis, :] * 1e3 + freqs[:, np.newaxis])
            z_mat = np.zeros((len(freqs), n_loc, n_loc), dtype=complex)
            for ii, jj in itertools.product(range(n_loc), range(n_loc)):
                z_mat[:, ii, jj] = np.sum(
                    gammas[np.newaxis, :, ii]
                    * gammas[np.newaxis, :, jj]
                    * y_activation,
                    1,
                )
        return z_mat

    def construct_net(
        self,
        dz=50.0,
        dx=10.0,
        eps=1e-4,
        use_hist=False,
        add_lin_terms=True,
        improve_input_impedance=False,
        pprint=False,
    ):
        """
        Construct a Neural Evaluation Tree (NET) for this cell. The locations
        for which impedance values are computed are stored under the name
        `net eval`

        Parameters
        ----------
        dz: float
            the impedance step for the NET model derivation
        dx: float
            the distance step to evaluate the impedance matrix
        eps: float
            the cutoff threshold in relative importance below which modes
            are truncated
        use_hist: bool
            whether or not to use histogram segmentations to find well
            separated parts of the dendritic tree (such ass apical tree)
        add_lin_terms:
            take into account that the optained NET will be used in conjunction
            with linear terms

        Returns
        -------
        `neat.NETree`
            The neural evaluation tree (Wybo et al., 2019) associated with the
            morphology.
        """
        # create a set of location at which to evaluate the impedance matrix
        self.distribute_locs_uniform(dx=dx, name="net eval")
        # compute the z_mat matrix
        alphas, gammas = self.get_important_modes(loc_arg="net eval", eps=eps)
        z_mat = self.calc_impedance_matrix(sov_data=(alphas, gammas))
        # derive the NET
        net = NET()
        self._add_layer_a(
            net,
            None,
            z_mat,
            alphas,
            gammas,
            0.0,
            0,
            np.arange(len(self.get_locs("net eval"))),
            dz=dz,
            use_hist=use_hist,
            add_lin_terms=add_lin_terms,
            pprint=pprint,
        )
        net.set_new_loc_idxs()
        if improve_input_impedance:
            self._improve_input_impedance(net, alphas, gammas)
        if add_lin_terms:
            lin_terms = self.compute_lin_terms(net, sov_data=(alphas, gammas))
            return net, lin_terms
        else:
            return net

    def _add_layer_a(
        self,
        net,
        pnode,
        z_mat,
        alphas,
        gammas,
        z_max_prev,
        z_ind_0,
        true_loc_idxs,
        dz=100.0,
        use_hist=True,
        add_lin_terms=False,
        pprint=False,
    ):
        # create a histogram
        n_bin = 15
        z_hist = np.histogram(z_mat[0, :], n_bin, density=False)
        # find the histogram partition
        h_ftc = hs.histogramSegmentator(z_hist)
        s_inds, p_inds = h_ftc.partition_fine_to_coarse(eps=1.4)
        while len(s_inds) > 3:
            s_inds = np.delete(s_inds, 1)

        # identify the necessary node indices and kernel computation indices
        node_inds = []
        kernel_inds = []
        min_inds = []
        for ii, si in enumerate(s_inds[:-1]):
            if si > 0:
                n_inds = np.where(z_mat[0, :] > z_hist[1][si + 1])[0]
                k_inds = np.where(
                    np.logical_and(
                        z_mat[0, :] > z_hist[1][si + 1],
                        z_mat[0, :] <= z_hist[1][s_inds[ii + 1] + 1],
                    )
                )[0]
                min_ind = np.argmin(z_mat[0, k_inds])
                min_inds.append(min_ind)
            else:
                n_inds = np.where(z_mat[0, :] >= z_hist[1][0])[0]
                k_inds = np.where(
                    np.logical_and(
                        z_mat[0, :] >= z_hist[1][0],
                        z_mat[0, :] <= z_hist[1][s_inds[ii + 1] + 1],
                    )
                )[0]
                min_ind = np.argmin(z_mat[0, k_inds])
                min_inds.append(min_ind)
            node_inds.append(n_inds)
            kernel_inds.append(k_inds)

        # add NET nodes to the NET tree
        for ii, n_inds in enumerate(node_inds):
            k_inds = kernel_inds[ii]
            if len(k_inds) != 0:
                if add_lin_terms:
                    # get the minimal kernel
                    gammas_avg = gammas[:, 0] * gammas[:, k_inds[min_inds[ii]]]
                else:
                    # get the average kernel
                    if len(k_inds) < 100000:
                        gammas_avg = np.mean(gammas[:, 0:1] * gammas[:, k_inds], 1)
                    else:
                        inds_ = np.random.choice(k_inds, size=100000)
                        gammas_avg = np.mean(gammas[:, 0:1] * gammas[:, inds_], 1)
                z_avg_approx = np.sum(gammas_avg / alphas).real
                self._subtract_parent_kernels(gammas_avg, pnode)
                # add a node to the tree
                node = NETNode(
                    len(net), true_loc_idxs[n_inds], z_kernel=(alphas, gammas_avg)
                )
                if pnode != None:
                    net.add_node_with_parent(node, pnode)
                else:
                    net.root = node
                # set new pnode
                pnode = node

                # print stuff
                if pprint:
                    print(node)
                    print("n_loc =", len(node.loc_idxs))
                    print("(locind0, size) = ", (k_inds[0], z_mat.shape[0]))
                    print("")

                if k_inds[0] == 0:
                    # start new branches, split where they originate from soma by
                    # checking where input impedance is close to somatic transfer
                    # impedance
                    z_max = z_hist[1][s_inds[ii + 1]]
                    # check where new dendritic branches start
                    z_diag = z_mat[k_inds, k_inds]
                    z_x0 = z_mat[k_inds, 0]
                    b_inds = np.where(np.abs(z_diag - z_x0) < dz / 2.0)[0][1:].tolist()
                    if len(b_inds) > 0:
                        if b_inds[0] != 1:
                            b_inds = [1] + b_inds
                        kk = len(b_inds) - 1
                        while kk > 0:
                            if b_inds[kk] - 1 == b_inds[kk - 1]:
                                del b_inds[kk]
                            kk -= 1
                    else:
                        b_inds = [1]
                    for jj, i0 in enumerate(b_inds):
                        # make new z_mat matrix
                        i1 = len(k_inds) if i0 == b_inds[-1] else b_inds[jj + 1]
                        inds = np.meshgrid(k_inds[i0:i1], k_inds[i0:i1], indexing="ij")
                        z_mat_new = copy.deepcopy(z_mat[inds[0], inds[1]])
                        # move further in the tree
                        self._add_layer_b(
                            net,
                            node,
                            z_mat_new,
                            alphas,
                            gammas,
                            z_max,
                            k_inds[i0:i1],
                            dz=dz,
                            use_hist=use_hist,
                            add_lin_terms=add_lin_terms,
                        )
                else:
                    # make new z_mat matrix
                    k_seqs = _consecutive(k_inds)
                    if pprint:
                        print("\n>>> consecutive")
                        print("nseq:", len(k_seqs))
                        for k_seq in k_seqs:
                            print("sequence:", k_seq)
                    for k_seq in k_seqs:
                        inds = np.meshgrid(k_seq, k_seq, indexing="ij")
                        z_mat_new = copy.deepcopy(z_mat[inds[0], inds[1]])
                        z_max = z_mat[0, 0] + 1
                        # move further in the tree
                        self._add_layer_b(
                            net,
                            node,
                            z_mat_new,
                            alphas,
                            gammas,
                            z_max,
                            k_seq,
                            dz=dz,
                            pprint=pprint,
                            use_hist=use_hist,
                            add_lin_terms=add_lin_terms,
                        )

    def _add_layer_b(
        self,
        net,
        pnode,
        z_mat,
        alphas,
        gammas,
        z_max_prev,
        true_loc_idxs,
        dz=100.0,
        use_hist=True,
        pprint=False,
        add_lin_terms=False,
    ):
        # print stuff
        if pprint:
            print(">>> node index = ", node._index)
            if pnode != None:
                print("parent index = ", pnode._index)
            else:
                print("start")
        # get the diagonal
        z_diag = np.diag(z_mat)

        if true_loc_idxs[0] == 0 and z_mat[0, 0] > z_max_prev:
            n_bins = "soma"
            z_max = z_mat[0, 0] + 1.0
            z_min = z_max_prev
        else:
            # histogram GF
            n_bins = max(
                int(z_mat.size / 50.0), int((np.max(z_mat) - np.min(z_mat)) / dz)
            )
            if n_bins > 1:
                if np.all(np.diff(z_diag) > 0):
                    z_min = z_max_prev
                    z_max = z_min + dz
                    if pprint:
                        print("--> +", dz)
                elif use_hist:
                    z_hist = np.histogram(z_mat.flatten(), n_bins, density=False)
                    # find the histogram partition
                    h_ftc = hs.histogramSegmentator(z_hist)
                    s_ind, p_ind = h_ftc.partition_fine_to_coarse()

                    # get the new min max values
                    z_histx = z_hist[1]
                    z_min = z_max_prev
                    z_max = z_histx[s_ind[1]]
                    ii = 1
                    while np.min(z_diag) > z_histx[s_ind[ii]]:
                        ii += 1
                        z_max = z_histx[s_ind[ii]]
                    ii = np.argmax(z_hist[0][s_ind[0] : s_ind[ii]])
                    z_avg = z_hist[0][ii]
                    if z_max - z_min > dz:
                        z_max = z_min + dz
                    if pprint:
                        print("--> hist: +", str(z_max - z_min))
                else:
                    z_min = z_max_prev
                    z_max = z_min + dz
                    if pprint:
                        print("--> +", dz)
            else:
                z_min = z_max_prev
                z_max = np.max(z_mat)
                if pprint:
                    print("--> all: +", str(z_max - z_min))
        d_inds = np.where(z_diag <= z_max + 1e-15)[0]
        # make sure that there is at least one element in the layer
        while len(d_inds) == 0:
            z_max += dz
            d_inds = np.where(z_diag <= z_max + 1e-15)[0]

        # identify different domains
        if add_lin_terms and true_loc_idxs[0] == 0:
            t0 = np.array([1])
            t1 = np.array([len(z_diag)])
        else:
            t0 = np.where(
                np.logical_and(z_diag[:-1] < z_max + 1e-15, z_diag[1:] >= z_max + 1e-15)
            )[0]
            if len(t0) > 0:
                t0 += 1
            if z_diag[0] >= z_max + 1e-15:
                t0 = np.concatenate(([0], t0))
            t1 = np.where(
                np.logical_and(z_diag[:-1] >= z_max + 1e-15, z_diag[1:] < z_max + 1e-15)
            )[0]
            if len(t1) > 0:
                t1 += 1
            if z_diag[-1] >= z_max + 1e-15:
                t1 = np.concatenate((t1, [len(z_diag)]))

        # identify where the kernels are within the interval
        l_inds = np.where(z_mat <= z_max + 1e-15)

        # get the average kernel
        if l_inds[0].size < 100000:
            gammas_avg = np.mean(
                gammas[:, true_loc_idxs[l_inds[0]]]
                * gammas[:, true_loc_idxs[l_inds[1]]],
                1,
            )
        else:
            inds_ = np.random.randint(l_inds[0].size, size=100000)
            gammas_avg = np.mean(
                gammas[:, true_loc_idxs[l_inds[0]][inds_]]
                * gammas[:, true_loc_idxs[l_inds[1]][inds_]],
                1,
            )
        self._subtract_parent_kernels(gammas_avg, pnode)

        # add a node to the tree
        node = NETNode(len(net), true_loc_idxs, z_kernel=(alphas, gammas_avg))
        if pnode != None:
            net.add_node_with_parent(node, pnode)
        else:
            net.root = node

        if pprint:
            print("(locind0, size) = ", (true_loc_idxs[0], z_mat.shape[0]))
            print("(zmin, zmax, n_bins) = ", (z_min, z_max, n_bins))
            print("")

        # move on to the next layers
        if len(d_inds) < len(z_diag):
            for jj, ind0 in enumerate(t0):
                ind1 = t1[jj]
                z_mat_new = copy.deepcopy(z_mat[ind0:ind1, ind0:ind1])
                true_loc_idxs_new = true_loc_idxs[ind0:ind1]
                self._add_layer_b(
                    net,
                    node,
                    z_mat_new,
                    alphas,
                    gammas,
                    z_max,
                    true_loc_idxs_new,
                    dz=dz,
                    use_hist=use_hist,
                    pprint=pprint,
                )

    def _subtract_parent_kernels(self, gammas, pnode):
        if pnode != None:
            gammas -= pnode.z_kernel["c"]
            self._subtract_parent_kernels(gammas, pnode.parent_node)

    def _improve_input_impedance(self, net, alphas, gammas):
        nmaxind = np.max([n.index for n in net])
        for node in net:
            if len(node.loc_idxs) == 1:
                # recompute the kernel of this single loc layer
                if node.parent_node is not None:
                    p_kernel = net.calc_total_kernel(node.parent_node)
                    p_k_c = p_kernel.c
                else:
                    p_k_c = np.zeros_like(gammas)
                gammas_real = gammas[:, node.loc_idxs[0]] ** 2
                node.z_kernel.c = gammas_real - p_k_c
            elif len(node.newloc_idxs) > 0:
                z_k_approx = net.calc_total_kernel(node)
                # add new input nodes for the nodes that don't have one
                for ind in node.newloc_idxs:
                    nmaxind += 1
                    gammas_real = gammas[:, ind] ** 2
                    z_k_real = Kernel(dict(a=alphas, c=gammas_real))
                    # add node
                    newnode = NETNode(nmaxind, [ind], z_kernel=z_k_real - z_k_approx)
                    newnode.newloc_idxs = [ind]
                    net.add_node_with_parent(newnode, node)
                # empty the new indices
                node.newloc_idxs = []
        net.set_new_loc_idxs()

    def compute_lin_terms(self, net, sov_data=None, eps=1e-4):
        """
        Construct linear terms for `net` so that transfer impedance to soma is
        exactly matched

        Parameters
        ----------
        net: `neat.NETree`
            the neural evaluation tree (NET)
        sov_data: None or tuple of mode matrices
            If ``sov_data`` is not ``None``, it is a tuple of a vector of
            the reciprocals of the mode timescales and a matrix with the
            corresponding spatial mode functions.
        eps: float
            the cutoff threshold in relative importance below which modes
            are truncated

        Returns
        -------
        lin_terms: dict of {int: `neat.Kernel`}
            the kernels associated with linear terms of the NET, keys are
            indices of their corresponding location stored inder 'net eval'
        """
        if sov_data != None:
            alphas = sov_data[0]
            gammas = sov_data[1]
        else:
            alphas, gammas = self.get_important_modes(loc_arg="net eval", eps=eps)
        lin_terms = {}
        for ii, loc in enumerate(self.get_locs("net eval")):
            if not self.is_root(self[loc["node"]]):
                # create the true kernel
                z_k_true = Kernel((alphas, gammas[:, ii] * gammas[:, 0]))
                # compute the NET approximation kernel
                z_k_net = net.get_reduced_tree([0, ii]).get_root().z_kernel
                # compute the lin term
                lin_terms[ii] = z_k_true - z_k_net

        return lin_terms
