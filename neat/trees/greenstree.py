"""
File contains:

    - `neat.GreensNode`
    - `neat.SomaGreensNode`
    - `neat.GreensTree`

Author: W. Wybo
"""

import numpy as np

import copy
from typing import Literal

from . import morphtree
from .morphtree import MorphLoc
from .phystree import PhysNode, PhysTree
from .netree import Kernel
from ..channels import ionchannels
from ..tools import kernelextraction as ke
from ..factorydefaults import DefaultPhysiology


CFG = DefaultPhysiology()


class GreensNode(PhysNode):
    '''
    Node that stores quantities and defines functions to implement the impedance
    matrix calculation based on Koch's algorithm (Koch & Poggio, 1985).

    Attributes
    ----------
    expansion_points: dict {str: np.ndarray}
        Stores ion channel expansion points for this segment.
    '''
    def __init__(self, index, p3d):
        super().__init__(index, p3d)
        self.expansion_points = {}

    def _rescaleLengthRadius(self):
        self.R_ = self.R * 1e-4 # convert to cm
        self.L_ = self.L * 1e-4 # convert to cm

    def setExpansionPoint(self, channel_name, statevar):
        """
        Set the choice for the state variables of the ion channel around which
        to linearize.

        Note that when adding an ion channel to the node, the default expansion
        point setting is to linearize around the asymptotic values for the state
        variables at the equilibrium potential store in `self.e_eq`.
        Hence, this function only needs to be called to change that setting.

        Parameters
        ----------
        channel_name: string
            the name of the ion channel
        statevar: dict
            The expansion points for each of the ion channel state variables
        """
        if statevar is None:
            statevar = {}
        self.expansion_points[channel_name] = statevar

    def getExpansionPoint(self, channel_name):
        try:
            return self.expansion_points[channel_name]
        except KeyError:
            self.expansion_points[channel_name] = {}
            return self.expansion_points[channel_name]

    def _constructChannelArgs(self, channel):
        """
        Returns the expansion points for the channel, around which the
        linearization in computed.

        For voltage, checks if 'v' key is in `self.expansion_points`, otherwise
        defaults to `self.e_eq`.

        For concentrations, checks if the ion is in `self.expansion_points`,
        otherwise checks if a concentration of the ion is given in
        `self.conc_eqs`, and otherwise defaults to the factory default in
        `neat.channels.ionchannels`.

        Parameters
        ----------
        channel: `neat.IonChannel` object
            the ion channel

        Returns
        v: float or np.ndarray
            The voltage values for the expansion points
        sv: dict ({str: np.ndarray})
            The state variables and/or concentrations at the expansion points.
        """
        # check if linearistation needs to be computed around expansion point
        sv = self.getExpansionPoint(channel.__class__.__name__).copy()

        # if voltage is not in expansion point, use equilibrium potential
        v = sv.pop('v', self.e_eq)

        # if concencentration is in expansion point, use it. Otherwise use
        # concentration in equilibrium concentrations (self.conc_eqs), if
        # it is there. If not, use default concentration.
        ions = [str(ion) for ion in channel.conc] # convert potential sympy symbols to str
        conc = {
            ion: sv.pop(
                    ion, self.conc_eqs.copy().pop(ion, CFG.conc[ion])
                ) \
            for ion in ions
        }
        sv.update(conc)

        return v, sv

    def _calcMembraneImpedance(self, freqs, channel_storage, use_conc=False):
        """
        Compute the impedance of the membrane at the node

        Parameters
        ----------
        freqs: `np.ndarray` (``dtype=complex``, ``ndim=1``)
            The frequencies at which the impedance is to be evaluated
        channel_storage: dict of ion channels (optional)
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized
        use_conc: bool
            if True, also uses concentration mechanisms to compute linearized
            membrane impedance

        Returns
        -------
        `np.ndarray` (``dtype=complex``, ``ndim=1``)
            The membrane impedance
        """
        if use_conc:
            g_m_ions = {ion: np.zeros_like(freqs) for ion in self.concmechs}

        g_m_aux = self.c_m * freqs + self.currents['L'][0]

        # loop over all active channels
        for channel_name in set(self.currents.keys()) - set('L'):
            g, e = self.currents[channel_name]

            if g < 1e-10:
                continue

            # recover the ionchannel object
            channel = channel_storage[channel_name]

            # get voltage(s), state variable expansion point(s) and
            # concentration(s) around which to linearize the channel
            v, sv = self._constructChannelArgs(channel)

            # compute linearized channel contribution to membrane impedance
            g_m_chan = g * channel.computeLinSum(v, freqs, e=e, **sv)

            # add channel contribution to total ionic current
            if use_conc and channel.ion in g_m_ions:
                g_m_ions[channel.ion] = g_m_ions[channel.ion] - g_m_chan

            # add channel contribution to membrane impedance
            g_m_aux = g_m_aux - g_m_chan

        if not use_conc:
            return 1. / (2. * np.pi * self.R_ * g_m_aux)

        for channel_name in set(self.currents.keys()) - set('L'):
            g, e = self.currents[channel_name]

            if g < 1e-10:
                continue

            # recover the ionchannel object
            channel = channel_storage[channel_name]

            for ion in self.concmechs:

                if ion not in channel.conc:
                    continue

                # get voltage(s), state variable expansion point(s) and
                # concentration(s) around which to linearize the channel
                v, sv = self._constructChannelArgs(channel)

                # add concentration contribution to linearized membrane
                # conductance
                g_m_aux = g_m_aux - \
                    g * \
                    channel.computeLinConc(v, freqs, ion, e=e, **sv) * \
                    self.concmechs[ion].computeLinear(freqs) * \
                    g_m_ions[ion]

        return 1. / (2. * np.pi * self.R_ * g_m_aux)


    def _calcChannelStatevarLinearTerms(self, freqs, v_resp, channel_storage):
        """
        Compute linearized responss of the ion channel state variables, given
        the linearized voltage responses

        Parameters
        ----------
        freqs: `np.ndarray` (``dtype=complex``, ``ndim=1``)
            The frequencies at which the impedance is to be evaluated
        v_resp: `np.ndarray` (``dtype=complex``, ``ndim=1``, ``shape=(s,k)``)
            Linearized voltage responses in the frequency domain, evaluated at
            ``s`` frequencies and ``k`` locations
        channel_storage: dict of ion channels (optional)
            The ion channels that have been initialized already. If not
            provided, a new channel is initialized
        use_conc: bool
            if True, also uses concentration mechanisms to compute linearized
            membrane impedance

        Returns
        -------
        `np.ndarray` (``dtype=complex``, ``ndim=1``)
            The membrane impedance
        """
        svar_terms = {}
        # loop over all active channels
        for channel_name in set(self.currents.keys()) - set('L'):

            # recover the ionchannel object
            channel = channel_storage[channel_name]

            # get voltage(s), state variable expansion point(s) and
            # concentration(s) around which to linearize the channel
            v, sv = self._constructChannelArgs(channel)

            # compute linearized channel contribution to membrane impedance
            svar_terms[channel_name] = channel.computeLinStatevarResponse(
                v, freqs, v_resp, **sv
            )

        return svar_terms


    def _setImpedance(self, freqs, channel_storage, use_conc=False):
        self.counter = 0
        self.z_m = self._calcMembraneImpedance(freqs, channel_storage,
                                              use_conc=use_conc)

        self.z_a = self.r_a / (np.pi * self.R_**2)
        self.gamma = np.sqrt(self.z_a / self.z_m)
        self.z_c = self.z_a / self.gamma

    def _setImpedanceDistal(self):
        """
        Set the boundary condition at the distal end of the segment
        """
        if len(self.child_nodes) == 0:
            # note that instantiating z_aux as a float, multiplying with np.infty,
            # and then converting it as a complex results in entries
            # inf + 0.j -- which is desired
            # where instatiating z_aux as complex, and then multiplying with
            # np.infty, would result in
            # inf + nanj -- which is not desired
            z_aux = np.ones(self.z_m.shape, dtype=float)

            if self.g_shunt > 1e-10:
                z_aux /= self.g_shunt
            else:
                z_aux *= np.infty

            self.z_distal = z_aux.astype(self.z_m.dtype)
        else:
            g_aux = np.ones_like(self.z_m) * self.g_shunt

            for cnode in self.child_nodes:
                g_aux = g_aux +  1. / cnode._collapseBranchToRoot()

            # note that a division by zero error is not possible here, since the
            # only case where this occurs would be a node with no child nodes,
            # which is caught in the if statement
            self.z_distal = 1. / g_aux

    def _setImpedanceProximal(self):
        """
        Set the boundary condition at the proximal end of the segment
        """
        # child nodes of parent node without the current node
        sister_nodes = copy.copy(self.parent_node.child_nodes[:])
        sister_nodes.remove(self)
        # compute the impedance
        val = 0.
        if self.parent_node is not None:
            val = val + 1. / self.parent_node._collapseBranchToLeaf()
            val += self.parent_node.g_shunt
        for snode in sister_nodes:
            val = val + 1. / snode._collapseBranchToRoot()
        self.z_proximal = 1. / val

    def _collapseBranchToLeaf(self):
        return self.z_c * (self.z_proximal * np.cosh(self.gamma * self.L_) + \
                           self.z_c * np.sinh(self.gamma * self.L_)) / \
                          (self.z_proximal * np.sinh(self.gamma * self.L_) +
                           self.z_c * np.cosh(self.gamma * self.L_))

    def _collapseBranchToRoot(self):
        zr = self.z_c * (np.cosh(self.gamma * self.L_) +
                         self.z_c / self.z_distal * np.sinh(self.gamma * self.L_)) / \
                        (np.sinh(self.gamma * self.L_) +
                         self.z_c / self.z_distal * np.cosh(self.gamma * self.L_))
        return zr

    def _setImpedanceArrays(self):
        self.gammaL = self.gamma * self.L_
        self.z_cp = self.z_c / self.z_proximal
        self.z_cd = self.z_c / self.z_distal
        self.wrongskian = np.cosh(self.gammaL) / self.z_c * \
                           (self.z_cp + self.z_cd + \
                            (1. + self.z_cp * self.z_cd) * np.tanh(self.gammaL))
        self.z_00 = (self.z_cd * np.sinh(self.gammaL) + np.cosh(self.gammaL)) / \
                    self.wrongskian
        self.z_11 = (self.z_cp * np.sinh(self.gammaL) + np.cosh(self.gammaL)) / \
                    self.wrongskian
        self.z_01 = 1. / self.wrongskian

    def _calcZF(self, x1, x2):
        if x1 <  1e-3 and x2 < 1e-3:
            return self.z_00
        elif x1 > 1.-1e-3 and x2 > 1.-1e-3:
            return self.z_11
        elif (x1 < 1e-3 and x2 > 1.-1e-3) or (x1 > 1.-1e-3 and x2 < 1e-3):
            return self.z_01
        elif x1 < x2:
            return (self.z_cp * np.sinh(self.gammaL*x1) + np.cosh(self.gammaL*x1)) * \
                   (self.z_cd * np.sinh(self.gammaL*(1.-x2)) + np.cosh(self.gammaL*(1.-x2))) / \
                   self.wrongskian
        else:
            return (self.z_cp * np.sinh(self.gammaL*x2) + np.cosh(self.gammaL*x2)) * \
                   (self.z_cd * np.sinh(self.gammaL*(1.-x1)) + np.cosh(self.gammaL*(1.-x1))) / \
                   self.wrongskian


class SomaGreensNode(GreensNode):
    def _calcMembraneImpedance(self, freqs, channel_storage, use_conc=False):
        z_m = super()._calcMembraneImpedance(freqs, channel_storage,
                                                                use_conc=use_conc)
        # rescale for soma surface instead of cylinder radius
        # return z_m / (2. * self.R_)
        return 1. / (2. * self.R_ / z_m + self.g_shunt)

    def _setImpedance(self, freqs, channel_storage, use_conc=False):
        self.counter = 0
        self.z_soma = self._calcMembraneImpedance(freqs, channel_storage,
                                                 use_conc=use_conc)

    def _collapseBranchToLeaf(self):
        return self.z_soma

    def _setImpedanceArrays(self):
        val = 1. / self.z_soma
        for node in self.child_nodes:
            val = val + 1. / node._collapseBranchToRoot()
        self.z_in = 1. / val

    def _calcZF(self, x1, x2):
        return self.z_in


class GreensTree(PhysTree):
    """
    Class that computes the Green's function in the Fourrier domain of a given
    neuronal morphology (Koch, 1985). This three defines a special
    `neat.SomaGreensNode` as a derived class from `neat.GreensNode` as some
    functions required for Green's function calculation are different and thus
    overwritten.

    The calculation proceeds on the computational tree (see docstring of
    `neat.MorphNode`). Thus it makes no sense to look for Green's function
    related quantities in the original tree.

    Attributes
    ----------
    freqs: np.array of complex
        Frequencies at which impedances are evaluated ``[Hz]``
    """
    def __init__(self, file_n=None, types=[1,3,4]):
        super().__init__(file_n=file_n, types=types)
        self.freqs = None

    def _createCorrespondingNode(self, node_index, p3d=None):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
        node_index: `int`
            index of the new node
        """
        if node_index == 1:
            return SomaGreensNode(node_index, p3d)
        else:
            return GreensNode(node_index, p3d)

    def removeExpansionPoints(self):
        """
        Remove expansion points from all nodes in the tree
        """
        for node in self:
            node.expansion_points = {}

    @morphtree.computationalTreetypeDecorator
    def setImpedance(self, freqs, use_conc=False, pprint=False):
        """
        Set the boundary impedances for each node in the tree

        Parameters
        ----------
        freqs: `np.ndarray` (``dtype=complex``, ``ndim=1``)
            frequencies at which the impedances will be evaluated ``[Hz]``.
        use_conc: bool
            whether or not to incorporate concentrations in the calculation
        pprint: bool (default ``False``)
            whether or not to print info on the progression of the algorithm

        """
        # cast to array to make sure there is always a shape attribute
        self.freqs = np.array(freqs)

        # set the node specific impedances
        for node in self:
            node._rescaleLengthRadius()
            node._setImpedance(self.freqs, self.channel_storage, use_conc=use_conc)
        # recursion
        if len(self) > 1:
            self._impedanceFromLeaf(self.leafs[0], self.leafs[1:], pprint=pprint)
            self._impedanceFromRoot(self.root)
        # clean
        for node in self:
            node.counter = 0
            node._setImpedanceArrays()

    def _impedanceFromLeaf(self, node, leafs, pprint=False):
        if pprint:
            print('Forward sweep: ' + str(node))
        pnode = node.parent_node
        # log how many times recursion has passed at node
        if not self.isLeaf(node):
            node.counter += 1
        # if the number of childnodes of node is equal to the amount of times
        # the recursion has passed node, the distal impedance can be set. Otherwise
        # we start a new recursion at another leaf.
        if node.counter == len(node.child_nodes):
            if not self.isRoot(node):
                node._setImpedanceDistal()
                self._impedanceFromLeaf(pnode, leafs, pprint=pprint)
        elif len(leafs) > 0:
                self._impedanceFromLeaf(leafs[0], leafs[1:], pprint=pprint)

    def _impedanceFromRoot(self, node):
        if node != self.root:
            node._setImpedanceProximal()
        for cnode in node.child_nodes:
            self._impedanceFromRoot(cnode)

    @morphtree.computationalTreetypeDecorator
    def calcZF(self, loc1, loc2):
        """
        Computes the transfer impedance between two locations for all frequencies
        in `self.freqs`.

        Parameters
        ----------
        loc1: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer impedance is computed
        loc2: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer impedance is computed

        Returns
        -------
        nd.ndarray (dtype = complex, ndim = 1)
            The transfer impedance ``[MOhm]`` as a function of frequency
        """
        # cast to morphlocs
        loc1 = MorphLoc(loc1, self)
        loc2 = MorphLoc(loc2, self)
        # the path between the nodes
        path = self.pathBetweenNodes(self[loc1['node']], self[loc2['node']])
        # compute the kernel
        z_f = np.ones_like(self.root.z_in)
        if len(path) == 1:
            # both locations are on same node
            z_f *= path[0]._calcZF(loc1['x'], loc2['x'])
        else:
            # different cases whether path goes to or from root
            if path[1] == self[loc1['node']].parent_node:
                z_f *= path[0]._calcZF(loc1['x'], 0.)
            else:
                z_f *= path[0]._calcZF(loc1['x'], 1.)
                z_f /= path[0]._calcZF(1., 1.)
            if path[-2] == self[loc2['node']].parent_node:
                z_f *= path[-1]._calcZF(loc2['x'], 0.)
            else:
                z_f *= path[-1]._calcZF(loc2['x'], 1.)
                z_f /= path[-1]._calcZF(1., 1.)
            # nodes within the path
            ll = 1
            for node in path[1:-1]:
                z_f /= node._calcZF(1., 1.)
                if path[ll-1] not in node.child_nodes or \
                   path[ll+1] not in node.child_nodes:
                    z_f *= node._calcZF(0., 1.)
                ll += 1

        return z_f

    @morphtree.computationalTreetypeDecorator
    def calcImpedanceMatrix(self, locarg, explicit_method=True):
        """
        Computes the impedance matrix of a given set of locations for each
        frequency stored in `self.freqs`.

        Parameters
        ----------
        locarg: `list` of locations or string
            if `list` of locations, specifies the locations for which the
            impedance matrix is evaluated, if ``string``, specifies the
            name under which a set of location is stored
        explicit_method: bool, optional (default ``True``)
            if ``False``, will use the transitivity property of the impedance
            matrix to further optimize the computation.

        Returns
        -------
        `np.ndarray` (``dtype = self.freqs.dtype``, ``ndim = 3``)
            the impedance matrix, first dimension corresponds to the
            frequency, second and third dimensions contain the impedance
            matrix ``[MOhm]`` at that frequency
        """
        locs = self._convertLocArgToLocs(locarg)

        n_loc = len(locs)
        z_mat = np.zeros((n_loc, n_loc) + self.root.z_in.shape,
                         dtype=self.root.z_in.dtype)

        if explicit_method:
            for ii, loc0 in enumerate(locs):
                # diagonal elements
                z_f = self.calcZF(loc0, loc0)
                z_mat[ii,ii] = z_f

                # off-diagonal elements
                jj = 0
                while jj < ii:
                    loc1 = locs[jj]
                    z_f = self.calcZF(loc0, loc1)
                    z_mat[ii,jj] = z_f
                    z_mat[jj,ii] = z_f
                    jj += 1
        else:
            for ii in range(len(locs)):
                self._calcImpedanceMatrixFromNode(ii, locs, z_mat)

        return np.moveaxis(z_mat, [0, 1], [-1, -2])

    def _calcImpedanceMatrixFromNode(self, ii, locs, z_mat):
        node = self[locs[ii]['node']]
        for jj, loc in enumerate(locs):
            if loc['node'] == node.index and jj >= ii:
                z_new = node._calcZF(locs[ii]['x'],loc['x'])
                z_mat[ii,jj] = z_new
                z_mat[jj,ii] = z_new

        # move down
        for c_node in node.child_nodes:
            z_new = node._calcZF(locs[ii]['x'], 1.)
            self._calcImpedanceMatrixDown(ii, z_new, c_node, locs, z_mat)

        if node.parent_node is not None:
            z_new = node._calcZF(locs[ii]['x'], 0.)
            # move to sister nodes
            for c_node in set(node.parent_node.child_nodes) - {node}:
                self._calcImpedanceMatrixDown(ii, z_new, c_node, locs, z_mat)
            # move up
            self._calcImpedanceMatrixUp(ii, z_new, node.parent_node, locs, z_mat)

    def _calcImpedanceMatrixUp(self, ii, z_0, node, locs, z_mat):
        # compute impedances
        z_in = node._calcZF(1.,1.)
        for jj, loc in enumerate(locs):
            if jj > ii and loc['node'] == node.index:
                z_new = z_0 / z_in * node._calcZF(1.,loc['x'])
                z_mat[ii,jj] = z_new
                z_mat[jj,ii] = z_new

        if node.parent_node is not None:
            z_new = z_0 / z_in * node._calcZF(0., 1.)
            # move to sister nodes
            for c_node in set(node.parent_node.child_nodes) - {node}:
                self._calcImpedanceMatrixDown(ii, z_new, c_node, locs, z_mat)
            # move to parent node
            z_new = z_0 / z_in * node._calcZF(0., 1.)
            self._calcImpedanceMatrixUp(ii, z_new, node.parent_node, locs, z_mat)

    def _calcImpedanceMatrixDown(self, ii, z_0, node, locs, z_mat):
        # compute impedances
        z_in = node._calcZF(0.,0.)
        for jj, loc in enumerate(locs):
            if jj > ii and loc['node'] == node.index:
                z_new = z_0 / z_in * node._calcZF(0., loc['x'])
                z_mat[ii,jj] = z_new
                z_mat[jj,ii] = z_new

        # recurse to child nodes
        z_new = z_0 / z_in * node._calcZF(0., 1.)
        for c_node in node.child_nodes:
            self._calcImpedanceMatrixDown(ii, z_new, c_node, locs, z_mat)

    @morphtree.computationalTreetypeDecorator
    def calcChannelResponseF(self, loc1, loc2):
        """
        Compute linearized ion channel state variable responses in the frequency
        domain  at `loc2` to a delta current pulse input at `loc1`.

        Parameters
        ----------
        loc1: Tuple(int, float) or `neat.MorphLoc`
            the location of the delta input current pulse
        loc2: Tuple(int, float) or `neat.MorphLoc`
            location of the ion channel response

        Returns
        -------
        dict of dict of `np.ndarray`
            The linearized responses of all channels at loc2 to the delta
            current pulse input. Can be accessed as:
            [channel_name][statevar_name][frequency]
        """
        # cast to morphlocs
        loc1 = MorphLoc(loc1, self)
        loc2 = MorphLoc(loc2, self)

        # compute channel responses
        c_resp = self[loc2['node']]._calcChannelStatevarLinearTerms(
            self.freqs, self.calcZF(loc1, loc2), self.channel_storage
        )

        return c_resp

    @morphtree.computationalTreetypeDecorator
    def calcChannelResponseMatrix(self, locarg):
        """
        Compute linearized ion channel state variable response matrix in the
        frequency domain at all locations in `locarg` to delta current pulse
        inputs at each of those loctions.

        Note that the matrix is returned as a list of nested dictionaries.

        Parameters
        ----------
        locarg: `list` of locations or string
            if `list` of locations, specifies the locations for which the
            ion channel responses are evaluated, if ``string``, specifies the
            name under which a set of location is stored

        Returns
        -------
        List of dict of dict of `np.ndarray`
            The linearized responses of all channels to current pulse input,
            can be accessed as
            [location_index][channel_name][statevar_name][frequency, input loc]
        """
        locs = self._convertLocArgToLocs(locarg)
        z_mat = self.calcImpedanceMatrix(locs)

        channel_responses = []
        for ii, loc in enumerate(locs):
            c_resp = self[loc['node']]._calcChannelStatevarLinearTerms(
                self.freqs, z_mat[:,ii,:], self.channel_storage
            )
            channel_responses.append(c_resp)

        return channel_responses


class GreensTreeTime(GreensTree):
    freqs_vfit = None
    ft = None

    _slice_vfit = None
    _slice_quad = None

    def __init__(self, file_n=None, types=[1,3,4]):
        super().__init__(file_n=file_n, types=types)

    def _checkInstantiated(self):
        if self.freqs_vfit is None or self.ft is None:
            raise AttributeError(
                "Frequency arrays are not instatiated, call `setImpedance()` first"
            )

    def _setDefaultFreqarrayVectorFit(self):
        # reasonable parameters to construct frequency array
        dt = 0.1*1e-3 # s
        N = 2**12
        smax = np.pi/dt # Hz
        ds = np.pi/(N*dt) # Hz

        # frequency array for vector fitting
        self.freqs_vfit = np.arange(-smax, smax, ds)*1j  # Hz

    def _setDefaultFreqarrayQuadrature(self, t_inp):
        # frequency array for time domain evaluation of the kernels through
        # quadrature is contained in `FourrierTools.s`
        if isinstance(t_inp, ke.FourrierTools):
            self.ft = t_inp
        else:
            # reasonable parameters for FourrierTools
            self.ft = ke.FourrierTools(t_inp, fmax=7., base=10., num=200)

    def setImpedance(self, t_inp):
        self._setDefaultFreqarrayVectorFit()
        self._setDefaultFreqarrayQuadrature(t_inp)

        # frequencies `self.freqs` will be set by parent class `setImpedance()`
        # function. We define the slices that index them seperately already
        self._slice_vfit = np.s_[:len(self.freqs_vfit)]
        self._slice_quad = np.s_[len(self.freqs_vfit):]
        super().setImpedance(np.concatenate((self.freqs_vfit, self.ft.s)))

    def _inverseFourrier(self, func_vals_f,
            method: Literal["", "exp fit", "quadrature"] = "",
            compute_time_derivative=True,
        ):
        if method not in ["", "exp fit", "quadrature"]:
            raise IOError(
                "Method should be empty string, 'exp fit' or 'quadrature'"
            )

        # compute in time domain, method depends on ratio between spectral
        # power in zero frequency vs max frequency component
        # typically, this will mean exponential fit is chosen for input
        # impedances and explicit quadrature for transfer impedances
        f_arr = func_vals_f[self._slice_quad]
        criterion_eval = np.abs(f_arr[-1]) / np.abs(f_arr[self.ft.ind_0s])
        criterion = criterion_eval <= 1e-3

        if criterion_eval > 1e-10:
            # if there is substantial spectral power in the max frequency
            # components, we smooth the function with a squared cosine window
            # to reduce oscillations
            window = np.cos(np.pi * self.ft.s.imag / (2.*np.abs(self.ft.s[-1])))**2
        else:
            window = np.ones_like(self.ft.s)

        # compute kernel through quadrature method
        func_vals_t = self.ft.ftInv(
            window * func_vals_f[self._slice_quad]
        )[1].real * 1e-3 # MOhm/s -> MOhm/ms
        if compute_time_derivative:
            # compute differentiated kernel
            dfunc_vals_t_dt = self.ft.ftInv(
                self.ft.s * window * func_vals_f[self._slice_quad]
            )[1].real * 1e-6 # MOhm/s^2 -> MOhm/ms^2

        # when the criterion is satified, or if the default method is
        # overridden to 'quadrature', we always return the the quadrature result
        if (method == "" and criterion ) or method == "quadrature":
            if compute_time_derivative:
                return func_vals_t, dfunc_vals_t_dt
            else:
                return func_vals_t

        # this code will only be reached when `method` is "exp_fit", or when
        # `method` is "" but the criterion is not satisfied

        # we set a custom set of initial poles for the vector fit algorithm
        initpoles = np.concatenate((
            np.linspace(.5, 10**1.3, 40)[:-1],
            np.logspace(
                1.3, np.log10(self.freqs[self._slice_vfit][-1].imag),
                num=40, base=10,
            )
        ))

        # compute kernel as superposition of exponentials in the frequency domain
        f_exp_fitter = ke.fExpFitter()
        alpha, gamma, pairs, rms = f_exp_fitter.fitFExp(
            self.freqs[self._slice_vfit], func_vals_f[self._slice_vfit],
            deg=40, initpoles=initpoles,
            realpoles=True, zerostart=False,
            constrained=True, reduce_numexp=False
        )
        zk = Kernel({'a': alpha*1e-3, 'c': gamma*1e-3})
        # linear fit of c in the time domain to the quadrature-computed kernels
        # can improve accuracy
        w = np.concatenate(
            (self.ft.t[self.ft.t < 1.], np.ones_like(self.ft.t[self.ft.t >= 1.]))
        )
        zk.fit_c(self.ft.t, func_vals_t, w=w)

        # evaluate kernel in the time domain
        func_vals_t = zk(self.ft.t)

        if compute_time_derivative:
            # compute differentiated kernel
            dfunc_vals_t_dt = zk.diff(self.ft.t)

            return func_vals_t, dfunc_vals_t_dt

        else:
            return func_vals_t

    @morphtree.computationalTreetypeDecorator
    def calcZT(self, loc1, loc2,
            method: Literal["", "exp fit", "quadrature"] = "",
            compute_time_derivative=True,
            _zf=None,
        ):
        """
        Computes the impulse response kernel between two locations for all
        time points in `self.ft.t` (the input times provided to `setImpedance()`).

        Parameters
        ----------
        loc1: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer impedance is computed
        loc2: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer impedance is computed
        method: str ("", "exp fit", "quadrature")
            The method to use when computing the kernel. "quadrature" for
            explicit integration of the inverse Fourrier integral, "exp fit" for
            a frequency domain fit with the Fourrier transforms of time domain
            exponentials, or "" choses the most appropriate method based on the
            case
        compute_time_derivative: bool
            if ``True``, also returns the time derivatives of the kernel

        Returns
        -------
        nd.ndarray (dtype = complex, ndim = 1)
            The transfer impedance ``[MOhm]`` as a function of frequency
        """
        self._checkInstantiated()

        # compute impedances in the frequency domain
        zf = self.calcZF(loc1, loc2) if _zf is None else _zf

        # convert frequency impedances to time domain kernels
        return self._inverseFourrier(zf, method=method,
            compute_time_derivative=compute_time_derivative
        )

    @morphtree.computationalTreetypeDecorator
    def calcImpulseResponseMatrix(self, locarg,
            method: Literal["", "exp fit", "quadrature"] = "",
            compute_time_derivative=False,
        ):
        """
        Computes the matrix of impulse response kernels at a given set of
        locations for all time-points defined in `self.ft.t` (the input times
        provided to `setImpedance()`).

        Parameters
        ----------
        locarg: `list` of locations or string
            if `list` of locations, specifies the locations for which the
            impulse response kernels are evaluated, if ``string``, specifies the
            name under which a set of location is stored
        method: str ("", "exp fit", "quadrature")
            The method to use when computing the kernels. "quadrature" for
            explicit integration of the inverse Fourrier integral, "exp fit" for
            a frequency domain fit with the Fourrier transforms of time domain
            exponentials, or "" choses the most appropriate method based on the
            case
        compute_time_derivative: bool
            if ``True``, also returns the time derivatives of the kernels

        Returns
        -------
        `np.ndarray` (``dtype = self.freqs.dtype``, ``ndim = 3``)
            the matrix of impulse responses, first dimension corresponds to the
            time axis, second and third dimensions contain the impulse response
            in ``[MOhm/ms]`` at that frequency
        """
        self._checkInstantiated()
        locs = self._convertLocArgToLocs(locarg)

        nt = len(self.ft.t) # number of time points
        nl = len(locs) # number of locations

        # compute impedance matrix in frequency domain
        zf_mat = self.calcImpedanceMatrix(locs)

        zt_mat = np.zeros((nt, nl, nl))
        if compute_time_derivative:
            dzt_dt_mat = np.zeros((nt, nl, nl))

        for ii, loc1 in enumerate(locs):
            for jj, loc2 in enumerate(locs):

                if jj > ii:
                    break

                if compute_time_derivative:
                    zt_mat[:, ii, jj], dzt_dt_mat[:, ii, jj] = self.calcZT(
                        loc1, loc2,
                        compute_time_derivative=True,
                        _zf=zf_mat[:, ii, jj],
                        method=method,
                    )
                    dzt_dt_mat[:, jj, ii] = dzt_dt_mat[:, ii, jj]

                else:
                    zt_mat[:, ii, jj] = self.calcZT(
                        loc1, loc2,
                        compute_time_derivative=False,
                        _zf=zf_mat[:, ii, jj],
                        method=method,
                    )

                zt_mat[:, jj, ii] = zt_mat[:, ii, jj]

        if compute_time_derivative:
            return zt_mat, dzt_dt_mat
        else:
            return zt_mat

    @morphtree.computationalTreetypeDecorator
    def calcChannelResponseT(self, loc1, loc2,
            method: Literal["", "exp fit", "quadrature"] = "",
            compute_time_derivative=False,
            _crf=None,
        ):
        """
        Compute linearized ion channel state variable responses in the time
        domain  at `loc2` to a delta current pulse input at `loc1`.

        Evaluated time-points are the ones in `self.ft.t` (the input times
        provided to `setImpedance()`).

        Parameters
        ----------
        loc1: Tuple(int, float) or `neat.MorphLoc`
            the location of the delta input current pulse
        loc2: Tuple(int, float) or `neat.MorphLoc`
            location of the ion channel response
        method: str ("", "exp fit", "quadrature")
            The method to use when computing the kernels. "quadrature" for
            explicit integration of the inverse Fourrier integral, "exp fit" for
            a frequency domain fit with the Fourrier transforms of time domain
            exponentials, or "" choses the most appropriate method based on the
            case
        compute_time_derivative: bool
            if ``True``, also returns the time derivatives of the kernels

        Returns
        -------
        dict of dict of `np.ndarray`
            The linearized responses of all channels at loc2 to the delta
            current pulse input. Can be accessed as:
            [channel_name][statevar_name][time]
        """
        self._checkInstantiated()
        loc1 = MorphLoc(loc1, self)
        loc2 = MorphLoc(loc2, self)

        # compute impedances in the frequency domain
        crf = self.calcChannelResponseF(loc1, loc2) if _crf is None else _crf

        crt, dcrt_dt = {}, {}
        for channel_name in crf:

            crt[channel_name] = {}
            if compute_time_derivative:
                dcrt_dt[channel_name] = {}

            for svar_name in crf[channel_name]:

                crt[channel_name][svar_name] = {}
                if compute_time_derivative:
                    dcrt_dt[channel_name][svar_name] = {}

                    # convert frequency impedances to time domain kernels
                    (
                        crt[channel_name][svar_name],
                        dcrt_dt[channel_name][svar_name]
                    ) = self._inverseFourrier(
                        crf[channel_name][svar_name],
                        method=method,
                        compute_time_derivative=compute_time_derivative,
                    )

                else:
                    # convert frequency impedances to time domain kernels
                    crt[channel_name][svar_name] = self._inverseFourrier(
                        crf[channel_name][svar_name],
                        method=method,
                        compute_time_derivative=compute_time_derivative,
                    )

        if compute_time_derivative:
            return crt, dcrt_dt
        else:
            return crt

    def calcChannelResponseMatrix(self, locarg, compute_time_derivative=False):
        """
        Compute linearized ion channel state variable response matrix at all
        locations in `locarg` to delta current pulse inputs at each of those
        loctions.

        Evaluated time-points are the ones in `self.ft.t` (the input times
        provided to `setImpedance()`).

        Note that the matrix is returned as a list of nested dictionaries.

        Parameters
        ----------
        locarg: `list` of locations or string
            if `list` of locations, specifies the locations for which the
            ion channel responses are evaluated, if ``string``, specifies the
            name under which a set of location is stored
        method: str ("", "exp fit", "quadrature")
            The method to use when computing the kernels. "quadrature" for
            explicit integration of the inverse Fourrier integral, "exp fit" for
            a frequency domain fit with the Fourrier transforms of time domain
            exponentials, or "" choses the most appropriate method based on the
            case
        compute_time_derivative: bool
            if ``True``, also returns the time derivatives of the kernels

        Returns
        -------
        List of dict of dict of `np.ndarray`
            The linearized responses of all channels to current pulse input,
            can be accessed as
            [output loc index][channel name][statevar name][time, input loc index]
        """
        locs = self._convertLocArgToLocs(locarg)

        nt = len(self.ft.t) # number of time points
        nl = len(locs) # number of locations

        crt_mat, dcrt_dt_mat = [{} for _ in locs], [{} for _ in locs]
        for ii, loc_out in enumerate(locs):
            for jj, loc_in in enumerate(locs):

                if compute_time_derivative:
                    crt_loc1, dcrt_dt_loc1 = self.calcChannelResponseT(
                        loc_in, loc_out,
                        compute_time_derivative=True,
                        method="quadrature",
                    )
                else:
                    crt_loc1 = self.calcChannelResponseT(
                        loc_in, loc_out,
                        compute_time_derivative=False,
                        method="quadrature",
                    )

                # in the first loop iteration, we initialize all dictionary elements
                # for this location as arrays filled with zeros
                if jj == 0:
                    for channel_name in crt_loc1:
                        crt_mat[ii][channel_name] = {}
                        for svar_name in crt_loc1[channel_name]:
                            crt_mat[ii][channel_name][svar_name] = np.zeros((nt, nl))

                    if compute_time_derivative:
                        for channel_name in dcrt_dt_loc1:
                            dcrt_dt_mat[ii][channel_name] = {}
                            for svar_name in crt_loc1[channel_name]:
                                crt_mat[ii][channel_name][svar_name] = np.zeros((nt, nl))

                # we fill the arrays with the time domain ion channel responses
                for channel_name in crt_loc1:
                    for svar_name in crt_loc1[channel_name]:
                        crt_mat[ii][channel_name][svar_name][:,jj] = \
                            crt_loc1[channel_name][svar_name]

                if compute_time_derivative:
                    for channel_name in dcrt_dt_loc1:
                        for svar_name in dcrt_dt_loc1[channel_name]:
                            dcrt_dt_mat[ii][channel_name][svar_name][:,jj] = \
                                dcrt_dt_loc1[channel_name][svar_name]

        if compute_time_derivative:
            return crt_mat, dcrt_dt_mat
        else:
            return crt_mat
