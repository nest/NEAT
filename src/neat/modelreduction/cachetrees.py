# -*- coding: utf-8 -*-
#
# cachetrees.py
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

import dill
import numpy as np

import os
import pickle
import warnings


from ..trees.morphtree import computational_tree_decorator, MorphLoc
from ..trees.phystree import PhysTree
from ..trees.greenstree import GreensTree, GreensTreeTime
from ..trees.sovtree import SOVTree
from ..trees.netree import NET, NETNode

try:
    from ..simulations.neuron import neuronmodel as neurm
except ModuleNotFoundError:
    warnings.warn(
        "NEURON not available, equilibrium evaluation not working", UserWarning
    )


def consecutive(inds):
    """
    split a list of ints into consecutive sublists
    """
    return np.split(inds, np.where(np.diff(inds) != 1)[0] + 1)


class CachedTree(PhysTree):
    def __init__(
        self,
        *args,
        recompute_cache=None,
        save_cache=None,
        cache_name=None,
        cache_path=None,
        **kwargs,
    ):
        # we want a behaviour where the cache parameters are initialized to certain defauls
        # if they are not provided, but where the initialization operation based on copying
        # the tree leaves the cache parameters intact if the input tree is a subclass
        # of CachedTree. However, we also want the optionally provided arguments to be
        # overwritten to their provided values. The constructor below achieve this
        self.set_cache_params(
            **self.get_cache_defaults(
                recompute_cache=recompute_cache,
                save_cache=save_cache,
                cache_name=cache_name,
                cache_path=cache_path,
            )
        )
        super().__init__(*args, **kwargs)
        self.set_cache_params(
            recompute_cache=recompute_cache,
            save_cache=save_cache,
            cache_name=cache_name,
            cache_path=cache_path,
        )

    def get_cache_defaults(
        self,
        recompute_cache=None,
        save_cache=None,
        cache_name=None,
        cache_path=None,
    ):
        cache_params = {}
        cache_params["recompute_cache"] = (
            False if recompute_cache is None else recompute_cache
        )
        cache_params["save_cache"] = True if save_cache is None else save_cache
        cache_params["cache_name"] = "" if cache_name is None else cache_name
        cache_params["cache_path"] = "neatcache/" if cache_path is None else cache_path
        return cache_params

    def set_cache_params(
        self,
        recompute_cache=None,
        save_cache=None,
        cache_name=None,
        cache_path=None,
    ):
        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)

        if cache_name is not None:
            self.cache_name = cache_name
        if cache_path is not None:
            self.cache_path = cache_path
        if save_cache is not None:
            self.save_cache = save_cache
        if recompute_cache is not None:
            self.recompute_cache = recompute_cache

    def get_cache_params(self):
        return {
            "cache_name": self.cache_name,
            "cache_path": self.cache_path,
            "save_cache": self.save_cache,
            "recompute_cache": self.recompute_cache,
        }

    def get_cache_params_in_dict(self, kwarg_dict):
        return {
            key: val
            for key, val in kwarg_dict.iteritems()
            if key in {"cache_name", "cache_path", "save_cache", "recompute_cache"}
        }

    def get_attributes_excluded_from_cache_override(self):
        """
        Returns a list of attributes that should NOT be overwritten by the cashed tree

        Returns
        -------
        list of str
            Attribute names that should not be overwritten
        """
        return ["cache_name", "cache_path", "save_cache", "recompute_cache"]

    def maybe_execute_funcs(
        self,
        funcs_args_kwargs=[],
        pprint=False,
    ):
        file_name = os.path.join(
            self.cache_path,
            f"{self.cache_name}_cache_{self.unique_hash()}.p",
        )

        if pprint:
            print(
                f"\n>>>> Cache file for {self.__class__.__name__}:\n    {file_name}\n<<<<"
            )

        try:
            # ensure that the funcs are recomputed if 'recompute' is true
            if self.recompute_cache:
                raise IOError

            with open(file_name, "rb") as file:
                tree_ = dill.load(file)

            for key in self.get_attributes_excluded_from_cache_override():
                tree_.__dict__[key] = self.__dict__[key]

            self.__dict__.update(tree_.__dict__)
            del tree_

        except (Exception, IOError, EOFError, KeyError) as err:
            if pprint:
                if self.recompute_cache:
                    logstr = ">>> Force recomputing cache..."
                else:
                    logstr = ">>> No cache found, recomputing..."
                print(logstr)

            # execute the functions
            for func, args, kwargs in funcs_args_kwargs:
                func(*args, **kwargs)

            if self.save_cache:
                with open(file_name, "wb") as file:
                    dill.dump(self, file)


class EquilibriumTree(CachedTree):
    """
    Subclass of `neat.PhysTree` that allows for the calculation of the
    equilibrium potential at each node.

    Uses the NEURON simulator to evaluate the equibrium potentials. Can cache
    the results of the computation.

    The equilibrium potential is stored under the `v_ep` attribute of each node.
    """

    def _calc_e_eq(self, loc_arg, ions=None, t_max=500.0, dt=0.1, factor_lambda=10.0):
        """
        Calculates equilibrium potentials and concentrations in the tree.
        Computes the equilibria through a NEURON simulations without inputs.

        Parameters
        ----------
        loc_arg: `list` of locations or string
            if `list` of locations, specifies the locations for which the
            equilibrium state evaluated, if ``string``, specifies the
            name under which a set of locations is stored
        ions: `iterable` of `str
            the names of the ions for which the concentration needs to be measured
        t_max: float
            duration of the simulation
        dt: float
            time-step of the simulation
        factor_lambda: `float`
            multiplies the number of compartments suggested by the lambda-rule
        """
        locs = self.convert_loc_arg_to_locs(loc_arg)
        if ions is None:
            ions = self.ions
        # use longer simulation for Eeq fit if concentration mechansims are present
        t_max = t_max * 20.0 if len(ions) > 0 else t_max

        # create a biophysical simulation model
        sim_tree_biophys = neurm.NeuronSimTree(self)

        # compute equilibrium potentials
        sim_tree_biophys.init_model(dt=dt, factor_lambda=factor_lambda)
        sim_tree_biophys.store_locs(locs, "rec locs", warn=False)
        res_biophys = sim_tree_biophys.run(
            t_max, dt_rec=20.0, record_concentrations=ions
        )
        sim_tree_biophys.delete_model()

        return (
            np.array([v_m[-1] for v_m in res_biophys["v_m"]]),
            {
                ion: np.array([ion_eq[-1] for ion_eq in res_biophys[ion]])
                for ion in ions
            },
        )

    def calc_e_eq(
        self, loc_arg, ions=None, method="interp", L_eps=50.0, pprint=False, **kwargs
    ):
        """
        Calculates equilibrium potentials and concentrations in the tree.

        Uses either linear interpolations between the stored equilibria at the
        midpoints of the nodes or computes the equilibria through a NEURON
        simulation without inputs.

        Parameters
        ----------
        loc_arg: `list` of locations or string
            if `list` of locations, specifies the locations for which the
            equilibrium state evaluated, if ``string``, specifies the
            name under which a set of locations is stored
        ions: `iterable` of `str
            the names of the ions for which the concentration needs to be measured
        method: Literal: 'interp' or 'sim'
            whether to use interpolation or simulation. Defaults to simulation if
            distance is larger than `L_eps`
        L_eps: float
            maximum distance (um) above which the method defaults to interpolation
        pprint: bool
            Whether or not to print additional information
        """
        if ions is None:
            ions = self.ions
        locs = self.convert_loc_arg_to_locs(loc_arg)
        ref_locs = [(n.index, 0.5) for n in self]
        self.store_locs(ref_locs, name="ref locs")

        e_eqs = []
        conc_eqs = {ion: [] for ion in ions}

        if method == "interp":
            if pprint:
                print("> computing e_eq through interpolation")

            idxs0 = self.get_nearest_loc_idxs(locs, "ref locs", direction=1)
            idxs1 = self.get_nearest_loc_idxs(locs, "ref locs", direction=2)

            for loc, idx0, idx1 in zip(locs, idxs0, idxs1):

                if idx0 is None or idx1 is None:
                    if idx0 is None and idx1 is None:
                        # ref locs probably not defined, computations should be redone
                        break

                    idx = idx0 if idx0 is not None else idx1

                    # locs[idx0] more distal than leaf ref loc
                    e_eqs.append(self[ref_locs[idx][0]].v_ep)

                    for ion in ions:
                        conc_eqs[ion].append(self[ref_locs[idx][0]].conc_eps[ion])

                else:
                    L0 = self.path_length(loc, ref_locs[idx0])
                    L1 = self.path_length(loc, ref_locs[idx1])

                    if L0 < 1e-10 or L1 < 1e-10:
                        idx = idx0 if L0 < L1 else idx1
                        # both neighbour locations are the same
                        e_eqs.append(self[ref_locs[idx][0]].v_ep)

                        for ion in ions:
                            # linear interpolation to compute the equilibrium concentration
                            conc_eqs[ion].append(self[ref_locs[idx][0]].conc_eps[ion])

                    elif L0 < L_eps and L1 < L_eps:
                        v_ep0 = self[ref_locs[idx0][0]].v_ep
                        v_ep1 = self[ref_locs[idx1][0]].v_ep

                        # linear interpolation to compute the equilibrium potential
                        e_eqs.append((v_ep0 * L1 + v_ep1 * L0) / (L1 + L0))

                        for ion in ions:
                            c_ep0 = self[ref_locs[idx0][0]].conc_eps[ion]
                            c_ep1 = self[ref_locs[idx0][0]].conc_eps[ion]

                            # linear interpolation to compute the equilibrium concentration
                            conc_eqs[ion].append((c_ep0 * L1 + c_ep1 * L0) / (L1 + L0))

                    else:
                        break

        if len(e_eqs) < len(locs):
            if pprint:
                print("> computing e_eq through interpolation failed, simulating")
            return self._calc_e_eq(loc_arg, ions=ions, **kwargs)

        else:
            if pprint:
                print("> equilibria:")
                for ii, loc in enumerate(locs):
                    conc_eq_str = str(
                        {ion: f"{conc_eq[ii]:.8f}" for ion, conc_eq in conc_eqs.items()}
                    )
                    print(f"    loc {loc}: e_eq = {e_eqs[ii]:.2f} mV, {conc_eq_str}")
            return (
                np.array(e_eqs),
                {ion: np.array(conc_eq) for ion, conc_eq in conc_eqs.items()},
            )

    def _set_e_eq(self, ions=None, t_max=500.0, dt=0.1, factor_lambda=10.0):
        if ions is None:
            ions = self.ions

        locs = [(n.index, 0.5) for n in self]
        res = self._calc_e_eq(
            locs, ions=ions, t_max=t_max, dt=dt, factor_lambda=factor_lambda
        )

        for ii, n in enumerate(self):
            n.set_v_ep(res[0][ii])
            for ion, conc in res[1].items():
                n.set_conc_ep(ion, conc[ii])

    def set_e_eq(
        self, ions=None, t_max=500.0, dt=0.1, factor_lambda=10.0, pprint=False
    ):
        """
        Set equilibrium potentials and concentrations in the tree. Computes
        the equilibria through a NEURON simulation without inputs.

        Parameters
        ----------
        ions: `list` of `str
            the names of the ions for which the concentration needs to be measured
        t_max: float
            duration of the simulation
        dt: float
            time-step of the simulation
        factor_lambda: `float`
            multiplies the number of compartments suggested by the lambda-rule
        """
        self.maybe_execute_funcs(
            pprint=pprint,
            funcs_args_kwargs=[
                (
                    self._set_e_eq,
                    (),
                    dict(ions=ions, t_max=t_max, dt=dt, factor_lambda=factor_lambda),
                ),
            ],
        )


class CachedGreensTree(GreensTree, CachedTree):
    """
    Derived class of `neat.GreensTree` that caches the impedance calculation at each
    node.
    """

    def __init__(
        self,
        *args,
        recompute_cache=None,
        save_cache=None,
        cache_name=None,
        cache_path=None,
        **kwargs,
    ):
        # we want a behaviour where the cache parameters are initialized to certain defauls
        # if they are not provided, but where the initialization operation based on copying
        # the tree leaves the cache parameters intact if the input tree is a subclass
        # of CachedTree. However, we also want the optionally provided arguments to be
        # overwritten to their provided values. The constructor below achieve this
        self.set_cache_params(
            **self.get_cache_defaults(
                recompute_cache=recompute_cache,
                save_cache=save_cache,
                cache_name=cache_name,
                cache_path=cache_path,
            )
        )
        super().__init__(*args, **kwargs)
        self.set_cache_params(
            recompute_cache=recompute_cache,
            save_cache=save_cache,
            cache_name=cache_name,
            cache_path=cache_path,
        )

    def set_impedances_in_tree(self, freqs, sv_h=None, pprint=False, **kwargs):
        """
        Sets the impedances in the tree.

        Parameters
        ----------
        freqs: np.ndarray of float or complex
            The frequencies at which to evaluate the impedances
        sv_hs: dict of {string: np.ndarray}
            Keys are the channel names and values are numpy arrays that contain
            the expansion point for each ion channel
        pprint: bool (optional, default is ``False``)
            Print info
        """
        if pprint:
            cname_string = ", ".join(list(self.channel_storage.keys()))
            print(f">>> evaluating impedances with {cname_string}")

        # we set freqs here already because it needs to be included in the
        # representation to generate a hash
        self.freqs = np.array(freqs)

        if sv_h is not None:
            # check if exansion point for all channels is defined
            assert sv_h.keys() == self.channel_storage.keys()

            for c_name, sv in sv_h.items():

                # set the expansion point
                for node in self:
                    node.set_expansion_point(c_name, statevar=sv)

        self.maybe_execute_funcs(
            pprint=pprint,
            funcs_args_kwargs=[
                (self.set_comp_tree, [], {}),
                (self.set_impedance, [freqs], {"pprint": pprint, **kwargs}),
            ],
        )

    @computational_tree_decorator
    def calc_net_steadystate(self, root_loc=None, dx=5.0, dz=5.0):
        if root_loc is None:
            root_loc = (1, 0.5)
        root_loc = MorphLoc(root_loc, self)
        # distribute locs on nodes
        st_nodes = self.gather_nodes(self[root_loc["node"]])
        d2s_loc = self.path_length(root_loc, (1, 0.5))
        net_locs = self.distribute_locs_at_d2s(
            d2s=np.arange(d2s_loc, 5000.0, dx), node_arg=st_nodes, name="net eval"
        )
        # compute the impedance matrix for net calculation
        z_mat = self.calc_impedance_matrix("net eval", explicit_method=False)[0]
        # assert np.allclose(z_mat, z_mat_)
        # derive the NET
        net = NET()
        self._add_node_to_net(
            0.0,
            z_mat[0, 0],
            z_mat,
            np.arange(z_mat.shape[0]),
            None,
            net,
            alpha=1.0,
            dz=dz,
        )
        net.set_new_loc_idxs()

        return net, z_mat

    def _add_node_to_net(
        self, z_min, z_max, z_mat, loc_idxs, pnode, net, alpha=1.0, dz=20.0
    ):
        # compute mean impedance of node
        inds = [[]]
        while len(inds[0]) == 0:
            inds = np.where((z_mat > z_min) & (z_mat < z_max))
            z_max += dz
        z_node = np.mean(z_mat[inds])
        # subtract impedances of parent nodes
        gammas = np.array([z_node])
        self._subtract_parent_kernels(gammas, pnode)
        # add a node to the tree
        node = NETNode(len(net), loc_idxs, z_kernel=(np.array([alpha]), gammas))
        if pnode != None:
            net.add_node_with_parent(node, pnode)
        else:
            net.root = node
        # recursion for following nodes
        d_inds = consecutive(np.where(np.diag(z_mat) > z_max)[0])
        for di in d_inds:
            if len(di) > 0:
                self._add_node_to_net(
                    z_max,
                    z_max + dz,
                    z_mat[di, :][:, di],
                    loc_idxs[di],
                    node,
                    net,
                    alpha=alpha,
                    dz=dz,
                )

    def _subtract_parent_kernels(self, gammas, pnode):
        if pnode != None:
            gammas -= pnode.z_kernel["c"]
            self._subtract_parent_kernels(gammas, pnode.parent_node)


class CachedGreensTreeTime(GreensTreeTime, CachedTree):
    """
    Derived class of `neat.GreensTreeTime` that caches the separation of variables calculation
    """

    def __init__(
        self,
        *args,
        recompute_cache=None,
        save_cache=None,
        cache_name=None,
        cache_path=None,
        **kwargs,
    ):
        # we want a behaviour where the cache parameters are initialized to certain defauls
        # if they are not provided, but where the initialization operation based on copying
        # the tree leaves the cache parameters intact if the input tree is a subclass
        # of CachedTree. However, we also want the optionally provided arguments to be
        # overwritten to their provided values. The constructor below achieve this
        self.set_cache_params(
            **self.get_cache_defaults(
                recompute_cache=recompute_cache,
                save_cache=save_cache,
                cache_name=cache_name,
                cache_path=cache_path,
            )
        )
        super().__init__(*args, **kwargs)
        self.set_cache_params(
            recompute_cache=recompute_cache,
            save_cache=save_cache,
            cache_name=cache_name,
            cache_path=cache_path,
        )

    def set_impedances_in_tree(self, t_arr, pprint=False):
        """
        Sets the impedances in the tree that are necessary for the evaluation
        of the response kernels.

        Parameters
        ----------
        t_arr: np.ndarray of float
            The time-points at which to evaluate the response kernels
        pprint: bool (optional, default is ``False``)
            Print info
        """
        if pprint:
            cname_string = ", ".join(list(self.channel_storage.keys()))
            print(f">>> evaluating response kernels with {cname_string}")

        self._set_freq_and_time_arrays(t_arr)

        self.maybe_execute_funcs(
            pprint=pprint,
            funcs_args_kwargs=[
                (self.set_comp_tree, [], {}),
                (self.set_impedance, [t_arr], {}),
            ],
        )


class CachedSOVTree(SOVTree, CachedTree):
    """
    Derived class of `neat.GreensTreeTime` that caches the impedance calculation at each
    node.
    """

    def __init__(
        self,
        *args,
        recompute_cache=None,
        save_cache=None,
        cache_name=None,
        cache_path=None,
        **kwargs,
    ):
        # we want a behaviour where the cache parameters are initialized to certain defauls
        # if they are not provided, but where the initialization operation based on copying
        # the tree leaves the cache parameters intact if the input tree is a subclass
        # of CachedTree. However, we also want the optionally provided arguments to be
        # overwritten to their provided values. The constructor below achieve this
        self.set_cache_params(
            **self.get_cache_defaults(
                recompute_cache=recompute_cache,
                save_cache=save_cache,
                cache_name=cache_name,
                cache_path=cache_path,
            )
        )
        super().__init__(*args, **kwargs)
        self.set_cache_params(
            recompute_cache=recompute_cache,
            save_cache=save_cache,
            cache_name=cache_name,
            cache_path=cache_path,
        )

    def set_sov_in_tree(self, maxspace_freq=100.0, pprint=False):
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
        pprint: `bool`
            Verbose if ``True``.
        """
        if pprint:
            print(f">>> evaluating SOV expansion")

        self.maxspace_freq = maxspace_freq

        self.maybe_execute_funcs(
            pprint=pprint,
            funcs_args_kwargs=[
                (self.set_comp_tree, [], {"eps": 1.0}),
                (
                    self.calc_sov_equations,
                    [],
                    {
                        "maxspace_freq": maxspace_freq,
                        "pprint": pprint,
                    },
                ),
            ],
        )
