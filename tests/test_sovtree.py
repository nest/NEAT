# -*- coding: utf-8 -*-
#
# test_sovtree.py
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
import matplotlib.pyplot as pl
import os

import pytest
from neat import SOVTree, SOVNode, Kernel, GreensTree
import neat.tools.kernelextraction as ke

MORPHOLOGIES_PATH_PREFIX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_morphologies")
)


class TestSOVTree:
    def load_T_tree(self):
        """
        Load the T-tree morphology in memory

          6--5--4--7--8
                |
                |
                1
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "Tsovtree.swc")
        self.tree = SOVTree(fname, types=[1, 3, 4])
        self.tree.fit_leak_current(-75.0, 10.0)
        self.tree.set_comp_tree()

    def test_string_representation(self):
        self.load_T_tree()
        self.tree.calc_sov_equations()

        assert (
            str(self.tree) == f">>> SOVTree\n"
            "    SomaSOVNode 1, Parent: None\n"
            "    SOVNode 4, Parent: 1\n"
            "    SOVNode 5, Parent: 4\n"
            "    SOVNode 6, Parent: 5\n"
            "    SOVNode 7, Parent: 4\n"
            "    SOVNode 8, Parent: 7"
        )

        repr_str = (
            "['SOVTree', "
            "\"{'node index': 1, 'parent index': -1, 'content': '{}', 'xyz': array([0., 0., 0.]), 'R': '10', 'swc_type': 1, 'currents': {'L': '(100, -75)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 4, 'parent index': 1, 'content': '{}', 'xyz': array([100.,   0.,   0.]), 'R': '1', 'swc_type': 4, 'currents': {'L': '(100, -75)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 5, 'parent index': 4, 'content': '{}', 'xyz': array([100. ,  50.5,   0. ]), 'R': '1', 'swc_type': 4, 'currents': {'L': '(100, -75)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 6, 'parent index': 5, 'content': '{}', 'xyz': array([100., 101.,   0.]), 'R': '0.5', 'swc_type': 4, 'currents': {'L': '(100, -75)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 7, 'parent index': 4, 'content': '{}', 'xyz': array([100. , -49.5,   0. ]), 'R': '1', 'swc_type': 4, 'currents': {'L': '(100, -75)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\", "
            "\"{'node index': 8, 'parent index': 7, 'content': '{}', 'xyz': array([100., -99.,   0.]), 'R': '0.5', 'swc_type': 4, 'currents': {'L': '(100, -75)'}, 'concmechs': {}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\""
            "]{'channel_storage': [], 'maxspace_freq': '500'}"
        )
        assert repr(self.tree) == repr_str

    def load_validation_tree(self):
        """
        Load the T-tree morphology in memory

        5---1---4
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "sovvalidationtree.swc")
        self.tree = SOVTree(fname, types=[1, 3, 4])
        self.tree.fit_leak_current(-75.0, 10.0)
        self.tree.set_comp_tree()

    def test_sov_calculation(self):
        # validate the calculation on analytical model
        self.load_validation_tree()
        # do SOV calculation
        self.tree.calc_sov_equations()
        alphas, gammas = self.tree.get_sov_matrices([(1, 0.5)])
        # compute time scales analytically
        with self.tree.as_computational_tree:
            lambda_m_test = np.sqrt(
                self.tree[4].R_sov / (2.0 * self.tree[4].g_m * self.tree[4].r_a)
            )
            tau_m_test = self.tree[4].c_m / self.tree[4].g_m * 1e3
            alphas_test = (
                1.0
                + (
                    np.pi
                    * np.arange(20)
                    * lambda_m_test
                    / (self.tree[4].L_sov + self.tree[5].L_sov)
                )
                ** 2
            ) / tau_m_test
            # compare analytical and computed time scales
            assert np.allclose(alphas[:20], alphas_test)
            # compute the spatial mode functions analytically
            ## TODO

        # test basic identities
        self.load_T_tree()
        self.tree.calc_sov_equations(maxspace_freq=500)
        # sets of location
        locs_0 = [(6, 0.5), (8, 0.5)]
        locs_1 = [(1, 0.5), (4, 0.5), (4, 1.0), (5, 0.5), (6, 0.5), (7, 0.5), (8, 0.5)]
        locs_2 = [(7, 0.5), (8, 0.5)]
        self.tree.store_locs(locs_0, "0")
        self.tree.store_locs(locs_1, "1")
        self.tree.store_locs(locs_2, "2")
        # test mode importance
        imp_a = self.tree.get_mode_importance(loc_arg=locs_0)
        imp_b = self.tree.get_mode_importance(loc_arg="0")
        imp_c = self.tree.get_mode_importance(
            sov_data=self.tree.get_sov_matrices(loc_arg=locs_0)
        )
        imp_d = self.tree.get_mode_importance(
            sov_data=self.tree.get_sov_matrices(loc_arg="0")
        )
        assert np.allclose(imp_a, imp_b)
        assert np.allclose(imp_a, imp_c)
        assert np.allclose(imp_a, imp_d)
        assert np.abs(1.0 - np.max(imp_a)) < 1e-12
        with pytest.raises(IOError):
            self.tree.get_mode_importance()
        # test important modes
        imp_2 = self.tree.get_mode_importance(loc_arg="2")
        assert not np.allclose(imp_a, imp_2)
        # test impedance matrix
        z_mat_a = self.tree.calc_impedance_matrix(
            sov_data=self.tree.get_important_modes(loc_arg="1", eps=1e-10)
        )
        z_mat_b = self.tree.calc_impedance_matrix(loc_arg="1", eps=1e-10)
        assert np.allclose(z_mat_a, z_mat_b)
        assert np.allclose(z_mat_a - z_mat_a.T, np.zeros(z_mat_a.shape))
        for ii, z_row in enumerate(z_mat_a):
            assert np.argmax(z_row) == ii
        # test Fourrier impedance matrix
        ft = ke.FourierQuadrature(np.arange(0.0, 100.0, 0.1))
        z_mat_ft = self.tree.calc_impedance_matrix(loc_arg="1", eps=1e-10, freqs=ft.s)
        assert np.allclose(
            z_mat_ft[ft.ind_0s, :, :].real, z_mat_a, atol=1e-1
        )  # check steady state
        assert np.allclose(
            z_mat_ft - np.transpose(z_mat_ft, axes=(0, 2, 1)), np.zeros(z_mat_ft.shape)
        )  # check symmetry
        assert np.allclose(
            z_mat_ft[: ft.ind_0s, :, :].real,
            z_mat_ft[ft.ind_0s + 1 :, :, :][::-1, :, :].real,
        )  # check real part even
        assert np.allclose(
            z_mat_ft[: ft.ind_0s, :, :].imag,
            -z_mat_ft[ft.ind_0s + 1 :, :, :][::-1, :, :].imag,
        )  # check imaginary part odd

    def load_ball(self):
        """
        Load point neuron model
        """
        fname = os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc")
        self.btree = SOVTree(fname, types=[1, 3, 4])
        self.btree.fit_leak_current(-75.0, 10.0)
        self.btree.set_comp_tree()

    def test_single_compartment(self):
        self.load_ball()
        # for validation
        greenstree = GreensTree(self.btree)
        greenstree.set_comp_tree()
        greenstree.set_impedance(np.array([0.0]))
        z_inp = greenstree.calc_impedance_matrix([(1.0, 0.5)])

        self.btree.calc_sov_equations(maxspace_freq=500)
        alphas, gammas = self.btree.get_sov_matrices(loc_arg=[(1.0, 0.5)])
        z_inp_sov = self.btree.calc_impedance_matrix(loc_arg=[(1.0, 0.5)])

        assert alphas.shape[0] == 1
        assert gammas.shape == (1, 1)
        assert np.abs(1.0 / np.abs(alphas[0]) - 10.0) < 1e-10

        g_m = self.btree[1].calc_g_tot(self.btree.channel_storage)
        g_s = g_m * 4.0 * np.pi * (self.btree[1].R * 1e-4) ** 2

        assert np.abs(gammas[0, 0] ** 2 / np.abs(alphas[0]) - 1.0 / g_s) < 1e-10
        assert np.abs(z_inp_sov - 1.0 / g_s) < 1e-10

    def test_kernel_calculation(self):
        self.load_T_tree()
        self.tree.calc_sov_equations(maxspace_freq=500)

        freqs = np.linspace(-10.0, 10.0, 11) * 1j
        times = np.linspace(0.1, 10.0, 10)

        locs = [(4, 0.1), (8, 0.4)]

        zf_trans = self.tree.calc_zf(locs[0], locs[1], freqs=freqs)
        zf_in1 = self.tree.calc_zf(locs[1], locs[1], freqs=freqs)
        zf_in0 = self.tree.calc_zf(locs[0], locs[0], freqs=freqs)
        zf_mat = self.tree.calc_impedance_matrix(loc_arg=locs, freqs=freqs)

        assert np.allclose(zf_mat[:, 0, 0], zf_in0)
        assert np.allclose(zf_mat[:, 1, 1], zf_in1)
        assert np.allclose(zf_mat[:, 0, 1], zf_trans)
        assert np.allclose(zf_mat[:, 1, 0], zf_trans)

        zt_trans = self.tree.calc_zt(locs[0], locs[1], times=times)
        zt_in1 = self.tree.calc_zt(locs[1], locs[1], times=times)
        zt_in0 = self.tree.calc_zt(locs[0], locs[0], times=times)
        zt_mat = self.tree.calc_impulse_response_matrix(loc_arg=locs, times=times)

        assert np.allclose(zt_mat[:, 0, 0], zt_in0)
        assert np.allclose(zt_mat[:, 1, 1], zt_in1)
        assert np.allclose(zt_mat[:, 0, 1], zt_trans)
        assert np.allclose(zt_mat[:, 1, 0], zt_trans)

    def test_net_derivation(self):
        # initialize
        self.load_validation_tree()
        self.tree.calc_sov_equations()
        # construct the NET
        net = self.tree.construct_net()
        # initialize
        self.load_T_tree()
        self.tree.calc_sov_equations()
        # construct the NET
        net = self.tree.construct_net(dz=20.0)
        # contruct the NET with linear terms
        net, lin_terms = self.tree.construct_net(dz=20.0, add_lin_terms=True)
        # check if correct
        alphas, gammas = self.tree.get_important_modes(
            loc_arg="net eval", eps=1e-4, sort_type="timescale"
        )
        for ii, lin_term in lin_terms.items():
            z_k_trans = net.get_reduced_tree([0, ii]).get_root().z_kernel + lin_term
            assert (
                np.abs(
                    z_k_trans.k_bar
                    - Kernel((alphas, gammas[:, 0] * gammas[:, ii])).k_bar
                )
                < 1e-8
            )


if __name__ == "__main__":
    tsov = TestSOVTree()
    tsov.test_string_representation()
    tsov.test_sov_calculation()
    tsov.test_single_compartment()
    tsov.test_net_derivation()
