import numpy as np
import matplotlib.pyplot as pl

import neuron
from neuron import h

import os
import subprocess

import pytest


try:
    import nest
    import nest.lib.hl_api_exceptions as nestexceptions
    from neat import NestCompartmentTree
    WITH_NEST = True
except ImportError as e:
    WITH_NEST = False

from neat import PhysTree, GreensTree, NeuronSimTree, CompartmentFitter
from neat import NeuronCompartmentTree
import neat.channels.ionchannels as ionchannels
from neat.factorydefaults import DefaultPhysiology

from channelcollection_for_tests import *
import channel_installer
channel_installer.load_or_install_neuron_testchannels()
channel_installer.load_or_install_nest_testchannels()


CFG = DefaultPhysiology()
MORPHOLOGIES_PATH_PREFIX = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    'test_morphologies'
))


class TestConcMechs:
    def loadAxonTree(self, w_ca_conc=True, gamma_factor=1.):
        '''
        Parameters taken from a BBP SST model for a subset of ion channels
        '''
        tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball_and_axon.swc'),
            types=[1,2,3,4],
        )
        # capacitance and axial resistance
        tree.setPhysiology(1.0, 100./1e6)
        # ion channels
        k_chan = SKv3_1()
        tree.addCurrent(k_chan,  0.653374 * 1e6, -85., node_arg=[tree[1]])
        tree.addCurrent(k_chan,  0.196957 * 1e6, -85., node_arg="axonal")
        na_chan = NaTa_t()
        tree.addCurrent(na_chan, 3.418459 * 1e6, 50., node_arg="axonal")
        ca_chan = Ca_HVA()
        tree.addCurrent(ca_chan, 0.000792 * 1e6, 132.4579341637009, node_arg=[tree[1]])
        tree.addCurrent(ca_chan, 0.000138 * 1e6, 132.4579341637009, node_arg="axonal")
        sk_chan = SK_E2()
        tree.addCurrent(sk_chan, 0.653374 * 1e6, -85., node_arg=[tree[1]])
        tree.addCurrent(sk_chan, 0.196957 * 1e6, -85., node_arg="axonal")
        # passive leak current
        tree.setLeakCurrent(0.000091 * 1e6, -62.442793, node_arg=[tree[1]])
        tree.setLeakCurrent(0.000094 * 1e6, -79.315740, node_arg="axonal")

        if w_ca_conc:
            # ca concentration mech
            tree.addConcMech(
                "ca",
                params={
                    "tau": 605.033222,
                    # "tau": 20.715642,
                    "gamma": gamma_factor * 0.000893 * 1e4 / (2.0 * 0.1 * neuron.h.FARADAY) * 1e-6,
                    # "gamma": 0.,
                },
                node_arg=[tree[1]],
            )
            tree.addConcMech(
                "ca",
                params={
                    "tau": 20.715642,
                    # "tau": 605.033222,
                    "gamma": gamma_factor * 0.003923 * 1e4 / (2.0 * 0.1 * neuron.h.FARADAY) * 1e-6,
                },
                node_arg="axonal",
            )
        else:
            # These parameters effectively disable changes in the Ca-concentration
            # In fact, it would be superfluous to add this mechanisms here,
            # if not for the fact that the default Ca concentration in neuron
            # is different from the asymptotic value in the ca_conc.mod
            # mechanism.
            #
            # TODO: find out how to set the default Ca concentration in Neuron
            # without inserting the mechanism
            tree.addConcMech(
                "ca",
                params={
                    "tau": 1e20,
                    "gamma": 0.,
                },
                node_arg=[tree[1]],
            )
            tree.addConcMech(
                "ca",
                params={
                    "tau": 1e20,
                    "gamma": 0.,
                },
                node_arg="axonal",
            )

        # set computational tree
        tree.setCompTree()

        return tree

    def loadPassiveAxonTree(self, gamma_factor=1.):
        '''
        Parameters taken from a BBP SST model for a subset of ion channels
        '''
        tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball_and_axon.swc'),
            types=[1,2,3,4],
        )
        # capacitance and axial resistance
        tree.setPhysiology(1.0, 100./1e6)
        # ion channels
        k_chan = SKv3_1()
        tree.addCurrent(k_chan,  0.653374 * 1e6, -85., node_arg=[tree[1]])
        ca_chan = Ca_HVA()
        tree.addCurrent(ca_chan, 0.000792 * 1e6, 132.4579341637009, node_arg=[tree[1]])
        sk_chan = SK_E2()
        tree.addCurrent(sk_chan, 0.653374 * 1e6, -85., node_arg=[tree[1]])
        # passive leak current
        tree.setLeakCurrent(0.000091 * 1e6, -62.442793, node_arg=[tree[1]])
        tree.setLeakCurrent(0.000094 * 1e6, -79.315740, node_arg="axonal")

        # ca concentration mech
        tree.addConcMech(
            "ca",
            params={
                "tau": 605.033222,
                "gamma": gamma_factor * 0.000893 * 1e4 / (2.0 * 0.1 * neuron.h.FARADAY) * 1e-6,
            },
            node_arg=[tree[1]],
        )

        # set computational tree
        tree.setCompTree()

        return tree

    def loadNoCaAxonTree(self, gamma_factor=1.):
        '''
        Parameters taken from a BBP SST model for a subset of ion channels
        '''
        tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball_and_axon.swc'),
            types=[1,2,3,4],
        )
        # capacitance and axial resistance
        tree.setPhysiology(1.0, 100./1e6)
        # ion channels
        k_chan = SKv3_1()
        tree.addCurrent(k_chan,  0.653374 * 1e6, -85., node_arg=[tree[1]])
        tree.addCurrent(k_chan,  0.196957 * 1e6, -85., node_arg="axonal")
        na_chan = NaTa_t()
        tree.addCurrent(na_chan, 3.418459 * 1e6, 50., node_arg="axonal")
        ca_chan = Ca_HVA()
        tree.addCurrent(ca_chan, 0.000792 * 1e6, 132.4579341637009, node_arg=[tree[1]])
        tree.addCurrent(ca_chan, 0.000138 * 1e6, 132.4579341637009, node_arg="axonal")
        sk_chan = SK_E2()
        tree.addCurrent(sk_chan, 0.653374 * 1e6, -85., node_arg=[tree[1]])
        tree.addCurrent(sk_chan, 0.196957 * 1e6, -85., node_arg="axonal")
        # passive leak current
        tree.setLeakCurrent(0.000091 * 1e6, -62.442793, node_arg=[tree[1]])
        tree.setLeakCurrent(0.000094 * 1e6, -79.315740, node_arg="axonal")

        # ca concentration mech
        tree.addConcMech(
            "ca",
            params={
                "tau": 605.033222,
                "gamma": gamma_factor * 0.000893 * 1e4 / (2.0 * 0.1 * neuron.h.FARADAY) * 1e-6,
            },
            node_arg=[tree[1]],
        )

        # set computational tree
        tree.setCompTree()

        return tree

    def loadBall(self, w_ca_conc=True, gamma_factor=1.):
        '''
        Parameters taken from a BBP SST model for a subset of ion channels
        '''
        tree = PhysTree(
            os.path.join(MORPHOLOGIES_PATH_PREFIX, 'ball.swc'),
            types=[1,2,3,4],
        )
        # capacitance and axial resistance
        tree.setPhysiology(1.0, 100./1e6)
        # ion channels
        k_chan = SKv3_1()
        tree.addCurrent(k_chan,  0.653374 * 1e6, -85., node_arg=[tree[1]])
        na_chan = NaTa_t()
        tree.addCurrent(na_chan, 3.418459 * 1e6, 50., node_arg=[tree[1]])
        ca_chan = Ca_HVA()
        tree.addCurrent(ca_chan, 0.000792 * 1e6, 132.4579341637009, node_arg=[tree[1]])
        ca_chan_ = Ca_LVAst()
        tree.addCurrent(ca_chan_, 0.005574 * 1e6, 132.4579341637009, node_arg=[tree[1]])
        sk_chan = SK_E2()
        tree.addCurrent(sk_chan, 0.653374 * 1e6, -85., node_arg=[tree[1]])
        # passive leak current
        tree.setLeakCurrent(0.000091 * 1e6, -62.442793, node_arg=[tree[1]])

        if w_ca_conc:
            # ca concentration mech
            tree.addConcMech(
                "ca",
                params={
                    "tau": 605.033222,
                    "gamma": gamma_factor * 0.000893 * 1e4 / (2.0 * 0.1 * neuron.h.FARADAY)*1e-6, # 1/mA -> 1/nA
                },
                node_arg=[tree[1]],
            )
        else:
            # These parameters effectively disable changes in the Ca-concentration
            # In fact, it would be superfluous to add this mechanisms here,
            # if not for the fact that the default Ca concentration in neuron
            # is different from the asymptotic value in the ca_conc.mod
            # mechanism.
            #
            # TODO: find out how to set the default Ca concentration in Neuron
            # without inserting the mechanism
            tree.addConcMech(
                "ca",
                params={
                    "tau": 100.,
                    "gamma": 0.,
                },
                node_arg=[tree[1]],
            )

        # set computational tree
        tree.setCompTree()

        return tree

    def testStringRepresentation(self):
        tree = self.loadBall(w_ca_conc=True)

        repr_str = "['PhysTree', \"{'node index': 1, 'parent index': -1, 'content': '{}', 'xyz': array([0., 0., 0.]), 'R': '12', 'swc_type': 1, " \
            "'currents': {'SKv3_1': '(653374, -85)', 'NaTa_t': '(3.41846e+06, 50)', 'Ca_HVA': '(792, 132.458)', 'Ca_LVAst': '(5574, 132.458)', 'SK_E2': '(653374, -85)', 'L': '(91, -62.4428)'}, " \
            "'concmechs': {'ca': ExpConcMech(ion=ca, gamma=4.62765e-10, tau=605.033, inf=0.0001)}, 'c_m': '1', 'r_a': '0.0001', 'g_shunt': '0', 'v_ep': '-75', 'conc_eps': {}}\"]" \
            "{'channel_storage': ['Ca_HVA', 'Ca_LVAst', 'NaTa_t', 'SK_E2', 'SKv3_1']}"

        assert repr_str == repr(tree)

    def _simulate(self, simtree, rec_locs, amp=0.8, dur=100., delay=10., cal=100., rec_currs=["ca", "k"]):
        # initialize simulation tree
        simtree.init_model(t_calibrate=cal, factor_lambda=10.)
        simtree.storeLocs(rec_locs, name='rec locs')

        # initialize input
        simtree.addIClamp(rec_locs[0], amp, delay, dur)

        # run test simulation
        res = simtree.run(1.5*dur,
            record_from_channels=True,
            record_concentrations=["ca"],
            record_currents=rec_currs,
            spike_rec_loc=rec_locs[0],
        )

        return res

    def testSpiking(self, pplot=False):
        tree_w_ca = self.loadAxonTree(w_ca_conc=True)
        tree_no_ca = self.loadAxonTree(w_ca_conc=False)

        locs = [(1, .5), (4, .5), (4, 1.), (5, .5), (5, 1.)]
        res_w_ca = self._simulate(NeuronSimTree(tree_w_ca), locs)
        res_no_ca = self._simulate(NeuronSimTree(tree_no_ca), locs)

        assert len(res_no_ca["spikes"]) == 25
        assert len(res_w_ca["spikes"]) == 7

        if pplot:
            pl.figure()
            ax = pl.subplot(311)
            ax.plot(res_w_ca['t'], res_w_ca['v_m'][0], c='r', label="w ca conc")
            ax.plot(res_no_ca['t'], res_no_ca['v_m'][0], 'b--', label="w/o ca conc")
            ax.legend(loc=0)

            ax = pl.subplot(312)
            ax.plot(res_w_ca['t'], res_w_ca['ca'][0], c='r')
            ax.plot(res_no_ca['t'], res_no_ca['ca'][0], 'b--')

            ax = pl.subplot(313)
            ax.plot(res_w_ca['t'], res_w_ca['ik'][0], c='r')
            ax.plot(res_no_ca['t'], res_no_ca['ik'][0], 'b--')
            pl.show()

    def _compute_gca_cca_analytical(self, tree, freqs):
        ion = 'ca'
        node = tree[1]
        g_m_ca = np.zeros_like(freqs)

        # loop over  all active channels
        for channel_name in set(node.currents.keys()) - set('L'):
            g, e = node.currents[channel_name]

            # recover the ionchannel object
            channel = tree.channel_storage[channel_name]

            if g < 1e-10 or channel.ion != ion:
                continue

            # check if linearistation needs to be computed around expansion point
            sv = node.getExpansionPoint(channel_name).copy()

            # if voltage is not in expansion point, use equilibrium potential
            v = sv.pop('v', node.v_ep)

            # if concencentration is in expansion point, use it. Otherwise use
            # concentration in equilibrium concentrations (self.conc_eps), if
            # it is there. If not, use default concentration.
            ions = [str(ion) for ion in channel.conc] # convert potential sympy symbols to str
            conc = {
                ion: sv.pop(
                        ion, node.conc_eps.copy().pop(ion, CFG.conc[ion])
                    ) \
                for ion in ions
            }
            sv.update(conc)

            # compute linearized channel contribution to membrane impedance
            g_m_ca = g_m_ca - g * channel.computeLinSum(v, freqs, e=e, **sv)

        c_ca = node.concmechs[ion].computeLinear(freqs) * g_m_ca

        return g_m_ca, c_ca

    def _compute_gk_analytical(self, tree, freqs):
        ion = 'k'
        node = tree[1]
        g_m_k = np.zeros_like(freqs)

        _, c_ca = self._compute_gca_cca_analytical(tree, freqs)

        # loop over  all active channels
        for channel_name in set(node.currents.keys()) - set('L'):
            g, e = node.currents[channel_name]

            # recover the ionchannel object
            channel = tree.channel_storage[channel_name]

            if g < 1e-10 or channel.ion != ion:
                continue

            # check if linearistation needs to be computed around expansion point
            sv = node.getExpansionPoint(channel_name).copy()

            # if voltage is not in expansion point, use equilibrium potential
            v = sv.pop('v', node.v_ep)

            # if concencentration is in expansion point, use it. Otherwise use
            # concentration in equilibrium concentrations (self.conc_eps), if
            # it is there. If not, use default concentration.
            ions = [str(ion_) for ion_ in channel.conc] # convert potential sympy symbols to str
            conc = {
                ion_: sv.pop(
                        ion_, node.conc_eps.copy().pop(ion_, CFG.conc[ion_])
                    ) \
                for ion_ in ions
            }
            sv.update(conc)

            # compute linearized channel contribution to membrane impedance
            g_m_k = g_m_k - g * channel.computeLinSum(v, freqs, e=e, **sv)

            if 'ca' in channel.conc:
                # add concentration contribution to linearized membrane
                # conductance
                g_m_k = g_m_k - \
                    g * \
                    channel.computeLinConc(v, freqs, 'ca', e=e, **sv) * \
                    c_ca #* 1e-6

        return g_m_k

    def testImpedance(self, pplot=False, amp=0.001):

        tree0 = GreensTree(self.loadBall(w_ca_conc=False))
        tree1 = GreensTree(self.loadBall(w_ca_conc=True, gamma_factor=1e2))
        tree2 = GreensTree(self.loadBall(w_ca_conc=True, gamma_factor=1e2))

        locs = [(1, .5)]
        res0 = self._simulate(NeuronSimTree(tree0),
            locs, amp=amp, dur=20000., delay=100., cal=10000.
        )
        res2 = self._simulate(NeuronSimTree(tree2),
            locs, amp=amp, dur=20000., delay=100., cal=10000.
        )

        sim_p_open2 = {
            "SKv3_1":   res2['chan']['SKv3_1']['z'][0][0],
            "Ca_HVA":   res2['chan']['Ca_HVA']['m'][0][0]**2 * res2['chan']['Ca_HVA']['h'][0][0],
            "Ca_LVAst": res2['chan']['Ca_LVAst']['m'][0][0]**2 * res2['chan']['Ca_LVAst']['h'][0][0],
            "SK_E2":    res2['chan']['SK_E2']['z'][0][0],
            "NaTa_t":   res2['chan']['NaTa_t']['m'][0][0]**3 * res2['chan']['NaTa_t']['h'][0][0],
        }

        # set equilbria in trees
        eq0 = {'v': res0['v_m'][0][0], 'ca': res0['ca'][0][0]}
        eq2 = {'v': res2['v_m'][0][0], 'ca': res2['ca'][0][0]}

        tree0.setVEP(eq0['v'], node_arg=[tree0[1]])
        tree1.setVEP(eq0['v'], node_arg=[tree1[1]]) # use eq0 -- without conc
        tree2.setVEP(eq2['v'], node_arg=[tree2[1]])

        tree0.setConcEP('ca', eq0['ca'], node_arg=[tree0[1]])
        tree1.setConcEP('ca', eq0['ca'], node_arg=[tree1[1]]) # use eq0 -- without conc
        tree2.setConcEP('ca', eq2['ca'], node_arg=[tree2[1]])

        # test whether computed and simulated open probabilities are the same
        calc_p_open2 = {
            cname: chan.computePOpen(eq2['v'], ca=eq2['ca']) \
            for cname, chan in tree2.channel_storage.items()
        }
        for cname, p_o_sim in sim_p_open2.items():
            p_o_calc = calc_p_open2[cname]
            assert np.abs(p_o_calc - p_o_sim) < 1e-9 * (p_o_sim + p_o_calc) / 2.

        # impedance calculation
        tree0.setCompTree()
        tree1.setCompTree()
        tree2.setCompTree()

        tree0.setImpedance(0.)
        tree1.setImpedance(0., use_conc=False) # omit concentration mechanism
        tree2.setImpedance(0., use_conc=True)

        z_in0 = tree0.calcZF((1,.5), (1,.5))
        z_in1 = tree1.calcZF((1,.5), (1,.5))
        z_in2 = tree2.calcZF((1,.5), (1,.5))

        # test whether omitting the concentration mechanisms from the impedance
        # calculation results in the same impedance as without the concentration
        # mechanism
        assert np.abs(z_in0 - z_in1) < 1e-15 * z_in0
        # test whether including the impedance mechanisms has an effect
        assert np.abs(z_in0 - z_in2) > .1 * z_in0

        # compute analytical conductance terms for combined ca and k currents,
        # and the analytical ca concentration
        d_gca0, d_cca0 = self._compute_gca_cca_analytical(tree0, 0.)
        d_gk0 = self._compute_gk_analytical(tree0, 0.) #np.array([0.]) * 1j)
        d_gca2, d_cca2 = self._compute_gca_cca_analytical(tree2, 0.)
        d_gk2 = self._compute_gk_analytical(tree2, 0.) #np.array([0.]) * 1j)

        # measured voltage deviations
        dv0 = res0['v_m'][0][int((100. + 20000.)/0.1)-10] - res0['v_m'][0][0]
        dv2 = res2['v_m'][0][int((100. + 20000.)/0.1)-10] - res2['v_m'][0][0]

        # measure current deviations
        dik0 = res0['ik'][0][int((100. + 20000.)/0.1)-10] - res0['ik'][0][0]
        dik2 = res2['ik'][0][int((100. + 20000.)/0.1)-10] - res2['ik'][0][0]
        dica0 = res0['ica'][0][int((100. + 20000.)/0.1)-10] - res0['ica'][0][0]
        dica2 = res2['ica'][0][int((100. + 20000.)/0.1)-10] - res2['ica'][0][0]

        # check whether the conductance terms are correct
        assert np.abs(dik0 - dv0 * d_gk0 * 1e-6) < 0.01 * np.abs(dik0)
        assert np.abs(dik2 - dv2 * d_gk2 * 1e-6) < 0.01 * np.abs(dik0)
        assert np.abs(dica0 - dv0 * d_gca0 * 1e-6) < 0.01 * np.abs(dica0)
        assert np.abs(dica2 - dv2 * d_gca2 * 1e-6) < 0.01 * np.abs(dica0)

        if pplot:
            pl.figure()
            ax = pl.subplot(411)
            ax.plot(res0['t'], res0['v_m'][0], 'r', label="no ca")
            ax.plot(res2['t'], res2['v_m'][0], 'g', label="w ca2")

            ax.axhline(res0['v_m'][0][0] + amp * z_in0, c="y", ls='-.')
            ax.axhline(res2['v_m'][0][0] + amp * z_in2, c="DarkGrey", ls='-.')

            ax.legend(loc=0)

            ax = pl.subplot(412)
            ax.plot(res2['t'], res2['ca'][0], 'g', label="w ca2")
            ax.axhline(res2['ca'][0][0] + dv2 * d_cca2 * 1e-6 * 1e6, c="DarkGrey", ls='-.')

            ax = pl.subplot(413)
            ax.plot(res0['t'], res0['ik'][0], 'r', label="no ca")
            ax.plot(res2['t'], res2['ik'][0], 'g', label="w ca2")

            ax.axhline(res0['ik'][0][0] + dv0 * d_gk0 * 1e-6, c="y", ls='-.')
            ax.axhline(res2['ik'][0][0] + dv2 * d_gk2 * 1e-6, c="DarkGrey", ls='-.')
            ax = pl.subplot(414)
            ax.plot(res0['t'], res0['ica'][0], 'r', label="no ca")
            ax.plot(res2['t'], res2['ica'][0], 'g', label="w ca2")
            ax.axhline(res0['ica'][0][0] + dv0 * d_gca0 * 1e-6, c="y", ls='-.')
            ax.axhline(res2['ica'][0][0] + dv2 * d_gca2 * 1e-6, c="DarkGrey", ls='-.')

            i_reconstr = (
                0.000792 * res2['chan']['Ca_HVA']['m'][0]**2 * res2['chan']['Ca_HVA']['h'][0] + \
                0.005574 * res2['chan']['Ca_LVAst']['m'][0]**2 * res2['chan']['Ca_LVAst']['h'][0]
            ) * (
                res2['v_m'][0] - 132.4579341637009
            )
            ax.plot(res2['t'], i_reconstr, ls='--', c="purple")

            pl.show()

    def testFittingBall(self, pplot=False, fit_tau=False, amp=0.1, eps_gamma=1e-6, eps_tau=1e-10):
        locs = [(1,.5)]

        tree = self.loadBall(w_ca_conc=True, gamma_factor=1e3)

        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        cfit.setCTree(locs)

        # fit the passive steady state model
        cfit.fitPassive(pprint=True, use_all_channels=False)

        # fit the capacitances
        cfit.fitCapacitance(pprint=True, pplot=False)

        # fit the ion channel
        cfit.fitChannels(pprint=True)

        # fit the concentration mechanism
        cfit.fitConcentration('ca', fit_tau=fit_tau, pprint=True)

        # fit the resting potentials
        cfit.fitEEq(ions=['ca'], t_max=10000)

        ctree = cfit.ctree
        clocs = ctree.getEquivalentLocs()

        # check whether parameters of original and fitted models match
        node = tree[1]
        cnode = ctree[0]
        A = 4. * np.pi * (node.R * 1e-4)**2

        # check fitting
        for channel_name in node.currents:
            # check conductances
            assert np.abs(
                cnode.currents[channel_name][0] - node.currents[channel_name][0] * A
            ) < 1e-8 * np.abs(node.currents[channel_name][0])
            # check reversals
            assert np.abs(
                cnode.currents[channel_name][1] - node.currents[channel_name][1]
            ) < 1e-8 * np.abs(node.currents[channel_name][1])

        for ion in node.concmechs:
            # check gamma factors
            assert np.abs(
                cnode.concmechs[ion].gamma * A - node.concmechs[ion].gamma
            ) < np.abs(node.concmechs[ion].gamma) * eps_gamma
            # check time scales
            assert np.abs(
                cnode.concmechs[ion].tau - node.concmechs[ion].tau
            ) < np.abs(node.concmechs[ion].tau) * eps_tau

        # run test simulations
        res_full = self._simulate(NeuronSimTree(tree),
            locs, amp=amp, dur=1000., delay=100., cal=1000.
        )
        res_reduced = self._simulate(NeuronCompartmentTree(ctree),
            clocs, amp=amp, dur=1000., delay=100., cal=1000.
        )

        # check whether the simulation results match
        assert np.allclose(res_full['v_m'], res_reduced['v_m'])

        if pplot:
            pl.figure()
            ax = pl.subplot(121)

            ax.plot(res_full['t'], res_full['v_m'][0], 'b')
            ax.plot(res_reduced['t'], res_reduced['v_m'][0], 'r--')

            ax = pl.subplot(122)

            ax.plot(res_full['t'], res_full['ca'][0], 'b')
            ax.plot(res_reduced['t'], res_reduced['ca'][0], 'r--')

            pl.show()

    def testTauFitBall(self, pplot=False):
        self.testFittingBall(fit_tau=True, pplot=pplot, eps_gamma=1e-3, eps_tau=1e-1)

    def testFittingBallAndStick(self, pplot=False, amp=0.1):
        locs = [(1,.5), (4.,0.5), (5,0.5)]

        tree = self.loadAxonTree(w_ca_conc=True, gamma_factor=1e3)
        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)

        # test explicit fit
        cfit.setCTree(locs)

        # fit the passive steady state model
        cfit.fitPassive(pprint=False, use_all_channels=False)

        # fit the capacitances
        cfit.fitCapacitance(pprint=False, pplot=False)

        # fit the ion channel
        cfit.fitChannels(pprint=False)

        # fit the concentration mechanism
        cfit.fitConcentration('ca', pprint=False)

        # fit the resting potentials
        cfit.fitEEq(ions=['ca'], t_max=10000)

        ctree = cfit.ctree
        clocs = ctree.getEquivalentLocs()

        # test fit with fitModel function
        ctree_ = cfit.fitModel(locs, use_all_channels_for_passive=False)

        # check whether both reductions are the same
        for ii in range(len(locs)):
            cnode = ctree[ii]
            cnode_ = ctree_[ii]

            # check channels
            for channel_name in cnode.currents:
                # check conductances
                assert np.abs(
                    cnode.currents[channel_name][0] - cnode_.currents[channel_name][0]
                ) < 1e-8 * np.abs(cnode_.currents[channel_name][0])
                # check reversals
                assert np.abs(
                    cnode.currents[channel_name][1] - cnode_.currents[channel_name][1]
                ) < 1e-8 * np.abs(cnode_.currents[channel_name][1])

            for ion in cnode.concmechs:
                # check gamma factors
                assert np.abs(
                    cnode.concmechs[ion].gamma - cnode_.concmechs[ion].gamma
                ) < np.abs(cnode_.concmechs[ion].gamma) * 1e-8
                # check time scales
                assert np.abs(
                    cnode.concmechs[ion].tau - cnode_.concmechs[ion].tau
                ) < np.abs(cnode_.concmechs[ion].tau) * 1e-8

        # run test simulations
        res_full = self._simulate(NeuronSimTree(tree),
            locs, amp=amp, dur=1000., delay=100., cal=1000.
        )
        res_reduced = self._simulate(NeuronCompartmentTree(ctree),
            clocs, amp=amp, dur=1000., delay=100., cal=1000.
        )

        v_error = np.sqrt(np.mean((res_full['v_m'] - res_reduced['v_m'])**2))
        assert v_error < 1e-2 # mV

        if pplot:
            pl.figure()
            ax = pl.gca()

            ax.plot(res_full['t'], res_full['v_m'][2], 'b')
            ax.plot(res_reduced['t'], res_reduced['v_m'][2], 'r--')

            pl.show()

    def testFiniteDifference(self, rtol_param=5e-2, pprint=False):
        tree = self.loadAxonTree(w_ca_conc=True, gamma_factor=1e3)
        # finite difference ctree
        ctree_fd, locs_fd = tree.createFiniteDifferenceTree(dx_max=22.)
        # fitted ctree
        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        ctree_fit = cfit.fitModel(locs_fd)

        # check whether both trees have the same parameters
        for node_fd, node_fit in zip(ctree_fd, ctree_fit):

            if pprint: print("---")
            # test capacitance match
            assert np.abs(node_fd.ca - node_fit.ca) < \
                                rtol_param * np.max([node_fd.ca, node_fit.ca])
            if pprint: print(f"ca_fd = {node_fd.ca}, ca_fit = {node_fit.ca}")

            # test coupling cond match
            if not ctree_fd.is_root(node_fd):
                if pprint: print(f"gc_fd = {node_fd.g_c}, gc_fit = {node_fit.g_c}")
                assert np.abs(node_fd.g_c - node_fit.g_c) < \
                                    rtol_param * np.max([node_fd.g_c, node_fit.g_c])

            # test leak current match
            for key in node_fd.currents:
                g_fd = node_fd.currents[key][0]
                g_fit = node_fit.currents[key][0]
                if pprint: print(f"g{key}_fd = {g_fd}, g{key}_fit = {g_fit}")
                assert np.abs(g_fd - g_fit) < \
                                rtol_param * np.max([g_fd, g_fit])


            for ion in node_fd.concmechs:
                gamma_fd = node_fd.concmechs[ion].gamma
                gamma_fit = node_fit.concmechs[ion].gamma
                if pprint: print(f"gamma_{ion}_fd = {gamma_fd}, gamma_{ion}_fit = {gamma_fit}")
                assert np.abs(gamma_fd - gamma_fit) < \
                                rtol_param * np.max([gamma_fd, gamma_fit])

    def _runLocalizedConcMech(self, tree):
        locs = [(1,.5), (4.,0.5), (5,0.5)]

        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        cfit.setCTree(locs)

        # fit the passive steady state model
        cfit.fitPassive(pprint=True, use_all_channels=False)

        # fit the capacitances
        cfit.fitCapacitance(pprint=True, pplot=False)

        # fit the ion channel
        cfit.fitChannels(pprint=True)

        # fit the concentration mechanism
        cfit.fitConcentration('ca', pprint=True)

        ctree = cfit.ctree

        # check whether parameters of original and fitted models match
        node = tree[1]
        cnode = ctree[0]
        A = 4. * np.pi * (node.R * 1e-4)**2

        assert np.abs(
            cnode.concmechs['ca'].gamma * A - node.concmechs['ca'].gamma
        ) < 1e-6 * np.abs(node.concmechs['ca'].gamma)
        assert np.abs(
            ctree[1].concmechs['ca'].gamma
        ) < np.abs(ctree[0].concmechs['ca'].gamma) * 1e-10
        assert np.abs(
            ctree[2].concmechs['ca'].gamma
        ) < np.abs(ctree[0].concmechs['ca'].gamma) * 1e-10

    def testLocalizedConcMechPasAxon(self):
        tree = self.loadPassiveAxonTree(gamma_factor=1e3)
        self._runLocalizedConcMech(tree)

    @pytest.mark.skip(reason="Fitting methodology fails in this case, should check if dynamics diverges significantly")
    def testLocalizedConcMechActAxon(self):
        tree = self.loadNoCaAxonTree(gamma_factor=1e3)
        self._runLocalizedConcMech(tree)

    def _simulateNest(self, simtree, loc_idxs, amp=0.8, dur=100., delay=10., cal=100.):
        dt = .025
        idx0 = int(cal / dt)
        nest.ResetKernel()
        nest.SetKernelStatus(dict(resolution=dt))

        # create the model
        nestmodel = simtree.init_model("multichannel_test", 1)

        # step current input
        dcg = nest.Create("step_current_generator",
            {
                "amplitude_times": [cal+dt, cal+delay, cal+delay+dur],
                "amplitude_values": [0., amp, 0.],
            }
        )
        nest.Connect(dcg, nestmodel,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": 1.0,
                "delay": 0.1,
                "receptor_type": loc_idxs[0],
            }
        )

        # voltage recording
        mm = nest.Create('multimeter', 1,
            {
                'record_from': [f"v_comp{idx}" for idx in loc_idxs] + [f"c_ca{idx}" for idx in loc_idxs],
                'interval': dt
            }
        )
        nest.Connect(mm, nestmodel)

        # simulate
        nest.Simulate(cal + 1.5 * dur)

        res_nest = nest.GetStatus(mm, 'events')[0]
        for key, arr in res_nest.items():
            res_nest[key] = arr[idx0:]
        res_nest['times'] -= cal

        return res_nest

    @pytest.mark.skipif(WITH_NEST, reason="NEST not installed")
    def testNestNeuronSimBall(self, pplot=False, fit_tau=False, amp=0.1, eps_gamma=1e-6, eps_tau=1e-10):
        locs = [(1,.5)]

        tree = self.loadBall(w_ca_conc=True, gamma_factor=1e3)

        cfit = CompartmentFitter(tree, save_cache=False, recompute_cache=True)
        ctree = cfit.fitModel(locs)

        clocs = ctree.getEquivalentLocs()
        cidxs = [n.index for n in ctree]

        res_neuron = self._simulate(NeuronCompartmentTree(ctree),
            clocs, amp=amp, dur=20000., delay=1000., cal=10000.,
            rec_currs=['ca'],
        )

        res_nest = self._simulateNest(NestCompartmentTree(ctree),
            cidxs, amp=amp, dur=20000., delay=1000., cal=10000.
        )

        assert np.allclose(
            res_neuron['v_m'][0][0:-5],
            res_nest['v_comp0'],
            atol=0.3,
        )
        assert np.allclose(
            res_neuron['ca'][0][0:-5],
            res_nest['c_ca0'],
            atol=1e-8,
        )

        if pplot:
            pl.figure("nest--neuron")
            ax = pl.subplot(211)

            ax.plot(res_neuron['t'], res_neuron['v_m'][0], "r-", lw=1)
            ax.plot(res_nest['times'], res_nest['v_comp0'], "b--", lw=1.3)

            ax = pl.subplot(212)

            ax.plot(res_neuron['t'], res_neuron['ca'][0], "r-", lw=1)
            ax.plot(res_nest['times'], res_nest['c_ca0'], "b--", lw=1.3)

            pl.show()


if __name__ == "__main__":
    tcm = TestConcMechs()
    # tcm.testStringRepresentation()
    # tcm.testSpiking(pplot=True)
    # tcm.testImpedance(pplot=True)
    # tcm.testFittingBall(pplot=True)
    # tcm.testTauFitBall(pplot=True)
    tcm.testFittingBallAndStick(pplot=True)
    # tcm.testFiniteDifference()
    # tcm.testLocalizedConcMechPasAxon()
    # tcm.testLocalizedConcMechActAxon()
    # tcm.testNestNeuronSimBall(pplot=True, amp=2.0)


