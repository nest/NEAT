# -*- coding: utf-8 -*-
#
# simtree.py
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


class SimTreeMixin:
    """
    Simulator-independent measurements on simulation trees, implemented on top
    of the simulation interface that the simulation trees share
    (:class:`NeuronSimTree`, :class:`ArborSimTree`): `init_model()`,
    `delete_model()`, `set_simulation_parameters()`, `add_i_clamp()` and
    `run()`.
    """

    def calc_e_eq(self, t_dur=100.0, set_v_ep=True):
        """
        Compute the equilibrium potentials in the middle (``x=0.5``) of each node.

        Parameters
        ----------
        t_dur: float (optional, default ``100.`` ms)
            The duration of the simulation
        set_v_ep: bool (optional, default ``True``)
            Store the equilibrium potential as the ``PhysNode.v_ep`` attribute
        """
        self.init_model(
            dt=self.dt,
            t_calibrate=self.t_calibrate,
            v_init=self.v_init,
            factor_lambda=self.factor_lambda,
        )
        self.store_locs([(n.index, 0.5) for n in self], name="rec locs")
        res = self.run(t_dur)
        v_eq = res["v_m"][:, -1]
        if set_v_ep:
            for node, e in zip(self, v_eq):
                node.set_v_ep(e)

        return v_eq

    def calc_impedance_matrix(
        self,
        loc_arg,
        i_amp=0.001,
        t_dur=100.0,
        pplot=False,
        factor_lambda=1.0,
        t_calibrate=0.0,
        dt=0.025,
        v_init=-75.0,
    ):
        """
        Computes the steady state impedance matrix between all pairs of
        locations in `loc_arg`, by measuring the voltage deflections in response
        to a constant input current.

        Parameters
        ----------
        loc_arg : `list` of locations or string
            if `list` of locations, specifies the locations for which the
            impedance matrix is evaluated, if ``string``, specifies the
            name under which a set of location is stored
        i_amp : float, optional
            amplitude of the input current [nA], by default 0.001
        t_dur : float, optional
            duration of the input current [ms], by default 100.0
        pplot : bool, optional
            whether to plot the voltage traces, by default False
        factor_lambda : float or int, optional
            If int, the number of segments per section. If float, multiplies the
            number of segments given by the standard lambda rule (Carnevale, 2004)
            to give the number of compartments simulated, by default 1.0
        t_calibrate : float, optional
            Time for the model to equilibrate``[ms]``. Not counted as part of the
            simulation., by default 0.0
        dt : float, optional
            Timestep of the simulation, by default 0.025
        v_init : float, optional
            The initial voltage at which the model is initialized ``[mV]``,
            by default -75.0

        Returns
        -------
        np.ndarray (ndim=2)
            The impedance matrix [MOhm]
        """
        locs = self.convert_loc_arg_to_locs(loc_arg)
        z_mat = np.zeros((len(locs), len(locs)))
        for ii, loc0 in enumerate(locs):
            for jj, loc1 in enumerate(locs):
                self.init_model(
                    dt=dt,
                    t_calibrate=t_calibrate,
                    v_init=v_init,
                    factor_lambda=factor_lambda,
                )
                self.add_i_clamp(loc0, i_amp, 0.0, t_dur)
                self.store_locs([loc0, loc1], "rec locs", warn=False)
                # simulate
                res = self.run(t_dur)
                self.delete_model()
                # voltage deflections
                v_trans = res["v_m"][1][-int(1.0 / self.dt)] - res["v_m"][1][0]
                # compute impedances
                z_mat[ii, jj] = v_trans / i_amp
                if pplot:
                    import matplotlib.pyplot as pl

                    pl.figure()
                    pl.plot(res["t"], res["v_m"][1])
                    pl.show()

        return z_mat

    def calc_zt(
        self,
        loc0,
        loc1,
        i_amp=0.001,
        dt_pulse=0.1,
        dstep=-2,
        t_max=100.0,
        factor_lambda=1.0,
        t_calibrate=0.0,
        dt=0.025,
        v_init=-75.0,
    ):
        """
        Computes the impulse response kernel between two locations by measuring the
        voltage at `loc1` in response to an input current pulse at `loc0`.

        Parameters
        ----------
        loc0: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer kernel is computed
        loc1: dict, tuple or `:class:MorphLoc`
            One of two locations between which the transfer kernel is computed
        i_amp : float, optional
            amplitude of the input current pulse [nA], by default 0.001
        dt_pulse : float, optional
            duration of the input current pulse, by default 0.1
        dstep : int, optional
            offset form t=0 in no. of timesteps, by default -2
        t_max : float, optional
            simulation time, by default 100.0
        factor_lambda : float, optional
            If int, the number of segments per section. If float, multiplies the
            number of segments given by the standard lambda rule (Carnevale, 2004)
            to give the number of compartments simulated (default value 1. gives
            the number given by the lambda rule), by default 1.0
        t_calibrate : float, optional
            Time for the model to equilibrate``[ms]``. Not counted as part of the
            simulation., by default 0.0
        dt : float, optional
            Timestep of the simulation, by default 0.025
        v_init : float, optional
            The initial voltage at which the model is initialized ``[mV]``
            Returns, by default -75.0

        Returns
        -------
        np.array
            The time array in [ms]
        np.array
            The impulse response kernel in [MOhm/ms]
        """
        loc0, loc1 = self.convert_loc_arg_to_locs([loc0, loc1])
        self.set_simulation_parameters(
            dt=dt, t_calibrate=t_calibrate, v_init=v_init, factor_lambda=factor_lambda
        )
        t0 = 5.0
        j0 = int(t0 / self.dt)
        nt = int(t_max / self.dt) - 1
        i0 = int(dt_pulse / self.dt)
        if dstep < -i0:
            dstep = -i0
        self.init_model(
            dt=dt,
            t_calibrate=t_calibrate,
            v_init=v_init,
            factor_lambda=factor_lambda,
        )
        self.add_i_clamp(loc0, i_amp, t0, dt_pulse)
        self.store_locs([loc0, loc1], "rec locs", warn=False)
        # simulate
        res = self.run(t_max + dt_pulse + 3.0 * t0)
        self.delete_model()
        # voltage deflections
        v_trans = (
            res["v_m"][1][j0 + i0 + dstep : j0 + i0 + dstep + nt] - res["v_m"][1][0]
        )
        # compute impedances
        z_trans = v_trans / (i_amp * dt_pulse)

        return res["t"][i0 + dstep : i0 + dstep + nt], z_trans

    def calc_impulse_response_matrix(
        self,
        loc_arg,
        i_amp=0.001,
        dt_pulse=0.1,
        dstep=-2,
        t_max=100.0,
        factor_lambda=1.0,
        t_calibrate=0.0,
        dt=0.025,
        v_init=-75.0,
    ):
        """
        Computes matrix of impulse response kernels between any pairs of locations
        in `loc_arg` by measuring the voltage in response to an input current pulse.

        Parameters
        ----------
        loc_arg : `list` of locations or string
            if `list` of locations, specifies the locations for which the
            impulse response kernels are evaluated, if ``string``, specifies the
            name under which a set of location is stored
        i_amp : float, optional
            amplitude of the input current pulse [nA], by default 0.001
        dt_pulse : float, optional
            duration of the input current pulse, by default 0.1
        dstep : int, optional
            offset form t=0 in no. of timesteps, by default -2
        t_max : float, optional
            simulation time, by default 100.0
        factor_lambda : float, optional
            If int, the number of segments per section. If float, multiplies the
            number of segments given by the standard lambda rule (Carnevale, 2004)
            to give the number of compartments simulated (default value 1. gives
            the number given by the lambda rule), by default 1.0
        t_calibrate : float, optional
            Time for the model to equilibrate``[ms]``. Not counted as part of the
            simulation., by default 0.0
        dt : float, optional
            Timestep of the simulation, by default 0.025
        v_init : float, optional
            The initial voltage at which the model is initialized ``[mV]``
            Returns, by default -75.0

        Returns
        -------
        np.array
            The time array in [ms]
        np.ndarray (ndim=3)
            The impulse response kernel in [MOhm/ms], first dimension corresponds to time,
            last two dimansion correspond to the lcoations
        """
        self.set_simulation_parameters(
            dt=dt, t_calibrate=t_calibrate, v_init=v_init, factor_lambda=factor_lambda
        )
        locs = self.convert_loc_arg_to_locs(loc_arg)
        t0 = 5.0
        j0 = int(t0 / self.dt)
        nt = int(t_max / self.dt) - 1
        i0 = int(dt_pulse / self.dt)
        if dstep < -i0:
            dstep = -i0
        zk_mat = np.zeros((nt, len(locs), len(locs)))
        for ii, loc_in in enumerate(locs):
            self.init_model(
                dt=dt,
                t_calibrate=t_calibrate,
                v_init=v_init,
                factor_lambda=factor_lambda,
            )
            self.add_i_clamp(loc_in, i_amp, t0, dt_pulse)
            self.store_locs(locs, "rec locs", warn=False)
            # simulate
            res = self.run(t_max + dt_pulse + 3.0 * t0)
            self.delete_model()
            # voltage deflections
            v_trans = (
                res["v_m"][:, j0 + i0 + dstep : j0 + i0 + dstep + nt]
                - res["v_m"][:, 0:1]
            )
            # compute impulse response kernels
            zk_mat[:, ii, :] = v_trans.T / (i_amp * dt_pulse)
        return res["t"][i0 + dstep : i0 + dstep + nt], zk_mat
