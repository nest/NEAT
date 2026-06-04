# -*- coding: utf-8 -*-
#
# kernelextraction.py
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
import scipy.linalg as la
from scipy.cluster.vq import kmeans

import math
import copy
from typing import Literal


class Kernel:
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
            self.a = copy.deepcopy(kernel["a"])
            self.c = copy.deepcopy(kernel["c"])
        elif isinstance(kernel, float) or isinstance(kernel, int):
            self.a = np.array([1.0])
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
        if ind == 0:
            return self.a
        elif ind == 1:
            return self.c
        elif ind == "a":
            return self.a
        elif ind == "c":
            return self.c
        elif ind == "alphas":
            return self.a
        elif ind == "gammas":
            return self.c
        else:
            raise IndexError("Index should be '0' or '1'")

    def __call__(self, t_arr):
        return (
            np.dot(
                np.exp(-t_arr[:, np.newaxis] * self.a[np.newaxis, :]),
                self.c[:, np.newaxis],
            )
            .flatten()
            .real
        )

    def __add__(self, kernel):
        if kernel.a.shape[0] == self.a.shape[0] and np.allclose(kernel.a, self.a):
            a = copy.copy(self.a)
            c = kernel.c + self.c
        else:
            a = np.concatenate((self.a, kernel.a))
            c = np.concatenate((self.c, kernel.c))
        return Kernel((a, c))

    def __sub__(self, kernel):
        if kernel.a.shape[0] == self.a.shape[0] and np.allclose(kernel.a, self.a):
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
        raise AttributeError(
            "`k_bar` is a read-only attribute, adjust attribute `c` "
            + "by multiplying with a factor to change `k_bar`"
        )

    k_bar = property(get_k_bar, set_k_bar)

    def __str__(self, as_timescale=False):
        if as_timescale:
            return (
                "t = "
                + np.array2string(1.0 / self.a, precision=4, max_line_width=1000)
                + "\n"
                + "c = "
                + np.array2string(self.c, precision=4, max_line_width=1000)
            )
        else:
            return (
                "a = "
                + np.array2string(self.a, precision=4, max_line_width=1000)
                + "\n"
                + "c = "
                + np.array2string(self.c, precision=4, max_line_width=1000)
            )

    def __repr__(self):
        return repr({"a": self.a, "c": self.c})

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
            return Kernel(
                {
                    "a": self.a,
                    "c": -self.a * self.c,
                }
            )
        else:
            return (
                np.dot(
                    -self.a[np.newaxis, :]
                    * np.exp(-t_arr[:, np.newaxis] * self.a[np.newaxis, :]),
                    self.c[:, np.newaxis],
                )
                .flatten()
                .real
            )

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
        return np.sum(
            self.c[:, None] * 1e3 / (self.a[:, None] * 1e3 + s_arr[None, :]), 0
        )

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
        A = np.exp(-t_arr[:, None] * self.a[None, :])

        self.c = np.linalg.lstsq(w[:, None] * A, w * func_arr, rcond=None)[0]


class Fitter(object):
    def der(self, x, arr):
        dx = x[1] - x[0]
        diffarr = (arr[1:] - arr[0:-1]) / dx
        return diffarr, x[0:-1] + dx / 2

    def derder(self, x, arr):
        dx = x[1] - x[0]
        diffarr, _ = self.der(x, arr)
        diffdiffarr = (diffarr[1:] - diffarr[0:-1]) / dx
        return diffdiffarr, x[1:-1]

    def zerocrossing(self, x, arr):
        arr = copy.copy(arr)
        inds = np.where(np.diff(np.sign(arr)))[0]
        return inds, x[inds]

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx


class ExpFitter(Fitter):
    def sumExp(self, x, a, c, flat=True):
        if flat:
            return (np.exp(x[:, None] * a[:, None].T).dot(c[:, None])).flatten().real
        else:
            return np.exp(x[:, None] * a[:, None].T).dot(c[:, None]).real

    def PronyExpFit(self, deg, x, y):
        """
        Construct a sum of exponentials fit to a given time-sequence y by
        using prony's method

        input:
            [deg]: int, number of exponentials
            [x]: numpy array, sequence of regularly spaced points at which y is evaluated
            [y]: numpy array, sequence

        output:
            [a]: numpy array, exponential coefficient
            [c]: numpy array, exponentail magnitudes
            [rms]: float, root mean square error of the data
        """
        # deg += 1
        # stepsize
        h = x[1] - x[0]
        # Build matrix
        A = la.hankel(y[:-deg], y[-deg - 1 :])
        a = -A[:, :deg]
        b = A[:, deg]
        # Solve it
        s = la.lstsq(a, b)[0]
        # Solve polynomial
        p = np.flipud(np.hstack((s, 1)))
        u = np.roots(p)
        # Only keep roots in unit circle
        inds = np.where(
            np.logical_and(
                (np.abs(u) < 1.0),
                np.logical_not(np.logical_and(np.imag(u) == 0.0, np.real(u) <= 0.0)),
            )
        )[0]
        u = u[inds]
        # Calc exponential factors
        a = np.log(u) / h
        # Build power matrix
        A = np.power(
            (np.ones((len(y), 1)) * u[:, None].T),
            np.arange(len(y))[:, None] * np.ones((1, len(inds))),
        )
        # solve it
        f = la.lstsq(A, y)[0]
        # calc amplitudes
        c = f / np.exp(a * x[0])
        # build x, approx and calc rms
        approx = self.sumExp(x, a, c).real
        rms = np.sqrt(((approx - y) ** 2).sum() / len(y))
        return a, c, rms

    def construct_Hankel_matrices(self, y):
        ind0 = int(len(y) / 2)
        # original and shifted hankel matrix
        H0 = la.hankel(y[0:ind0], y[ind0 - 1 : 2 * ind0 - 1])
        H1 = la.hankel(y[1 : ind0 + 1], y[ind0 : 2 * ind0])
        return H0, H1

    # def Z_McE_ExpFit(self, x, y, deg=2):
    #     # construct the Hankel matrices
    #     H0, H1 = self.construct_Hankel_matrices(y)
    #     # compute the singular value decomposition
    #     U, s, Vh = la.svd(H0)
    #     U_ = U[:, 0:deg]
    #     Vh_ = Vh[0:deg, :]
    #     s_ = s[0:deg]
    #     # compute system matrix
    #     F0 = np.diag(1./np.sqrt(s_)).dot(U_.T)
    #     F1 = Vh_.T.dot(np.diag(1./np.sqrt(s_)))
    #     A = F0.dot(H1.dot(F1))
    #     # find eigenvalues of system matrix
    #     u, v = la.eig(A)
    #     # system time-scales (inverse)
    #     alphas = np.log(u) / dx
    #     return alphas

    def fitExp_Z_McE(self, x, y, rtol=1e-2, maxdeg=10):
        deg = 1
        rms = 1.0
        # stepsize
        dx = x[1] - x[0]
        # construct the Hankel matrices
        H0, H1 = self.construct_Hankel_matrices(y)
        # compute the singular value decomposition
        U, s, Vh = la.svd(H0)
        # loop over increasing number of exponentials
        while rms > rtol and deg < maxdeg:
            U_ = U[:, 0:deg]
            Vh_ = Vh[0:deg, :]
            s_ = s[0:deg]
            # compute system matrix
            F0 = np.diag(1.0 / np.sqrt(s_)).dot(U_.T)
            F1 = Vh_.T.dot(np.diag(1.0 / np.sqrt(s_)))
            A = F0.dot(H1.dot(F1))
            # find eigenvalues of system matrix
            u, v = la.eig(A)
            # system time-scales (inverse)
            alphas = np.log(u.real) / dx
            # solve weights
            A = np.exp(x[:, None] * alphas[None, :] * dx)
            gammas = la.lstsq(A, y)[0]
            # compute rmse
            approx = self.sumExp(x, alphas, gammas)
            rms = np.sqrt(((approx - y) ** 2).sum() / len(y))
            # increase degree
            deg += 1

        return alphas, gammas, rms

    def reduceSeries(self, a, c, x, y, rtol=1e-2):
        """
        Reduces the number of exponential terms in a series, till a given tolerance
        is reached

        input:
            [a]: numpy array of exponential timescales
            [c]: numpy array of exponential magnitudes
            [x]: numpy array of x-values at which the function is evaluated
            [y]: numpy array of function values
            [rtol]: float, relative tolerance given the largest function value
        output:
            [alpha]: exponential coefficients
            [gamma]: magnitudes
            [rms]: float, root mean square error
        """
        k = 1
        rms = 2 * rtol

        while rms > rtol and k <= len(a):
            sortind = np.argsort(np.abs(c))[::-1]
            alpha = a[sortind][0:k]
            gamma = c[sortind][0:k]

            approx = self.sumExp(x, alpha, gamma).real
            rms = np.sqrt(
                ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
            )
            k += 1

        return alpha, gamma, rms

    def fitExp(self, x, y, deg=30, rtol=1e-2, surface=False, A=None):
        a, c, rms = self.PronyExpFit(deg, x, y)
        alpha, gamma, rms = self.reduceSeries(a, c, x, y, rtol=rtol)
        if surface:
            dx = x[1] - x[0]
            if A == None:
                A = dx * np.sum(y)
            Afit = np.sum(
                gamma * (np.exp(alpha * x[-1]) - np.exp(alpha * x[0])) / alpha
            )
            gamma = gamma * A / Afit
            approx = self.sumExp(x, alpha, gamma).real
            rms = np.sqrt(
                ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
            )

        return alpha, gamma, rms


class fExpFitter(Fitter):
    def sumFExp(self, s, alphas, gammas):
        return np.sum(self.fexps(s, alphas, gammas), 0)

    def fexps(self, s, alphas, gammas):
        return gammas[:, None] / (alphas[:, None] + s[None, :])

    def trialFunFit(self, s, arr, alphas, pairs=None):
        # construct matrix for extended fitting problem
        A = np.concatenate(
            (
                1.0 / (s[:, None] + alphas[None, :]),
                arr[:, None] / (s[:, None] + alphas[None, :]),
            ),
            axis=1,
        )
        # find auxiliary residues
        c = la.lstsq(A, arr)[0][-len(alphas) :]
        # find zeros of fitted auxiliary function
        H = np.diag(alphas) - np.dot(
            np.ones((len(alphas), 1), dtype=complex), c[None, :]
        )
        alphanew = np.linalg.eig(H)[0]
        # find real residues
        Anew = 1.0 / (s[:, None] + alphanew[None, :])
        cnew = la.lstsq(Anew, arr)[0]

        return alphanew, cnew, None

    def trialFunFit_constrained(self, s, arr, alphas, pairs, zerostart=False):
        deg = len(alphas)
        carr = np.concatenate((arr.real, arr.imag))
        # construct matrix for extended fitting problem
        A = np.concatenate(
            (
                1.0 / (s[:, None] + alphas[None, :]),
                arr[:, None] / (s[:, None] + alphas[None, :]),
            ),
            axis=1,
        )
        # implement the constraint
        pairsnew = np.concatenate((pairs, pairs))
        for i, p in enumerate(pairsnew):
            if p:
                x1 = A[:, i] + A[:, i + 1]
                x2 = 1j * (A[:, i] - A[:, i + 1])
                A[:, i] = x1
                A[:, i + 1] = x2
        A = np.concatenate((A.real, A.imag), axis=0)
        # find auxiliary residues
        c = la.lstsq(A, carr)[0][-len(alphas) :]
        # find zeros of fitted auxiliary function
        a = np.diag(alphas)
        b = np.ones(deg)
        # implement similarity transform
        for i, p in enumerate(pairs):
            if p:
                a[i : i + 2, i : i + 2] = np.array(
                    [
                        [alphas[i].real, alphas[i].imag],
                        [-alphas[i].imag, alphas[i].real],
                    ]
                )
                b[i : i + 2] = np.array([2, 0])
        H = a.real - np.dot(b[:, None], c[None, :])
        alphanew = np.linalg.eig(H)[0]
        inds = np.argsort(alphanew)
        alphanew = alphanew[inds]
        # indicates where pairs of complex conjugate poles occur
        auxarr = np.abs(
            (np.abs(alphanew[:-1]) - np.abs(alphanew[1:])) / np.abs(alphanew[:-1])
        )
        auxarr2 = np.abs(alphas.imag) > 1e-15
        pairs = np.logical_and(
            np.concatenate((auxarr < 1e-15, np.zeros(1, dtype=bool))), auxarr2
        )
        # find residues
        Anew = 1.0 / (s[:, None] + alphanew[None, :])
        for i, p in enumerate(pairs):
            if p:
                x1 = Anew[:, i] + Anew[:, i + 1]
                x2 = 1j * (Anew[:, i] - Anew[:, i + 1])
                Anew[:, i] = x1
                Anew[:, i + 1] = x2
        Anew = np.concatenate((Anew.real, Anew.imag), axis=0)
        if zerostart:
            # enforce K(t=0)=0 constraint
            row1 = np.ones(2 * deg)
            for i, p in enumerate(pairs):
                if p:
                    row1[i + 1] = 0
            Anew = np.concatenate((np.ones((1, deg), dtype=complex), Anew), axis=0)
            carr = np.concatenate((np.zeros(1, dtype=complex), carr))
        cnew = la.lstsq(Anew, carr)[0]
        cnew = np.array(cnew, dtype=complex)
        # recast cnew to complex values
        for i, p in enumerate(pairs):
            if p:
                cnew[i : i + 2] = np.array(
                    [cnew[i] + 1j * cnew[i + 1], cnew[i] - 1j * cnew[i + 1]]
                )

        return alphanew, cnew, pairs

    def fit_residues(self, s, arr, alphas, pairs):
        carr = np.concatenate((arr.real, arr.imag))
        A = 1.0 / (s[:, None] + alphas[None, :])
        for i, p in enumerate(pairs):
            if p:
                x1 = A[:, i] + A[:, i + 1]
                x2 = 1j * (A[:, i] - A[:, i + 1])
                A[:, i] = x1
                A[:, i + 1] = x2
        A = np.concatenate((A.real, A.imag), axis=0)
        cnew = la.lstsq(A, carr)[0]
        cnew = np.array(cnew, dtype=complex)
        # recast cnew to complex values
        for i, p in enumerate(pairs):
            if p:
                cnew[i : i + 2] = np.array(
                    [cnew[i] + 1j * cnew[i + 1], cnew[i] - 1j * cnew[i + 1]]
                )
        return cnew

    def trialFunFit_constrained_2d(self, s, arr2d, alphas, pairs):
        print(">>> multifun fit test v2 <<<")
        deg = len(alphas)
        # construct f array
        arr1d = np.array([], dtype=complex)
        for ind, arr in enumerate(arr2d):
            arr1d = np.concatenate((arr1d, arr))
        # construct matrix A
        ns = len(s)
        ncols = (len(arr2d) + 1) * deg
        nrows = len(arr1d)
        A = np.zeros((nrows, ncols), dtype=complex)
        for ind, fis in enumerate(arr1d):
            indA = int(ind / ns)
            A[ind, deg * indA : deg * (indA + 1)] = 1.0 / (s[ind % ns] + alphas)
            # try:
            #     A[ind,deg*indA:deg*(indA+1)] = 1./(s[ind%ns] + alphas)
            # except ValueError:
            #     print indA
            #     print deg*indA
            #     print deg*(indA+1)
            #     print ncols
            A[ind, -deg:] = -fis / (s[ind % ns] + alphas)
        # implement the constraint
        for j in range(len(arr2d) + 1):
            for i, p in enumerate(pairs):
                if p:
                    x1 = A[:, j * deg + i] + A[:, j * deg + i + 1]
                    x2 = 1j * (A[:, j * deg + i] - A[:, j * deg + i + 1])
                    A[:, j * deg + i] = x1
                    A[:, j * deg + i + 1] = x2
        A = np.concatenate((A.real, A.imag), axis=0)
        arr1d = np.concatenate((arr1d.real, arr1d.imag))
        # find auxiliary residues
        c = la.lstsq(A, arr1d)[0][-len(alphas) :]
        print("cnew: ", c)
        # find zeros of fitted auxiliary function
        a = np.diag(alphas)
        b = np.ones(deg)
        # implement similarity transform
        for i, p in enumerate(pairs):
            if p:
                a[i : i + 2, i : i + 2] = np.array(
                    [
                        [alphas[i].real, alphas[i].imag],
                        [-alphas[i].imag, alphas[i].real],
                    ]
                )
                b[i : i + 2] = np.array([2, 0])
        # compute zeros of sum sigmafit
        H = a.real - np.dot(b[:, None], c[None, :])
        print("H: ", H)
        alphanew = np.linalg.eig(H)[0]
        print("alphanew: ", alphanew)
        inds = np.argsort(alphanew)
        alphanew = alphanew[inds]
        # indicates where pairs of complex conjugate poles occur
        auxarr = np.abs(
            (np.abs(alphanew[:-1]) - np.abs(alphanew[1:])) / np.abs(alphanew[:-1])
        )
        auxarr2 = np.abs(alphanew.imag) > 1e-15  # np.abs(alphas.imag) > 1e-15
        pairs = np.logical_and(
            np.concatenate((auxarr < 1e-15, np.zeros(1, dtype=bool))), auxarr2
        )
        # find residues
        # compute matrix for residue calculation
        Anew = 1.0 / (s[:, None] + alphanew[None, :])
        for i, p in enumerate(pairs):
            if p:
                x1 = Anew[:, i] + Anew[:, i + 1]
                x2 = 1j * (Anew[:, i] - Anew[:, i + 1])
                Anew[:, i] = x1
                Anew[:, i + 1] = x2
        Anew = np.concatenate((Anew.real, Anew.imag), axis=0)
        # compute residues
        c2dnew = np.zeros((arr2d.shape[0], deg), dtype=complex)
        for ind, arr in enumerate(arr2d):
            carr = np.concatenate((arr.real, arr.imag))
            cnew = la.lstsq(Anew, carr)[0]
            cnew = np.array(cnew, dtype=complex)
            # recast cnew to complex values
            for i, p in enumerate(pairs):
                if p:
                    cnew[i : i + 2] = np.array(
                        [cnew[i] + 1j * cnew[i + 1], cnew[i] - 1j * cnew[i + 1]]
                    )
            c2dnew[ind, :] = cnew

        print("cnew: ", c2dnew)

        return alphanew, c2dnew, pairs

    def reduceSeries(self, s, y, a, c, pairs=None, rtol=1e-2, pprint=False):
        """
        reduce the series of exponentials after the fitting
        """
        k = 1
        rms = 1.0

        # ensure stability of approximation
        inds = np.where(a.real > 0.0)[0]
        a = a[inds]
        c = c[inds]
        if pairs is not None:
            pairs = pairs[inds]

        # construct indices for ranking the exponentials
        pairs_alltrue = copy.copy(pairs)
        for i, p in enumerate(pairs):
            if p:
                pairs_alltrue[i + 1] = True
        magnitudes = np.zeros(a.shape)
        for i in range(len(pairs_alltrue)):
            if pairs_alltrue[i]:
                c_ = c[i].real
                c__ = c[i].real
                a_ = a[i].real
                a__ = a[i].real
                magnitudes[i] = (c_ * a_ + c__ * a__) / (a_**2 + a__**2)
            else:
                magnitudes[i] = c[i].real / a[i].real

        sortind = np.argsort(np.abs(magnitudes))[::-1]

        anew = copy.copy(a[sortind])
        alphas = anew
        cnew = copy.copy(c[sortind])
        gammas = cnew
        # look for pairs to be sure they are correct
        auxarr = np.abs(
            (np.abs(alphas[:-1]) - np.abs(alphas[1:])) / np.abs(alphas[:-1])
        )
        auxarr2 = np.abs(alphas.imag) > 1e-15
        pairs = np.logical_and(
            np.concatenate((auxarr < 1e-15, np.zeros(1, dtype=bool))), auxarr2
        )
        npairs = copy.copy(pairs)

        approx = self.sumFExp(s, alphas, gammas)

        while rms > rtol and k < len(a) + 1:
            if (pairs is not None) and pairs[k - 1]:
                k += 1
            alphas = anew[0:k]
            gammas = cnew[0:k]

            auxarr = np.abs(
                (np.abs(alphas[:-1]) - np.abs(alphas[1:])) / np.abs(alphas[:-1])
            )
            auxarr2 = np.abs(alphas.imag) > 1e-15
            npairs = np.logical_and(
                np.concatenate((auxarr < 1e-15, np.zeros(1, dtype=bool))), auxarr2
            )

            approx = self.sumFExp(s, alphas, gammas)
            rms = np.sqrt(
                ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
            )
            k += 1

        if pprint:
            pairinds = copy.copy(npairs)
            inds = np.where(npairs)[0]
            for i in inds:
                pairinds[i + 1] = True
            inds = np.where(np.logical_not(pairinds))[0]
            if len(inds) > 0:
                if np.max(np.abs(alphas[inds].imag)) > 1e-6:
                    print("!!! Warning: invalid pairs !!!")
                    print("original alphas: ", anew)
                    print("original gammas: ", cnew)
                    print("original pairs: ", pairs)
                    print("new alphas: ", alphas)
                    print("new gammas: ", gammas)
                    print("new pairs: ", npairs)

        return alphas, gammas, rms, approx, npairs

    def _find_start_nodes(self, s, deg, realpoles, initpoles):
        if not isinstance(initpoles, str):
            trialpoles = initpoles.astype(complex)
        elif initpoles == "lin":
            trialpoles = np.linspace(s[int(len(s) / 2.0) + 1].imag, s[-1].imag, deg)
        elif initpoles == "log10":
            trialpoles = np.logspace(1, np.log10(s[-1].imag), num=deg, base=10)
        elif initpoles == "log":
            trialpoles = np.logspace(1, np.log(s[-1].imag), num=deg, base=math.e)
        elif initpoles == "random":
            trialpoles = s[-1].imag * np.random.rand(deg)
        else:
            raise Exception("initpoles invalid")
        if realpoles:
            pairs = np.zeros(trialpoles.shape, dtype=bool)
        else:
            trialpoles = np.array(
                [[tp + 1j * tp, tp - 1j * tp] for tp in trialpoles]
            ).flatten()
            pairs = np.array([[True, False] for _ in range(deg)]).flatten()
        return trialpoles, pairs

    def _run_fit(
        self,
        s,
        y,
        trialpoles,
        pairs,
        rtol,
        maxiter,
        constrained,
        zerostart,
        pole_flip=True,
        pprint=True,
    ):
        """
        performs iterations of the actual fitting process
        """
        k = 0
        rms = rtol + 1.0
        l = 0
        m = 0

        alist = []
        clist = []
        rmslist = []
        pairslist = []

        trialpoles_orig = copy.copy(trialpoles)
        pairs_orig = copy.copy(pairs)

        if constrained:
            while rms > rtol and k < maxiter:
                a, c, pairs = self.trialFunFit_constrained(
                    s, y, trialpoles, pairs, zerostart=zerostart
                )
                approx = self.sumFExp(s, a, c)
                rms = np.sqrt(
                    ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
                )
                # if unstable poles, make sure to run again
                # if np.min(a) < 0.:
                #     rms = rtol + 1.
                #     if m < 10.:
                #         if pole_flip:
                ind = np.where(a < 0.0)[0]  # where poles are unstable
                if len(ind) > 0:
                    a[ind] *= -1.0
                    c = self.fit_residues(s, y, a, pairs)
                    approx = self.sumFExp(s, a, c)
                    rms = np.sqrt(
                        ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
                    )
                    # if rms < rtol:
                    #     alist.append(copy.deepcopy(a)); clist.append(copy.deepcopy(c)); rmslist.append(rms); pairslist.append(pairs)
                    # else:
                    #     ind = np.where(a > 0.)[0] # where poles are stable
                    #     newpole, newpair = self._find_start_nodes(s, len(a)-len(ind), True, 'random')
                    #     trialpoles = np.concatenate((a[ind], newpole))
                    #     pairs = np.concatenate((pairs[ind], newpair))
                    # else:
                    #     trialpoles, pairs = self._find_start_nodes(s, len(trialpoles_orig), True, 'random')
                    #     m = 0
                    # l += 1; m += 1
                # else:
                alist.append(copy.deepcopy(a))
                clist.append(copy.deepcopy(c))
                rmslist.append(rms)
                pairslist.append(pairs)
                trialpoles = copy.copy(a)
                k += 1
            if pprint and l > 5:
                print("Often found unstable poles (" + str(l) + " times)")
            return alist, clist, rmslist, pairslist
        else:
            while rms > rtol and k < maxiter:
                a, c, _ = self.trialFunFit(s, y, trialpoles, zerostart)
                approx = self.sumFExp(s, a, c)
                rms = np.sqrt(
                    ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
                )
                trialpoles = a
                k += 1
            return alist, clist, rmslist, None

    def _run_fit_vector(self, s, ys, trialpoles, pairs, rtol, maxiter):
        # eps = 2.
        # k = 0; rms = 1.
        # rms_ = rms

        # alist = []; clist = []; rmslist = []; pairslist = []

        # while rms > rtol and k < maxiter:

        #     a, c2d, pairs = self.trialFunFit_constrained_2d(s, ys, trialpoles, pairs)
        #     rms = 0.
        #     for ind, y in enumerate(ys):
        #         approx = self.sumFExp(s, a, c2d[ind])
        #         rms += np.sqrt(((np.abs(approx-y) / np.max(np.abs(y)))**2).sum() / len(y))
        #     alist.append(copy.deepcopy(a)); clist.append(copy.deepcopy(c2d)); rmslist.append(rms); pairslist.append(pairs)
        #     # randomize poles a bit
        #     skip = False
        #     tp = copy.deepcopy(a)
        #     if (rms_ - rms) / rms_ < eps:
        #         for i, p in enumerate(pairs):
        #             if not skip:
        #                 if p:
        #                     x1 = 0.1 * tp[i].real; x2 = 0.1 * np.abs(tp[i].imag)
        #                     r1 = x1 * (2. * np.random.rand() - 1); r2 = x2 * (2. * np.random.rand() - 1)
        #                     tp[i:i+2] = np.array([tp[i] + r1 + 1j*r2, tp[i+1] + r1 - 1j*r2])
        #                     skip = True
        #                 else:
        #                     x = 0.1 * tp[i]
        #                     r = x * (2. * np.random.rand() - 1)
        #                     tp[i] += r
        #                     skip = False
        #     trialpoles = tp
        #     k += 1
        #     rms_ = rms
        eps = 2.0
        k = 0
        rms = 1.0
        rms_ = rms

        alist = []
        clist = []
        rmslist = []
        pairslist = []

        while rms > rtol and k < maxiter:
            a2d = np.zeros((len(ys), len(trialpoles)), dtype=complex)
            c2d = np.zeros((len(ys), len(trialpoles)), dtype=complex)
            pairs2d = np.zeros((len(ys), len(trialpoles)), dtype=bool)
            for ind, y in enumerate(ys):
                a2d[ind], c2d[ind], pairs2d[ind] = self.trialFunFit_constrained(
                    s, y, trialpoles, pairs
                )
                # put complex conjugates with positive part first
                for i, p in enumerate(pairs2d[ind]):
                    if p:
                        if a2d[ind, i] < 0:
                            a2d[ind, i] = a2d[ind, i].real - 1j * a2d[ind, i].imag
                            a2d[ind, i + 1] = (
                                a2d[ind, i + 1].real - 1j * a2d[ind, i + 1].imag
                            )
                            c2d[ind, i] = c2d[ind, i].real - 1j * c2d[ind, i].imag
                            c2d[ind, i + 1] = (
                                c2d[ind, i + 1].real - 1j * c2d[ind, i + 1].imag
                            )
            a, pairs = self._Kmeans(a2d, pairs2d)
            c2d = np.zeros((len(ys), len(a)), dtype=complex)
            for ind, y in enumerate(ys):
                c2d[ind] = self.fit_residues(s, y, a, pairs)
                approx = self.sumFExp(s, a, c2d[ind])
                rms += np.sqrt(
                    ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
                )
            alist.append(copy.deepcopy(a))
            clist.append(copy.deepcopy(c2d))
            rmslist.append(rms)
            pairslist.append(pairs)
            # randomize poles a bit
            skip = False
            tp = copy.deepcopy(a)
            if (rms_ - rms) / rms_ < eps:
                for i, p in enumerate(pairs):
                    if not skip:
                        if p:
                            x1 = 0.1 * tp[i].real
                            x2 = 0.1 * np.abs(tp[i].imag)
                            r1 = x1 * (2.0 * np.random.rand() - 1)
                            r2 = x2 * (2.0 * np.random.rand() - 1)
                            tp[i : i + 2] = np.array(
                                [tp[i] + r1 + 1j * r2, tp[i + 1] + r1 - 1j * r2]
                            )
                            skip = True
                        else:
                            x = 0.1 * tp[i]
                            r = x * (2.0 * np.random.rand() - 1)
                            tp[i] += r
                            skip = False
            trialpoles = tp
            k += 1
        return alist, clist, rmslist, pairslist

    def _Kmeans(
        self, a2d, pairs2d
    ):  # do the kmeans algorithm to make sure all nodes are the same
        a1d = np.array([], dtype=complex)
        for i, a in enumerate(a2d):
            # determine the coefficients not to take into account in the algorithm
            paux = np.concatenate((np.array([False]), pairs2d[i, :-1]))
            inds = np.where(np.logical_not(paux))[0]
            a1d = np.concatenate((a1d, a[inds]))
        adata = np.concatenate((a1d.real[:, None], a1d.imag[:, None]), 1)
        astart = np.concatenate(
            (a2d[-1].real[inds][:, None], a2d[-1].imag[inds][:, None]), 1
        )
        a = kmeans(adata, astart)[0]
        # check for complex conjugates
        anew = []
        pairsnew = []
        for alpha in a:
            if np.abs(alpha[1]) > 1e-9:
                anew.append(alpha[0] + 1j * alpha[1])
                anew.append(alpha[0] - 1j * alpha[1])
                pairsnew.append(True)
                pairsnew.append(False)
            else:
                anew.append(alpha[0] + 1j * 0.0)
                pairsnew.append(False)
        # a = a[:,0] + 1j* a[:,1]
        # look for pairs to be sure they are correct

        # auxarr = np.abs((np.abs(a[:-1]) - np.abs(a[1:])) / np.abs(a[:-1]))
        # auxarr2 = np.abs(a.imag) > 1e-15
        # pairs = np.logical_and(np.concatenate((auxarr < 1e-15, np.zeros(1, dtype=bool))), auxarr2)
        return np.array(anew), np.array(pairsnew)

    def reduceNumExp(self, s, y, a, c, pairs, lim=0.1, pprint=True, pplot=True):
        """
        pools the short timescale exponentials
        """
        # find inds of exponentials that have to be taken together
        inds = np.where(np.abs(a.real) > (1e3 / lim))[0]
        # the other indices stay the same
        inds_no = np.where(np.abs(a.real) <= (1e3 / lim))[0]
        anew = a[inds_no]
        cnew = c[inds_no]
        pairsnew = pairs[inds_no]
        if len(inds) > 1:
            amin = np.min(a[inds])
            EF = ExpFitter()
            if pplot == True:
                import matplotlib.pyplot as pl

                y_f_full = self.sumFExp(s, a, c)
                y_f_part = self.sumFExp(s, a[inds], c[inds])
                pl.figure("reduceNumExp problem")
                pl.plot(s.imag, y_f_full.real, "r")
                pl.plot(s.imag, y_f_part.real, "b")
                pl.plot(s.imag, y_f_full.imag, "r--")
                pl.plot(s.imag, y_f_part.imag, "b--")
                pl.show()
            # multiple step approach
            t = np.linspace(0.0, 5.0 / amin.real, 1000)
            y_t = EF.sumExp(t, -a[inds], c[inds])
            y_t_full = EF.sumExp(t, -a, c)
            A_t = -np.sum(
                c[inds] * (np.exp(-a[inds] * t[-1]) - np.exp(-a[inds] * t[0])) / a[inds]
            )
            y_t_lim = EF.sumExp(np.array([lim * 1e-3]), -a[inds], c[inds])
            y_t_lim_full = EF.sumExp(np.array([lim * 1e-3]), -a, c)
            # ~ print 'full sum at 1ms: ', y_t_lim_full
            # ~ print 'partial sum at 1ms: ', y_t_lim
            # ~ print 'max full sum: ', np.max(y_t_full[1:])
            # fit first outside of first timestep if necessary
            if amin.real < (2e4 / lim) and np.abs(
                y_t_lim_full - y_t_lim
            ) > 0.001 * np.max(y_t_full[1:]):
                t_out = np.linspace(lim * 1e-3, 5.0 / amin.real, 1000.0)
                y_out = EF.sumExp(t_out, -a[inds], c[inds])
                A_out = -np.sum(
                    c[inds]
                    * (np.exp(-a[inds] * t_out[-1]) - np.exp(-a[inds] * t_out[0]))
                    / a[inds]
                )
                try:
                    # if the maximum of the to be grouped exponentials is past lim,
                    # we use two exponentials, otherwise one
                    # ~ else:
                    A, C, _ = EF.fitExp(
                        t_out, y_out, deg=1, rtol=0.0001, surface=True, A=A_out
                    )
                    A = -A.real
                    C = C.real
                    Ptemp = [False]
                except ValueError:
                    A = np.array([amin.real])
                    C = A_out / ((np.exp(-A * t_out[-1]) - np.exp(-A * t_out[0])) / A)
                    Ptemp = [False]
                # check if we need to fit inside first timestep
                t_in = np.linspace(0.0, lim * 1e-3, 100)
                y_in = EF.sumExp(t_in, -a[inds], c[inds]) - EF.sumExp(t_in, -A, C)
                A_in_full = -np.sum(
                    c[inds]
                    * (np.exp(-a[inds] * t_in[-1]) - np.exp(-a[inds] * t_in[0]))
                    / a[inds]
                )
                A_in = -np.sum(C * (np.exp(-A * t_in[-1]) - np.exp(-A * t_in[0])) / A)
                if np.abs(A_in - A_in_full) < 0.01 * np.abs(A_in_full):
                    # we don't need to fit an extra exponential,
                    # but just rescale surfaces a bit
                    A_tot = np.sum(c[inds] / a[inds])
                    A_part = np.sum(C / A)
                    C = C * A_tot / A_part
                    P = np.array(Ptemp, dtype=bool)
                else:
                    # we need to fit an extra exponential
                    t = np.linspace(0.0, 3.0 / amin.real, 1000.0)

                    A_t = np.sum(c[inds] / a[inds])
                    A_exp1 = np.sum(C / A)
                    A2 = np.array([1e4 / lim], dtype=complex)
                    C2 = (A_t - A_exp1) * A2

                    P = np.array(Ptemp + [False], dtype=bool)
                    A = np.concatenate((A, A2))
                    C = np.concatenate((C, C2))
            else:
                # we can just fit inside the first timestep
                # construct new exponential naively
                A = np.array([amin.real], dtype=complex)
                C = np.sum(c[inds] / a[inds]) * A
                P = np.array([False], dtype=bool)

            # concatenate the arrays
            anew = np.concatenate((anew, A))
            cnew = np.concatenate((cnew, C))
            pairsnew = np.concatenate((pairsnew, P))

            if pprint or pplot:
                t = np.linspace(0.0, 0.050, 100000)
                A_original = -np.sum(c * (np.exp(-a * t[-1]) - np.exp(-a * t[0])) / a)
                A_new = -np.sum(
                    cnew * (np.exp(-anew * t[-1]) - np.exp(-anew * t[0])) / anew
                )
                if np.abs(A_original - A_new) > 1e-12 or np.isnan(A_new.real):
                    print("!!! Warning: surfaces under kernels not equal !!!")
                    print("oringal surface: ", A_original)
                    print("new surface: ", A_new)
                    print("all a's: ", a)
                    print("all gamma's: ", c)
                    print("all pairs: ", pairs)
                    print("tbg a's: ", a[inds])
                    print("tbg gamma's: ", c[inds])
                    print("tbg pairs: ", pairs[inds])
                    print("ntbg a's: ", a[inds_no])
                    print("ntbg gamma's: ", c[inds_no])
                    print("ntbg pairs: ", pairs[inds_no])
                    print("new a's: ", anew)
                    print("new c's: ", cnew)
                    print("new pairss: ", pairsnew)

            if pplot and (np.abs(A_original - A_new) > 1e-12 or np.isnan(A_new.real)):
                # ~ if pplot:
                t = np.linspace(0.0, 0.050, 100000)
                dt = t[1] - t[0]
                ef = ExpFitter()
                se_ = ef.sumExp(t, -a[inds], c[inds])
                e_ = ef.sumExp(t, -A, C)
                se = ef.sumExp(t, -a, c)
                e = ef.sumExp(t, -anew, cnew)
                print(
                    "integral original reduced: ",
                    -np.sum(
                        c[inds]
                        * (np.exp(-a[inds] * t[-1]) - np.exp(-a[inds] * t[0]))
                        / a[inds]
                    ),
                )
                print(
                    "integral fit reduced: ",
                    -np.sum(C * (np.exp(-A * t[-1]) - np.exp(-A * t[0])) / A),
                )
                print("final a's :", anew)
                print("new a's :", A)
                import matplotlib.pyplot as pl

                pl.figure("reduce_exp problem")
                pl.plot(t * 1000, se, "r", label="original kernel")
                pl.plot(t * 1000, e, "b", label="new kernel")
                pl.plot(t * 1000, se_, "r--", label="exps to be reduced")
                pl.plot(t * 1000, e_, "b--", label="to be reduced exp")
                pl.legend(loc=0)
                pl.show()

        # new approximation and rmse
        approx = self.sumFExp(s, anew, cnew)
        rms = np.sqrt(((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y))

        return anew, cnew, rms, approx, pairsnew

    def fitFExp_increment(
        self,
        s,
        y,
        rtol=1e-2,
        maxiter=20,
        maxiter_step=3,
        realpoles=True,
        constrained=True,
        zerostart=False,
        pprint=True,
    ):

        # find the start nodes
        trialpoles, pairs = self._find_start_nodes(s, 1, realpoles, "log10")

        for k in range(maxiter):
            alist, clist, rmslist, pairslist = self._run_fit(
                s,
                copy.copy(y),
                trialpoles,
                pairs,
                rtol,
                maxiter_step,
                constrained,
                zerostart,
                pole_flip=True,
                pprint=pprint,
            )
            indmin = np.argmin(np.array(rmslist))
            alpha = alist[indmin]
            gamma = clist[indmin]
            rms = rmslist[indmin]
            pairs = pairslist[indmin]
            if rms < rtol:
                break
            else:
                if realpoles:
                    # alphanew = [s[-1].imag * np.random.rand()]
                    alphanew = [np.random.choice(1.0 / s.imag)]
                    pairsnew = [False]
                else:
                    areal = s[-1].imag * np.random.rand()
                    aimag = s[-1].imag * np.random.rand()
                    alphanew = [areal + 1j * aimag, areal - 1j * aimag]
                    pairsnew = [True, False]
            trialpoles = np.array(alpha.tolist() + alphanew)
            # print trialpoles
            pairs = np.array(pairs.tolist() + pairsnew)
        rmsfinal = rms

        if pprint and rmsfinal > rtol:
            print("Target accuracy was not reached")

        return alpha, gamma, pairs, rmsfinal

    def fitFExp(
        self,
        s,
        y,
        deg=20,
        rtol=1e-2,
        maxiter=5,
        lim=None,
        realpoles=True,
        initpoles="lin",
        zerostart=False,
        constrained=True,
        reduce_numexp=False,
        return_real=False,
        lower_limit=None,
    ):
        """
        Fits a function in fourrierspace by a series of fourrier transformed exponentials.

        input:
            -args
            [s]: numpy array of frequencies (imaginary) at which value function is evaluated
            [y]: numpy array of complex function values
            -kwargs
            [deg]: int, number of exponential terms used (real number is dubbeled if realpoles=False)
            [rtol]: float, relative toleranse after which iterations stop
            [maxiter]: int, maximum number of iterations
            [lim]: float, smallest timescale to take into account [ms], if not None, the algorithm
                fits the slowest timescale first, then the next slowest, etc. !!Use only for
                decaying transfer functions!!
            [realpoles]: boolean, use real starting poles if true, use complex conjugate poles if
                false
            [initpoles]: numpy array or str
                'lin' for linearly spaced initial poles, 'log10' and 'log' for
                logarithmically spaced poles. If a numpy array is given, those values
                will be the initial polez (Hz).
            [zerostart]: boolean, constrain the function to be 0 at t=0 if true
            [constrained]: fix the poles to be complex conjugate pairs
            [reduce_numexp]: boolean, pool short time scale exponentials together if true
            [return_real]: boolean, fix the output to only contain real poles if true
            [lower_limit]: float, if not None, lower limit on the exponential prefactors

        output:
            [alpha]: numpy array of (complex) timescales of exponentials
            [gamma]: numpy array of complex magnitudes of exponentials
            [pairs]: boolean array that indicates True at every index where a complex
                conjugate pair occurs
            [rms]: float, root mean square error
        """
        trialpoles, pairs = self._find_start_nodes(s, deg, realpoles, initpoles)

        if lim != None:
            a_s = []
            c_s = []
            pair_s = []
            y_decr = copy.copy(y)
            deg_decr = deg
            keep_going = True
            count = 0
            while keep_going:
                alist, clist, rmslist, pairslist = self._run_fit(
                    s, y_decr, trialpoles, pairs, rtol, maxiter, constrained, zerostart
                )
                indmin = np.argmin(np.array(rmslist))
                anew, cnew, rmsnew, approx, pairsnew = self.reduceSeries(
                    s,
                    y_decr,
                    alist[indmin],
                    clist[indmin],
                    pairs=pairslist[indmin],
                    rtol=rtol,
                )
                if count == 0:
                    # save parameters for later purposes
                    asave = copy.copy(anew)
                    csave = copy.copy(cnew)
                    rmssave = rmsnew
                    pairssave = pairsnew
                    surface_original = np.sum(cnew / anew)
                ind = []
                # take the longest timescale out
                ind.append(np.argmin(anew.real))
                if pairsnew[ind]:
                    ind.append(ind[0] + 1)
                a_tba = anew[ind]
                c_tba = cnew[ind]
                pair_tba = pairsnew[ind]
                surface = np.sum(cnew / anew)
                y_old = copy.copy(y_decr)
                y_decr = self.sumFExp(s, anew, cnew) - self.sumFExp(
                    s, anew[ind], cnew[ind]
                )
                # ~ deg_decr -= len(ind)
                trialpoles, pairs = self._find_start_nodes(s, deg_decr, True, initpoles)
                # stop if timescale is small enough
                if anew[ind][0] > 1e3 / (0.2 * lim):
                    if len(ind) == 1:
                        c_tba = surface * a_tba
                    elif len(ind) == 2:
                        c_tba[0] = (
                            surface
                            * (a_tba[0].real ** 2 + a_tba[0].imag ** 2)
                            / (2.0 * a_tba[0].real)
                        )
                        c_tba[1] = c_tba[0]
                    else:
                        raise ValueError("invalid array length")
                    keep_going = False
                # stop if rmse is small enough
                approx = self.sumFExp(s, a_tba, c_tba)
                rms = np.sqrt(
                    ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
                )
                if rms < rtol:
                    keep_going = False
                # append new parameters to lists
                a_s += a_tba.tolist()
                c_s += c_tba.tolist()
                pair_s += pair_tba.tolist()
                # stop if to many parameters
                if count >= 9.0:
                    keep_going = False
                count += 1

            # for returning
            alpha = np.array(a_s, dtype=complex)
            gamma = np.array(c_s, dtype=complex)
            pairs = np.array(pair_s, dtype=bool)
            approx = self.sumFExp(s, alpha, gamma)
            rms = np.sqrt(
                ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
            )
            # check whether it was better to go with the first parameters
            if len(asave) < len(alpha) and rmssave < rtol:
                alpha = asave
                gamma = csave
                pairs = pairssave
                approx = self.sumFExp(s, alpha, gamma)
                rms = np.sqrt(
                    ((np.abs(approx - y) / np.max(np.abs(y))) ** 2).sum() / len(y)
                )
            surface_after = np.sum(gamma / alpha)
            if np.abs(surface_original - surface_after) > rtol * surface_original:
                print("surface original: ", surface_original)
                print("surface after: ", surface_after)
            if np.min(alpha.real) < 0.0:
                print("!!!owowow!!!")
        else:
            alist, clist, rmslist, pairslist = self._run_fit(
                s, y, trialpoles, pairs, rtol, maxiter, constrained, zerostart
            )
            indmin = np.argmin(np.array(rmslist))
            alpha, gamma, rms, approx, pairs = self.reduceSeries(
                s, y, alist[indmin], clist[indmin], pairs=pairslist[indmin], rtol=rtol
            )
            if reduce_numexp:
                alpha, gamma, rms, approx, pairs = self.reduceNumExp(
                    s, y, alpha, gamma, pairs, pplot=False
                )

        if return_real:
            inds = np.where(np.roll(pairs, 1))[0]
            alpha_real = np.delete(alpha.real, inds)
            alpha = alpha_real + 0j
            pairs = np.zeros(alpha.size, dtype=bool)
            gamma = self.fit_residues(s, y, alpha, pairs)

        if lower_limit:
            inds = np.where(alpha.real < lower_limit)[0]
            alpha = np.delete(alpha, inds)
            pairs = np.delete(pairs, inds)
            gamma = self.fit_residues(s, y, alpha, pairs)

        return alpha, gamma, pairs, rms

    def fitFExp_vector(
        self,
        s,
        ys,
        deg=20,
        rtol=1e-2,
        maxiter=5,
        extra_startpoles=[],
        extra_startpoles_pairs=[],
        realpoles=True,
        initpoles="lin",
        reduce_series=False,
    ):
        """
        Fit multiple data-arrays in Fourrier-domain simultaneously with a shared set of nodes

        input:
            [s]: numpy array of complex number, frequencies of data
            [ys]: numpy ndarray of complex numbers, rows are different data-arrays
            [deg]: int, the starting number of nodes
            [rtol]: float, the relative tolercance at which to stop
            [maxiter]: int, the maximal number of iterations after which to stop when rtol
                is not reached
            [extra_startpoles]: numpy array of complex number, additional initial poles
            [extra_startpoles_pairs]: numpy bolean array, indicates complex conjugate pairs
                associated with the extra initial poles
            [realpoles]: boolean, if True the starting poles are real, if false the starting
                poles are complex conjugates (and then the real degree is 2*deg)
            [initpoles]: string specifying how the initial poles are distributed, choices are
                'lin', 'log' and 'log10'
            [reduce_series]: boolean, whether to delete expontentials of small influence after
                the fitting

        output:
            [alpha]: complex numpy array of exponential coefficients
            [gamma]: 2d complex numpy array, each row contains the residues corresponding to
                the respective data arrays
            [pairs]: boolean numpy array, indicates where a pair of complex conjugate
                exponentials occurs
            [rms]: float, aggregated root mean square error
        """
        trialpoles, pairs = self._find_start_nodes(s, deg, realpoles, initpoles)
        if len(extra_startpoles) > 0:
            trialpoles = np.concatenate((trialpoles, extra_startpoles))
            pairs = np.concatenate((pairs, extra_startpoles_pairs))
        alist, clist, rmslist, pairslist = self._run_fit_vector(
            s, ys, trialpoles, pairs, rtol, maxiter
        )
        indmin = np.argmin(np.array(rmslist))
        if reduce_series:
            # reduce the number of exponentials for each function separately
            alpha_arr = np.array([], dtype=complex)
            rms = 0.0
            for ind, c in enumerate(clist[indmin]):
                alpha, gamma, rms_ind, approx, pair = self.reduceSeries(
                    s, ys[ind], alist[indmin], c, pairs=pairslist[indmin], rtol=rtol
                )
                rms += rms_ind
                alpha_arr = np.concatenate((alpha_arr, alpha))
            alpha_arr = np.unique(alpha_arr)
            # search positions of common alphas
            asortind = np.argsort(alist[indmin])
            alphapos = np.searchsorted(alist[indmin][asortind], alpha_arr)
            inds = asortind[alphapos]
            return (
                alist[indmin][inds],
                clist[indmin][:, inds],
                pairslist[indmin][inds],
                rms,
            )
        else:
            return alist[indmin], clist[indmin], pairslist[indmin], rmslist[indmin]


def create_logspace_freqarray(fmax=7, base=10, num=200):
    a = np.logspace(1, fmax, num=num, base=base)
    b = np.linspace(-base, base, num=num // 2 + 1)
    # b = np.linspace(-base, base, num=num/2+(num/2)%2)[:-1]
    return 1j * np.concatenate((-a[::-1], b[1:-1], a))


class FourierQuadrature(object):
    """
    Performs an accurate Fourrier transform on functions
    evaluated at a given array of temporal grid points

    Parameters
    ----------
    tarr: `np.array` of floats,
        the time points (ms) at which the function is evaluated, have to be
        regularly spaced
    fmax: float, optional (default ``7.``)
        the maximum value to which the logarithm is evaluated to get the
        maximum evaluation frequency
    base: float, optional (defaul ``10``)
        the base of the logarithm used to generated the logspace
    num: int, even, optional (default ``200``)
        Number of points. the eventual number of points in frequency space
        is (2+1/2)*num

    Attributes
    ----------
    s: np.array of complex
        The frequencies at which input arrays in the Fourrier domain are
        supposed to be evaluated
    t: np.array of real
        The time array at which input arrays in the time domain are supposed
        to be evaluated
    ind_0s: int
        Index of the zero frequency component in `self.s`
    """

    def __init__(self, tarr, fmax=7, base=10, num=200):
        assert num % 2 == 0
        # create the frequency points at which to evaluate the transform
        self.s = create_logspace_freqarray(fmax=fmax, base=base, num=num)
        self.t = tarr
        self.ind_0s = len(self.s) // 2
        # create the quadrature matrix
        self._setQuad()
        self._setQuadInv()

    def _setQuad(self):
        s = self.s
        t = self.t
        N = len(t)
        dt = (t[1] - t[0]) * 1e-3
        c = np.zeros((len(s), N), dtype=complex)
        Nr = np.arange(1, N - 1)[np.newaxis, :]
        sc = s[:, np.newaxis]
        mask_arr = np.abs(sc) > 1e-12
        # first frequency integral
        np.divide(np.exp(-sc * dt) - 1.0, -sc * dt, out=c[:, 0:1], where=mask_arr)
        np.divide(-1.0 + c[:, 0:1], -sc, out=c[:, 0:1], where=mask_arr)
        c[np.where(np.logical_not(mask_arr))[0], 0] = dt / 2.0
        # middle integrals
        np.divide(
            np.exp(-sc * dt * (Nr + 1))
            - 2.0 * np.exp(-sc * dt * Nr)
            + np.exp(-sc * dt * (Nr - 1)),
            sc**2 * dt,
            out=c[:, 1:-1],
            where=mask_arr,
        )
        c[np.where(np.logical_not(mask_arr))[0], 1:-1] = dt
        # last frequency integral
        np.divide(
            np.exp(-sc * dt * N) - np.exp(-sc * dt * (N - 1)),
            -sc * dt,
            out=c[:, N - 1 : N],
            where=mask_arr,
        )
        np.divide(
            np.exp(-sc * dt * N) - c[:, N - 1 : N],
            -sc,
            out=c[:, N - 1 : N],
            where=mask_arr,
        )
        c[np.where(np.logical_not(mask_arr))[0], N - 1] = dt / 2.0
        self.c = c

    def _setQuadInv(self):
        t = self.t[:, np.newaxis] * 1e-3
        s = self.s[np.newaxis, :]
        ic = np.zeros((len(self.t), len(self.s)), dtype=complex)
        mask_arr = np.abs(t) > 1e-12
        # compute integrals
        I1 = np.divide(
            np.exp(s[:, 1:] * t) - np.exp(s[:, :-1] * t),
            1j * t,
            out=np.zeros_like(ic[:, :-1]),
            where=mask_arr,
        )
        I1[np.where(np.logical_not(mask_arr))[0], :] = (s[:, 1:] - s[:, :-1]) / 1j
        I2_ = np.divide(
            np.exp(s[:, 1:] * t) - np.exp(s[:, :-1] * t),
            1j * t,
            out=np.zeros_like(ic[:, :-1]),
            where=mask_arr,
        )
        # I2_[0,:] = s[:,1:] - s[:,:-1] / 1j
        I2 = np.divide(
            (s[:, 1:] - s[:, :-1]).imag * np.exp(s[:, 1:] * t) - I2_,
            1j * t,
            out=np.zeros_like(ic[:, :-1]),
            where=mask_arr,
        )
        # I2[0:1,:] = s[:,1:] * (s[:,1:] - s[:,:-1])
        # compute matrix elements
        ic[:, 0] = I1[:, 0] - I2[:, 0] / (s[:, 1] - s[:, 0]).imag
        ic[:, 1:-1] = (
            I1[:, 1:]
            - I2[:, 1:] / (s[:, 2:] - s[:, 1:-1]).imag
            + I2[:, :-1] / (s[:, 1:-1] - s[:, :-2]).imag
        )
        ic[:, -1] = I2[:, -1] / (s[:, -1] - s[:, -2]).imag
        self.ic = ic / (2.0 * np.pi)

    def __call__(self, arr):
        """
        Evaluate the Fourrier transform of `arr`

        Parameters
        ----------
            arr: `np.array`
                Should have the same length as `self.t`

        Returns
        -------
            s: `np.array`
                the frequency points at which the Fourrier
                transform is evaluated (in Hz)
            farr: `np.array`
                the Fourrier transform of `arr`
        """
        farr = np.dot(self.c, arr)
        return self.s, farr

    def ft(self, arr):
        """
        Evaluate the Fourrier transform of `arr`

        Parameters
        ----------
            arr: `np.array`
                Should have the same length as `self.t`

        Returns
        -------
            s: `np.array`
                the frequency points at which the Fourrier
                transform is evaluated (in Hz)
            farr: `np.array`
                the Fourrier transform of `arr`
        """
        return self(arr)

    def ft_inv(self, arr):
        """
        Evaluate the inverse Fourrier transform of `arr`

        Parameters
        ----------
            arr: `np.array`
                Should have the same length as `self.s`

        Returns
        -------
            t: `np.array`
                the time points at which the inverse Fourrier
                transform is evaluated (in ms)
            tarr: `np.array`
                the Fourrier transform of `arr`
        """
        tarr = np.dot(self.ic, arr)
        return self.t, tarr


class expExtractor(object):
    def __call__(self, N=1, recalc=False, atol=5e-2, pprint=True):
        """
        Fit a partial fraction decomposition to the obtained kernel in the frequency domain

        input:
            N (int): the number of exponenstials to use
            recalc (bool): whether to force a refit if a fit with the given N already exists
            atol (float): the tolerance, if the RMSE of the fit is higher
                and pprint is True a warning is printed
            pprint (bool): whether to print the warning

        output:
            kfit (dict): {'a': array of the exponental factors (1/tau), 'c': array of the
                coefficients, 'p': idicators whether or not a given exponential is part
                of a complex conjugate pair}
        """
        if N not in self.kfit or recalc:
            FEF = fExpFitter()
            ak, ck, pk, rms = FEF.fitFExp(
                self.s_f,
                self.k_f,
                rtol=1e-2,
                deg=N,
                maxiter=20,
                initpoles="log10",
                realpoles=True,
                zerostart=False,
                constrained=True,
                reduce_numexp=False,
            )
            ak *= 1e-3
            # ck *= 1e-3  # convert units to ms
            # also try time domain approach
            if N < 10:
                EF = ExpFitter()
                if "k_t" not in self.__dict__:
                    k_t = EF.sumExp(self.tarr, -self.kfit[30]["a"], self.kfit[30]["c"])
                else:
                    k_t = self.k_t
                ak_, ck_, rms_ = EF.PronyExpFit(N, self.tarr, k_t)
                ak_ = -ak_
                pk_ = np.zeros(ak_.shape, dtype=bool)
                # go for the approach with the lowest error
                rms = np.sqrt(
                    ((k_t - EF.sumExp(self.tarr, -ak, ck)) ** 2).sum() / len(k_t)
                )
                if rms > rms_:
                    ak = ak_
                    ck = ck_
                    pk = pk_
                    rms = rms_
            if rms > atol and pprint:
                print("No sane fit achieved for N=" + str(N))
                print("RMSE:", rms)
            self.kfit[N] = {"a": ak, "c": ck, "p": pk}
        return self.kfit[N]

    def fit_vector(self, N, atol=5e-2, pprint=True, store=True):
        FEF = fExpFitter()
        ak, ck, pk, rms = FEF.fitFExp(
            self.s_f,
            self.k_f,
            rtol=1e-2,
            deg=N,
            maxiter=20,
            initpoles="log10",
            realpoles=True,
            zerostart=False,
            constrained=True,
            reduce_numexp=False,
        )
        ak *= 1e-3
        # ck *= 1e-3  # convert units to ms
        if rms > atol and pprint:
            print("No sane fit achieved for N=" + str(N))
            print("RMSE:", rms)
        res = {"a": ak, "c": ck, "p": pk}
        if store:
            self.kfit[N] = res
        return res

    def fit_prony(self, N, atol=5e-2, pprint=True, store=True):
        EF = ExpFitter()
        if "k_t" not in self.__dict__:
            k_t = EF.sumExp(self.tarr, -self.kfit[30]["a"], self.kfit[30]["c"])
        else:
            k_t = self.k_t
        ak, ck, rms = EF.PronyExpFit(N, self.tarr, k_t)
        ak *= -1.0
        # ck *= 1e-3
        pk = np.zeros(ak.shape, dtype=bool)
        if rms > atol and pprint:
            print("No sane fit achieved for N=" + str(N))
            print("RMSE:", rms)
        res = {"a": ak, "c": ck, "p": pk}
        if store:
            self.kfit[N] = res
        return res

    def k_freq(self, N):
        self(N)
        FEF = fExpFitter()
        return FEF.sumFExp(self.s_f, self.kfit[N]["a"] * 1e3, self.kfit[N]["c"])

    def k_time(self, N):
        self(N)
        EF = ExpFitter()
        return EF.sumExp(self.tarr, -self.kfit[N]["a"], self.kfit[N]["c"])


class IzExtractor(expExtractor):
    def __init__(self, Iz, yarr):
        self.tarr = Iz
        FT = FourierQuadrature(self.tarr)
        self.s_f, self.k_f = FT(yarr)
        # for storing kernel fit results
        self.kfit = {}
        # initial fit
        self(30)


class simpleExpExtractor(expExtractor):
    def __init__(self, arr, tarr):
        dt = tarr[1] - tarr[0]
        self.tarr = tarr
        self.k_t = arr
        # create a smoothing window if arr hasn't gone to zero yet
        if arr[-1] > 1e-9:
            vwindow = np.cos((np.pi / 2.0) * (self.tarr / self.tarr[-1]))
        else:
            vwindow = np.ones(self.tarr.shape)
        # compute Fourrier transform
        FT = FourierQuadrature(
            self.tarr, fmax=np.log10(1.0 / (dt * 1e-3)), base=10.0, num=200
        )
        self.s_f, self.k_f = FT(self.k_t)
        # for storing kernel fit results
        self.kfit = {}
        # initial fit
        self(30)


class kernelExtractor(expExtractor):
    """
    This class computes the rescale kernel between the
    synapse in the dendrite and the synapse in the soma.

    input:
        vdend (numpy 1d array): the somatic voltage recording
            with the synapse in the dendrites
        vsoma (numpy 1d array): the somatic voltage recording
            with the synapse at the some
        vbase (numpy 1d array): the somatic voltage recording
            without the synapse
        time  (numpy 1d array): the time array associated with
            the recordings

    attributes:
        vdend: somatic voltage from dendritic spike arrival onwards
        vsoma: somatic voltage from somatic spike arrival onwards
        vbase: somatic voltage without spike arrival
        vdend_: vdend - vbase
        vsoma_: vsoma - vbase
        s_f: frequency array
        vdend_f: fourier transform of vdend_
        vdend_f: fourier transform of vdend_
        k_f: kernel in the frequency domain
    """

    def __init__(self, vdend, vsoma, vbase, time):
        # time step
        dt = time[1] - time[0]
        # voltage deviations from baseline
        vdend_ = vdend - vbase
        vsoma_ = vsoma - vbase
        # find when the spike arrives
        istart = np.where(vdend_ > 1e-9)[0][0] - 1.0
        # time array PSP
        self.tarr = time[istart:] - istart * dt
        # full potentials after spike arrival
        self.vdend = vdend[istart:]
        self.vsoma = vsoma[istart:]
        self.vbase = vbase[istart:]
        # create a smoothing window if PSP hasn't gone to zero yet
        if vdend_[-1] > 1e-9:
            vwindow = np.cos((np.pi / 2.0) * (self.tarr / self.tarr[-1]))
        else:
            vwindow = np.ones(self.tarr.shape)
        # PSPs
        self.vdend_ = vdend_[istart:] * vwindow
        self.vsoma_ = vsoma_[istart:] * vwindow
        # fourrier transform
        FT = FourierQuadrature(
            self.tarr, fmax=np.log10(1.0 / (dt * 1e-3)), base=10.0, num=200
        )
        self.s_f, self.vdend_f = FT(self.vdend_)
        self.s_f, self.vsoma_f = FT(self.vsoma_)
        # compute the kernel
        self.k_f = self.vdend_f / self.vsoma_f
        # for storing kernel fit results
        self.kfit = {}
        # initial fit
        self(30)


class expNReducer(expExtractor):
    def __init__(self, alphas, gammas):
        FEF = fExpFitter()
        # creat time array
        self.tarr = np.linspace(0.0, 100.0, 1e3)
        # create frequency array
        self.s_f = create_logspace_freqarray(fmax=7, base=10, num=200)
        # kernel in freqency domain
        self.k_f = FEF.sumFExp(self.s_f, -alphas * 1e3, gammas * 1e3)
        # for storing kernel fit results
        self.kfit = {}
        # initial fit
        self(30)


class FourierTools:
    def __init__(self, t_inp):
        self._set_freq_and_time_arrays(t_inp)

    def _set_default_freq_array_vector_fit(self):
        # reasonable parameters to construct frequency array
        dt = 0.1 * 1e-3  # s
        N = 2**12
        smax = np.pi / dt  # Hz
        ds = np.pi / (N * dt)  # Hz

        # frequency array for vector fitting
        self.freqs_vfit = np.arange(-smax, smax, ds) * 1j  # Hz

    def _set_default_freq_array_quadrature(self, t_inp):
        self.t = t_inp
        if isinstance(t_inp, FourierQuadrature):
            self.t = t_inp.t
        # reasonable parameters for FourierQuadrature
        self.fq = FourierQuadrature(self.t, fmax=7.0, base=10.0, num=200)

    def _set_freq_and_time_arrays(self, t_inp):
        self._set_default_freq_array_vector_fit()
        self._set_default_freq_array_quadrature(t_inp)

        self._slice_vfit = np.s_[: len(self.freqs_vfit)]
        self._slice_quad = np.s_[len(self.freqs_vfit) :]
        self.freqs = np.concatenate((self.freqs_vfit, self.fq.s))

    def inverse_fourier(
        self,
        func_vals_f,
        method: Literal["", "exp fit", "quadrature"] = "",
        compute_time_derivative=True,
    ):
        if method not in ["", "exp fit", "quadrature"]:
            raise IOError("Method should be empty string, 'exp fit' or 'quadrature'")

        # compute in time domain, method depends on ratio between spectral
        # power in zero frequency vs max frequency component
        # typically, this will mean exponential fit is chosen for input
        # impedances and explicit quadrature for transfer impedances
        f_arr = func_vals_f[self._slice_quad]
        criterion_eval = np.abs(f_arr[-1]) / np.abs(f_arr[self.fq.ind_0s])
        criterion = criterion_eval <= 1e-3

        if criterion_eval > 1e-10:
            # if there is substantial spectral power in the max frequency
            # components, we smooth the function with a squared cosine window
            # to reduce oscillations
            window = np.cos(np.pi * self.fq.s.imag / (2.0 * np.abs(self.fq.s[-1]))) ** 2
        else:
            window = np.ones_like(self.fq.s)

        # compute kernel through quadrature method
        func_vals_t = (
            self.fq.ft_inv(window * func_vals_f[self._slice_quad])[1].real * 1e-3
        )  # MOhm/s -> MOhm/ms
        if compute_time_derivative:
            # compute differentiated kernel
            dfunc_vals_t_dt = (
                self.fq.ft_inv(self.fq.s * window * func_vals_f[self._slice_quad])[
                    1
                ].real
                * 1e-6
            )  # MOhm/s^2 -> MOhm/ms^2

        # when the criterion is satified, or if the default method is
        # overridden to 'quadrature', we always return the the quadrature result
        if (method == "" and criterion) or method == "quadrature":
            if compute_time_derivative:
                return func_vals_t, dfunc_vals_t_dt
            else:
                return func_vals_t

        # this code will only be reached when `method` is "exp_fit", or when
        # `method` is "" but the criterion is not satisfied

        # we set a custom set of initial poles for the vector fit algorithm
        # NOTE: not used at the moment, have to figure out why
        # initpoles = np.concatenate((
        #     np.linspace(.5, 10**1.3, 40)[:-1],
        #     np.logspace(
        #         1.3, np.log10(self.freqs[self._slice_vfit][-1].imag),
        #         num=40, base=10,
        #     )
        # ))

        # compute kernel as superposition of exponentials in the frequency domain
        f_exp_fitter = fExpFitter()
        alpha, gamma, pairs, rms = f_exp_fitter.fitFExp(
            self.freqs[self._slice_vfit],
            func_vals_f[self._slice_vfit],
            deg=40,
            initpoles="log10",
            realpoles=True,
            zerostart=False,
            constrained=True,
            reduce_numexp=False,
        )

        zk = Kernel({"a": alpha * 1e-3, "c": gamma * 1e-3})
        if compute_time_derivative:
            dzk_dt = zk.diff()
        # linear fit of c in the time domain to the quadrature-computed kernels
        # can improve accuracy
        # NOTE: not used at the moment, have to figure out why
        # w = np.concatenate(
        #     (self.fq.t[self.fq.t < 1.], np.ones_like(self.fq.t[self.fq.t >= 1.]))
        # )
        # zk.fit_c(self.fq.t, func_vals_t, w=w)

        # evaluate kernel in the time domain
        func_vals_t = zk(self.fq.t)

        if compute_time_derivative:
            # linear fit of c in the time domain to the quadrature-computed kernels
            # can improve accuracy
            # NOTE: not used at the moment, have to figure out why
            # dzk_dt.fit_c(self.fq.t, dfunc_vals_t_dt, w=w)
            # compute differentiated kernel
            dfunc_vals_t_dt = zk.diff(self.fq.t)

            return func_vals_t, dfunc_vals_t_dt

        else:
            return func_vals_t
