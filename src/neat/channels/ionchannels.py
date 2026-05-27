# -*- coding: utf-8 -*-
#
# ionchannels.py
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

import sympy as sp
import numpy as np

import os
import ast
import warnings

from ..factorydefaults import DefaultPhysiology

# CONC_DICT = {
#     'na': 10.,  # mM
#     'k': 54.4,  # mM
#     'ca': 1e-4,  # mM
# }

# TEMP_DEFAULT = 36.

# E_ION_DICT = {
#     'na': 50.,
#     'k': -85.,
#     'ca': 50.,
# }


class IfExpVisitor(ast.NodeVisitor):
    """
    Returns the first `IfExp` node in the ast, signalling an if statement
    """

    def __init__(self):
        self.ifexp_node = None

    def visit_IfExp(self, node):
        """
        Function can NOT be renamed, is implicitly called by `ast.NodeVisitor.visit()`
        """
        self.ifexp_node = node

    def find_IfExp_node(self, node):
        self.visit(node)
        return_node = self.ifexp_node
        self.ifexp_node = None
        return return_node


class _func(object):
    def __init__(self, eval_func_aux, eval_func_vtrap, e_trap):
        self.eval_func_aux = eval_func_aux
        self.eval_func_vtrap = eval_func_vtrap
        self.e_trap = e_trap

    def __call__(self, *args):
        vv = args[0]
        if isinstance(vv, float):
            if np.abs(vv - self.e_trap) < 0.001:
                return self.eval_func_vtrap(*args)
            else:
                return self.eval_func_aux(*args)
        else:
            fv_return = np.zeros_like(vv)
            bool_vtrap = np.abs(vv - self.e_trap) < 0.0001
            inds_vtrap = np.where(bool_vtrap)
            args_ = [a[inds_vtrap] for a in args]
            fv_return[inds_vtrap] = self.eval_func_vtrap(*args_)
            inds = np.where(np.logical_not(bool_vtrap))
            args_ = [a[inds] for a in args]
            fv_return[inds] = self.eval_func_aux(*args_)
            return fv_return


def _insert_function_prefixes(
    string, prefix="np", functions=["exp", "sin", "cos", "tan", "pi"]
):
    """
    Prefix all occurences in the input `string` of the functions in the
    `functions` list with the provided `prefix`.

    Parameters
    ----------
    string: string
        the input string
    prefix: string, optional
        the prefix that is put before each function. Defaults to `'np'`
    functions: list of strings, optional
        the list of functions that will be prefixed. Defaults to
        `['exp', 'sin', 'cos', 'tan', 'pi']`

    Returns
    -------
    string

    Examples
    --------
    >>> _insert_function_prefixes('5. * exp(0.) + 3. * cos(pi)')
    '5. * np.exp(0.) + 3. * np.cos(pi)'
    """
    for func_name in functions:
        numpy_string = ""
        while len(string) > 0:
            ind = string.find(func_name)
            if ind == -1:
                numpy_string += string
                string = ""
            else:
                numpy_string += string[0:ind] + prefix + "." + func_name
                string = string[ind + len(func_name) :]
        string = numpy_string
    return string


def _broadcast(fun):
    """
    This function is to be used in together with `sympy.lambdify` to ensure that
    lambda functions generated from constant expressions are broadcast to the
    input shape
    """
    return lambda *x: np.broadcast_arrays(fun(*x), *x)[0]


class SPDict(dict):
    """
    Dictionary that accepts both strings and similarly name sympy symbols as keys
    """

    def __getitem__(self, key):
        try:
            return super(SPDict, self).__getitem__(key)
        except KeyError:
            if isinstance(key, sp.Symbol):
                return super(SPDict, self).__getitem__(str(key))
            else:
                return super(SPDict, self).__getitem__(sp.symbols(key))

    def __contains__(self, key):
        return super().__contains__(key) or super().__contains__(sp.symbols(key))


class CallDict(SPDict):
    """
    Callable dictionary, items are supposed to be callables
    that all accept an identical argument list
    """

    def __call__(self, *args):
        """
        Calls dictionary items (supposed to be callable)
        """
        return SPDict({str(k): f(*args) for k, f in self.items()})


def _safe_simplify(expr):
    if expr.has(sp.Piecewise):
        return expr
    else:
        return sp.simplify(expr)


def _nmodl_ccode(expr):
    return " ".join(sp.printing.ccode(expr).replace(";", "").split())


def _rebuild_sympy_expr(expr, args):
    try:
        return expr.func(*args, evaluate=False)
    except TypeError:
        return expr.func(*args)


def _nmodl_boolean_assignment_lines(lhs, expr, temp_counter=0):
    expr = sp.sympify(expr)

    if expr == True or expr == sp.true:
        return [f"{lhs} = 1"], temp_counter, []
    if expr == False or expr == sp.false:
        return [f"{lhs} = 0"], temp_counter, []

    lines, expr, temp_counter, locals_ = _hoist_piecewise_for_nmodl(expr, temp_counter)
    if getattr(expr, "is_Boolean", False):
        lines += [
            f"if ({_nmodl_ccode(expr)}) {{",
            f"    {lhs} = 1",
            "}",
            "else {",
            f"    {lhs} = 0",
            "}",
        ]
    else:
        lines.append(f"{lhs} = {_nmodl_ccode(expr)}")
    return lines, temp_counter, locals_


def _hoist_piecewise_for_nmodl(expr, temp_counter=0):
    expr = sp.sympify(expr)

    if expr.func == sp.ITE:
        temp = sp.symbols(f"pw{temp_counter}")
        temp_counter += 1
        test, then_value, else_value = expr.args
        lines, test_expr, temp_counter, locals_ = _hoist_piecewise_for_nmodl(
            test, temp_counter
        )
        then_lines, temp_counter, then_locals = _nmodl_boolean_assignment_lines(
            temp, then_value, temp_counter
        )
        else_lines, temp_counter, else_locals = _nmodl_boolean_assignment_lines(
            temp, else_value, temp_counter
        )
        locals_ = [str(temp)] + locals_ + then_locals + else_locals
        lines += [f"if ({_nmodl_ccode(test_expr)}) {{"]
        lines.extend("    " + line for line in then_lines)
        lines += ["}", "else {"]
        lines.extend("    " + line for line in else_lines)
        lines.append("}")
        return lines, temp, temp_counter, locals_

    if isinstance(expr, sp.Piecewise):
        temp = sp.symbols(f"pw{temp_counter}")
        temp_counter += 1
        lines = []
        locals_ = [str(temp)]
        branch_specs = []

        for value, condition in expr.args:
            condition_lines, condition_expr, temp_counter, condition_locals = (
                _hoist_piecewise_for_nmodl(condition, temp_counter)
            )
            value_lines, value_expr, temp_counter, value_locals = (
                _hoist_piecewise_for_nmodl(value, temp_counter)
            )
            lines.extend(condition_lines)
            locals_.extend(condition_locals)
            branch_specs.append((condition_expr, value_lines, value_expr, value_locals))

        for branch_index, (condition, value_lines, value_expr, value_locals) in enumerate(branch_specs):
            locals_.extend(value_locals)
            if condition == True or condition == sp.true:
                lines.append("else {")
            else:
                condition_code = _nmodl_ccode(condition)
                if branch_index == 0:
                    lines.append(f"if ({condition_code}) {{")
                else:
                    lines.append(f"else if ({condition_code}) {{")
            lines.extend("    " + line for line in value_lines)
            lines.append(f"    {temp} = {_nmodl_ccode(value_expr)}")
            lines.append("}")

        if lines and lines[0] == "else {":
            lines = [f"{temp} = {_nmodl_ccode(branch_specs[0][2])}"]

        return lines, temp, temp_counter, locals_

    if not expr.args:
        return [], expr, temp_counter, []

    lines = []
    locals_ = []
    new_args = []
    for arg in expr.args:
        arg_lines, arg_expr, temp_counter, arg_locals = _hoist_piecewise_for_nmodl(
            arg, temp_counter
        )
        lines.extend(arg_lines)
        locals_.extend(arg_locals)
        new_args.append(arg_expr)

    if any(new_arg != old_arg for new_arg, old_arg in zip(new_args, expr.args)):
        expr = _rebuild_sympy_expr(expr, new_args)

    return lines, expr, temp_counter, locals_


def _nmodl_assignment_lines(lhs, expr, temp_counter=0):
    lines, expr, temp_counter, locals_ = _hoist_piecewise_for_nmodl(expr, temp_counter)
    lines.append(f"{lhs} = {_nmodl_ccode(expr)}")
    return lines, temp_counter, locals_


class IonChannel(object):
    """
    Base ion channel class that implements linearization and code generation for
    NEURON (.mod-files) and C++.

    Userdefined ion channels should inherit from this class and implement the
    `define()` function, where the specific attributes of the ion channel are set.

    The ion channel current is of the form

    .. math:: i_{chan} = \overline{g} \, p_o(x_1, ... , x_n) \, (e - v)

    where $p_o$ is the open probability defined as a function of a number of
    state variables. State variables evolve according to

    .. math:: \dot{x}_i = f_i(x_i, v, c_1, ..., c_k)

    with $c_1, ..., c_n$ the (optional) set of concentrations the ion channel
    depends on. There are two canonical ways to define $f_i$, either based on
    reaction rates :math:`\\alpha` and :math:`\\beta`:

    .. math:: \dot{x}_i = \\alpha_i(v) \, (1 - x_i) - \\beta_i(v) \, x_i,

    or based on an asymptotic value :math:`x_i^{\infty}`  and time-scale :math:`\\tau_i`

    .. math:: \dot{x}_i = \\frac{x_i^{\infty}(v) - x_i}{\\tau_i(v)}.

    `IonChannel` accepts handles either description. For the former description,
    dicts `self.alpha` and `self.beta` must be defined with as keys the names
    of every state variable in the open probability. Similarly, for the latter
    description, dicts `self.tauinf` and `self.varinf` must be defined with as
    keys the name of every state variable.

    The user **must** define the attributes `p_open`, and either `alpha` and
    `beta` or `tauinf` and `varinf` in the `define()` function. The other
    attributes `ion`, `conc`, `q10`, `temp`, and `e` are optional.

    Parameters
    ----------
    p_open: str
        The open probability of the ion channel.
    alpha, beta: dict {str: str}
        dictionary of the rate function for each state variables. Keys must
        correspond to the name of every state variable in `p_open`, values must
        be formulas written as strings with `v` and possible ion as variabels
    tauinf, varinf: dict {str: str}
        state variable time scale and asymptotic activation level. Keys must
        correspond to the name of every state variable in `p_open`, values must
        be formulas written as strings with `v` and possible ion as variabels

    ion: str ('na', 'ca', 'k' or ''), optional
        The ion to which the ion channel is permeable
    conc: set of str (containing 'na', 'ca', 'k') or dict of {str: float}
        The concentrations the ion channel activation depends on. Can be a set
        of ions or a dict with the ions as keys and default values as float.
    q10: str, optional
        Temperature dependence of the state variable rate functions. May be a
        float or a string convertible to a sympy expression containing the
        `temp` parameter (temperature in ``[deg C]``). This factor divides the
        time-scales :math:`\tau_i(v)` of the ion channel. If not given, default
        is 1.
    temp: float, optional
        The temperature at which the ion channel is evaluated. Can be modified
        after initializiation by calling
        `IonChannel.set_default_params(temp=new_temperature)`. If not given, the
        evaluates `self.q10` at the default temperature of 36 degC.
    e: float, optional
        Reversal of the ion channel in ``[mV]``. functions that need it allow
        the default value to be overwritten with a keyword argument. If nothing
        is provided, will take a default reversal for `self.ion` (which is
        -85 mV for 'K', 50 mV for 'Na' and 50 mV for 'Ca'). If no ion is
        provided, errors will occur if functions that need `e` are called
        without specifying the value as a keyword argument.

    Examples
    --------
    >>> class Na_Ta(IonChannel):
    >>>     def define(self):
    >>>         # from (Colbert and Pan, 2002), Used in (Hay, 2011)
    >>>         self.ion = 'na'
    >>>         # concentrations the ion channel depends on
    >>>         self.conc = {}
    >>>         # define channel open probability
    >>>         self.p_open = 'h * m ** 3'
    >>>         # define activation functions
    >>>         self.alpha, self.beta = {}, {}
    >>>         self.alpha['m'] =  '0.182 * (v + 38.) / (1. - exp(-(v + 38.) / 6.))' # 1/ms
    >>>         self.beta['m']  = '-0.124 * (v + 38.) / (1. - exp( (v + 38.) / 6.))' # 1/ms
    >>>         self.alpha['h'] = '-0.015 * (v + 66.) / (1. - exp( (v + 66.) / 6.))' # 1/ms
    >>>         self.beta['h']  =  '0.015 * (v + 66.) / (1. - exp(-(v + 66.) / 6.))' # 1/ms
    >>>         # temperature factor for reaction rates
    >>>         self.q10 = '2.3^((temp - 23.)/10.)'
    """

    def __init__(self, **kwargs):
        """
        Will give an ``AttributeError`` if initialized as is. Should only be
        initialized from its' derived classes that implement specific ion
        channel types.
        """
        # initialize default configuration
        self.cfg = DefaultPhysiology()

        # define the channel based on user specified state variables and activations
        self.define()

        # ion that carries the channel current
        if not hasattr(self, "ion"):
            self.ion = ""

        # temperature factor, if it exist
        if not hasattr(self, "q10"):
            self.q10 = "1."
        self.q10 = sp.sympify(self.q10, evaluate=False)

        # sympy temperature symbols
        assert len(self.q10.free_symbols) <= 1
        if len(self.q10.free_symbols) > 0:
            assert str(list(self.q10.free_symbols)[0]) == "temp"
            self.sp_t = list(self.q10.free_symbols)[0]
        else:
            self.sp_t = sp.symbols("temp")

        # the voltage variable
        self.sp_v = sp.symbols("v")

        # extract the state variables
        self.p_open = sp.sympify(self.p_open)
        self.statevars = self.p_open.free_symbols
        # if voltage occurs directly in open probability,
        # remove it from statevars
        if self.sp_v in self.statevars:
            self.statevars.remove(self.sp_v)

        if not "tauinf" in self.__dict__:
            self.tauinf = {}
        if not "varinf" in self.__dict__:
            self.varinf = {}

        for svar in self.ordered_statevars:
            key = str(svar)
            if key in (self.varinf.keys() | self.tauinf.keys()):
                self.varinf[svar] = sp.sympify(self.varinf[key], evaluate=False)
                self.tauinf[svar] = sp.sympify(self.tauinf[key], evaluate=False)
                self.varinf[svar] = _safe_simplify(self.varinf[svar])
                self.tauinf[svar] = _safe_simplify(self.tauinf[svar] / self.q10)
                del self.varinf[key]
                del self.tauinf[key]

        # construct the rate functions
        if "alpha" in self.__dict__ and "beta" in self.__dict__:
            for svar in self.ordered_statevars:
                key = str(svar)
                if key in (self.alpha.keys() | self.beta.keys()):
                    self.alpha[svar] = sp.sympify(self.alpha[key], evaluate=False)
                    self.beta[svar] = sp.sympify(self.beta[key], evaluate=False)
                    self.varinf[svar] = sp.simplify(
                        self.alpha[svar] / (self.alpha[svar] + self.beta[svar])
                    )
                    self.tauinf[svar] = sp.simplify(
                        (1.0 / self.q10) / (self.alpha[svar] + self.beta[svar])
                    )
            del self.alpha
            del self.beta

        # check if rate equations where defined
        if len(self.varinf) == 0 or len(self.tauinf) == 0:
            raise AttributeError(
                "Necessary attributes not defined, define either "
                + "`alpha` and `beta` or `tauinf` and `varinf`."
            )

        self.varinf, self.tauinf = SPDict(self.varinf), SPDict(self.tauinf)

        # set the right hand side of the differential equation for
        # state variables
        self.fstatevar = SPDict()
        for svar in self.ordered_statevars:
            self.fstatevar[svar] = (-svar + self.varinf[svar]) / self.tauinf[svar]

        # concentrations the ion channel depends on
        if not hasattr(self, "conc"):
            # if concentration ions are not defined, attempt to extract them from
            # the state variable functions
            self.conc = set()
            for key, expr in self.fstatevar.items():
                self.conc |= expr.free_symbols  # set union
            # remove everything that is not a concentration
            self.conc -= self.statevars
            self.conc -= {self.sp_v, self.sp_t}
        # if no default concentrations are defined, default values are taken
        # from default concentration values
        if not hasattr(self.conc, "values"):
            self.conc = SPDict(
                {
                    sp.symbols(str(ion)): self.cfg.conc[str(ion)]
                    for ion in list(sorted(self.conc))
                }
            )
        # sympy concentration symbols
        self.sp_c = [ion for ion in self.conc]

        # default parameters
        self.default_params = SPDict({})
        self.default_params[str(self.sp_t)] = (
            self.temp if "temp" in self.__dict__ else self.cfg.temp
        )
        try:
            self.default_params["e"] = (
                self.e if "e" in self.__dict__ else self.cfg.e_rev[self.ion]
            )
        except KeyError:
            warnings.warn("No default reversal potential defined.")

        # self._lambdify_channel()
        self.set_default_params(**kwargs)

    def __getstate__(self):
        """
        remove lambdified functions from dict as they can not be pickled
        """
        d = dict(self.__dict__)

        del d["f_statevar"]
        del d["f_varinf"]
        del d["f_tauinf"]
        del d["f_p_open"]
        del d["dp_dx"], d["df_dv"], d["df_dx"], d["df_dc"]

        return d

    def __setstate__(self, s):
        """
        since lambdified functions were not pickled we need to restore them
        """
        self.__dict__ = s
        self._lambdify_channel()

    def set_default_params(self, **kwargs):
        """
        **kwargs
            Default values for temperature (`temp`), reversal (`e`)
        """
        self.default_params.update(kwargs)
        # set the lambda functions for efficient numpy evaluation
        self._lambdify_channel()

    def _substitute_defaults(self, expr):
        """
        Substitute default values in input expression

        Parameters
        ----------
        expr: sympy expression
        """
        for param, val in self.default_params.items():
            expr = expr.subs(sp.symbols(param), val)
        return expr

    @property
    def ordered_statevars(self):
        return list(sorted(self.statevars, key=str))

    def _lambdify_channel(self):
        """
        Create lambda functions based on sympy expression for relevant ion
        channel functions
        """
        from sympy.utilities.autowrap import ufuncify

        # arguments for lambda function
        args = [self.sp_v] + self.ordered_statevars + self.sp_c
        args_ = [self.sp_v] + self.sp_c

        # lambdified open probability
        self.f_p_open = _broadcast(sp.lambdify(args, self.p_open))
        # storatestate variable function
        self.f_statevar = CallDict()
        self.f_varinf, self.f_tauinf = CallDict(), CallDict()
        # storage of derivatives
        self.dp_dx = CallDict()
        self.df_dv, self.df_dx, self.df_dc = CallDict(), CallDict(), CallDict()

        for svar, f_svar in self.fstatevar.items():
            f_svar = self._substitute_defaults(f_svar)
            varinf = self._substitute_defaults(self.varinf[svar])
            tauinf = self._substitute_defaults(self.tauinf[svar])

            # state variable function
            self.f_statevar = _broadcast(sp.lambdify(args, f_svar))

            # state variable activation & timescale
            self.f_varinf[svar] = _broadcast(sp.lambdify(args_, varinf))
            self.f_tauinf[svar] = _broadcast(sp.lambdify(args_, tauinf))

            # derivatives of open probability to state variables
            self.dp_dx[svar] = _broadcast(
                sp.lambdify(args, sp.diff(self.p_open, svar, 1))
            )

            # derivatives of state variable function to voltage
            self.df_dv[svar] = _broadcast(
                sp.lambdify(args, sp.diff(f_svar, self.sp_v, 1))
            )

            # derivatives of state variable function to state variable
            self.df_dx[svar] = _broadcast(sp.lambdify(args, sp.diff(f_svar, svar, 1)))

            # derivatives of state variable function to concentrations
            self.df_dc[svar] = CallDict(
                {
                    c: _broadcast(sp.lambdify(args, sp.diff(f_svar, c, 1)))
                    for c in self.sp_c
                }
            )

    def _args_as_list(self, v, w_statevar=True, **kwargs):
        """
        Converts arguments to list for lambdified functions
        """
        arg_list = [v]

        if w_statevar:
            for svar in self.ordered_statevars:
                key = str(svar)
                try:
                    arg_list.append(kwargs[key])
                except KeyError:
                    # state variable is not in kwargs
                    # set default value based on voltage
                    args = self._args_as_list(v, w_statevar=False, **kwargs)
                    arg_list.append(self.f_varinf[svar](*args))

        for c in self.sp_c:
            key = str(c)
            try:
                arg_list.append(kwargs[key])
            except KeyError:
                # ion is not in kwargs
                # set stored default value
                arg_list.append(self.conc[c])

        return arg_list

    def compute_p_open(self, v, **kwargs):
        """
        Compute the open probability of the ion channel

        Parameters
        ----------
        v: float or `np.ndarray` of float
            The voltage at which to evaluate the open probability
        **kwargs: float or `np.ndarray`
            Optional values for the state variables and concentrations.

        Returns
        -------
        float or `np.ndarray` of float
            The open probability
        """
        args = self._args_as_list(v, **kwargs)
        return self.f_p_open(*args)

    def compute_derivatives(self, v, **kwargs):
        """
        Compute:
        (i) the derivatives of the open probability to the state variables
        (ii) The derivatives of state functions to the voltage
        (iii) The derivatives of state functions to the state variables

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage at which to evaluate the open probability
        **kwargs: float or `np.ndarray`
            Optional values for the state variables and concentrations.

        Returns
        -------
        tuple of three floats or three `np.ndarray`s of float
            The derivatives
        """
        args = self._args_as_list(v, **kwargs)
        return self.dp_dx(*args), self.df_dv(*args), self.df_dx(*args)

    def compute_derivativesConc(self, v, **kwargs):
        """
        Compute the derivatives of the state functions to the concentrations

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage at which to evaluate the open probability
        **kwargs: float or `np.ndarray`
            Optional values for the state variables and concentrations.

        Returns
        -------
        tuple of three floats or three `np.ndarray`s of float
            The derivatives
        """
        args = self._args_as_list(v, **kwargs)
        return self.df_dc(*args)

    def compute_varinf(self, v):
        """
        Compute the asymptotic values for the state variables at a given
        activation level

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage at which to evaluate the open probability

        Returns
        -------
        dict of `np.ndarray` of dict of float
            The asymptotic activations, items are of same type (and shape) as `v`
        """
        args = self._args_as_list(v, w_statevar=False, **{})
        return self.f_varinf(*args)

    def compute_tauinf(self, v):
        """
        Compute the time-scales for the state variables at a given
        activation level

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage at which to evaluate the open probability

        Returns
        -------
        dict of `np.ndarray` of dict of float
            The asymptotic activations, items are of same type (and shape) as `v`
        """
        args = self._args_as_list(v, w_statevar=False, **{})
        return self.f_tauinf(*args)

    def compute_lin_statevar_response(self, v, freqs, v_resp, **kwargs):
        """
        Combute the linearizations of the individual state variables

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage(s) ``[mV]`` around which to linearize the ion channel
        freqs float, complex, or `np.ndarray` of float or complex:
            The frequencies ``[Hz]`` at which to evaluate the linearized contribution
        v_resp: `np.ndarray` (``dtype=complex``, ``ndim=1``, ``shape=(s,k)``)
            Linearized voltage responses in the frequency domain, evaluated at
            ``s`` frequencies and ``k`` locations
        **kwargs: float or `np.ndarray`
            Optional values for the state variables and concentrations.

        Returns
        -------
        `SPDict` of float, complex or `np.ndarray` of float or complex
            The linearized current. Key are the state variable name. Shape of
            each entry is dimension of `freqs` followed by the dimensions of `v`.
        """
        dp_dx, df_dv, df_dx = self.compute_derivatives(v, **kwargs)

        # determine the output shape according to numpy broadcasting rules
        args_aux = [v_resp] + self._args_as_list(v, **kwargs)
        out_shape = np.broadcast(*args_aux).shape

        lin_svar = SPDict(
            {
                str(svar): np.zeros(out_shape, dtype=np.array(freqs).dtype)
                for svar in self.ordered_statevars
            }
        )
        for svar, dp_dx_ in dp_dx.items():
            df_dv_ = df_dv[svar] * 1e3  # convert to 1 / s
            df_dx_ = df_dx[svar] * 1e3  # convert to 1 / s
            # add to the impedance contribution
            lin_svar[str(svar)] = df_dv_ / (freqs - df_dx_) * v_resp
        return lin_svar

    def compute_linear(self, v, freqs, **kwargs):
        """
        Combute the contributions of the state variables to the linearized
        channel current

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage ``[mV]`` at which to evaluate the open probability
        freqs float, complex, or `np.ndarray` of float or complex:
            The frequencies ``[Hz]`` at which to evaluate the linearized contribution
        **kwargs: float or `np.ndarray`
            Optional values for the state variables and concentrations.

        Returns
        -------
        float, complex or `np.ndarray` of float or complex
            The linearized current. Shape is dimension of `freqs` followed by
            the dimensions of `v`.
        """
        dp_dx, df_dv, df_dx = self.compute_derivatives(v, **kwargs)

        # determine the output shape according to numpy broadcasting rules
        args_aux = [freqs] + self._args_as_list(v, **kwargs)
        out_shape = np.broadcast(*args_aux).shape

        lin_f = np.zeros(out_shape, dtype=np.array(freqs).dtype)
        for svar, dp_dx_ in dp_dx.items():
            df_dv_ = df_dv[svar] * 1e3  # convert to 1 / s
            df_dx_ = df_dx[svar] * 1e3  # convert to 1 / s
            # add to the impedance contribution
            lin_f += dp_dx_ * df_dv_ / (freqs - df_dx_)
        return lin_f

    def compute_linear_conc(self, v, freqs, ion, **kwargs):
        """
        Combute the contributions of the state variables to the linearized
        channel current

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage ``[mV]`` at which to evaluate the open probability
        freqs: float, complex, or `np.ndarray` of float or complex:
            The frequencies ``[Hz]`` at which to evaluate the linearized contribution
        ion: str
            The ion name for which to compute the linearized contribution
        **kwargs: float or `np.ndarray`
            Optional values for the state variables and concentrations.

        Returns
        -------
        float, complex or `np.ndarray` of float or complex
            The linearized current. Shape is dimension of `freqs` followed by
            the dimensions of `v`.
        """
        dp_dx, df_dv, df_dx = self.compute_derivatives(v, **kwargs)
        df_dc = self.compute_derivativesConc(v, **kwargs)

        # determine the output shape according to numpy broadcasting rules
        args_aux = [freqs] + self._args_as_list(v, **kwargs)
        out_shape = np.broadcast(*args_aux).shape

        lin_f = np.zeros(out_shape, dtype=np.array(freqs).dtype)
        for svar, dp_dx_ in dp_dx.items():
            df_dc_ = df_dc[svar][ion] * 1e3  # convert to 1 / s
            df_dx_ = df_dx[svar] * 1e3  # convert to 1 / s
            # add to the impedance contribution
            lin_f += dp_dx_ * df_dc_ / (freqs - df_dx_)
        return lin_f

    def _get_reversal(self, e):
        if e is None:
            try:
                e = self.default_params["e"]
            except KeyError:
                raise KeyError("No default reversal defined, provide value for `e`.")
        return e

    def compute_lin_sum(self, v, freqs, e=None, **kwargs):
        """
        Combute the linearized channel current contribution
        (without concentributions from the concentration - see `compute_lin_conc()`)

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage ``[mV]`` at which to evaluate the open probability
        freqs: float, complex, or `np.ndarray` of float or complex:
            The frequencies ``[Hz]`` at which to evaluate the linearized contribution
        e: float or `None`
            The reversal potential of the channel. Defaults to the value stored
            in `self.default_params['e']` if not provided.
        **kwargs: float or `np.ndarray`
            Optional values for the state variables and concentrations.

        Returns
        -------
        float, complex or `np.ndarray` of float or complex
            The linearized current. Shape is dimension of `freqs` followed by
            the dimensions of `v`.
        """
        e = self._get_reversal(e)
        return (e - v) * self.compute_linear(v, freqs, **kwargs) - self.compute_p_open(
            v, **kwargs
        )

    def compute_lin_conc(self, v, freqs, ion, e=None, **kwargs):
        """
        Combute the linearized channel current contribution from the concentrations

        Parameters
        ----------
        v: float or `np.ndarray`
            The voltage ``[mV]`` at which to evaluate the open probability
        freqs: float, complex, or `np.ndarray` of float or complex:
            The frequencies ``[Hz]`` at which to evaluate the linearized contribution
        ion: str
            The ion name for which to compute the linearized contribution
        e: float or `None`
            The reversal potential of the channel. Defaults to the value stored
            in `self.default_params['e']` if not provided.
        **kwargs: float or `np.ndarray`
            Optional values for the state variables and concentrations.

        Returns
        -------
        float, complex or `np.ndarray` of float or complex
            The linearized current. Shape is dimension of `freqs` followed by
            the dimensions of `v`.
        """
        e = self._get_reversal(e)
        return (e - v) * self.compute_linear_conc(v, freqs, ion, **kwargs)

    def write_mod_file(self, path, g=0.0, e=None):
        """
        Writes a modfile of the ion channel for simulations with neuron
        """
        cname = self.__class__.__name__
        sv = [str(svar) for svar in self.ordered_statevars]
        cs = [str(conc) for conc in self.conc]
        e = self._get_reversal(e)

        modname = "I" + cname + ".mod"
        fname = os.path.join(path, modname)

        file = open(fname, "w")

        file.write(
            ": This mod file is automaticaly generated by the "
            + "``neat.channels.ionchannels`` module\n\n"
        )

        file.write("NEURON {\n")
        file.write("    SUFFIX I%s\n" % cname)
        if self.ion == "":
            file.write("    NONSPECIFIC_CURRENT i" + "\n")
        else:
            file.write("    USEION %s WRITE i%s\n" % (self.ion, self.ion))
        for c in cs:
            file.write("    USEION %s READ %si\n" % (c, c))
        file.write("    RANGE  g, e" + "\n")

        taustring = "tau_" + ", tau_".join(sv)
        varstring = "_inf, ".join(sv) + "_inf"
        file.write("    RANGE %s, %s\n" % (varstring, taustring))
        file.write("    THREADSAFE" + "\n")
        file.write("}\n\n")

        file.write("PARAMETER {\n")
        file.write("    g = " + str(g * 1e-6) + " (S/cm2)" + "\n")
        file.write("    e = " + str(e) + " (mV)" + "\n")
        file.write("    celsius (degC)\n")
        file.write("}\n\n")

        file.write("UNITS {\n")
        file.write("    (mA) = (milliamp)" + "\n")
        file.write("    (mV) = (millivolt)" + "\n")
        file.write("    (mM) = (milli/liter)" + "\n")
        file.write("}\n\n")

        file.write("ASSIGNED {\n")
        file.write("    i%s (mA/cm2)\n" % self.ion)
        for var in sv:
            file.write("    %s_inf      \n" % var)
            file.write("    tau_%s (ms) \n" % var)
        for ion in cs:
            file.write("    " + ion + "i (mM)" + "\n")
        file.write("    v (mV)" + "\n")
        file.write("    %s (degC)\n" % (self.sp_t))
        file.write("}\n\n")

        file.write("STATE {\n")
        for var in sv:
            file.write("    %s\n" % var)
        file.write("}\n\n")

        calcstring = "i%s = g * (%s) * (v - e)" % (
            self.ion,
            sp.printing.ccode(self.p_open),
        )

        file.write("BREAKPOINT {\n")
        file.write("    SOLVE states METHOD cnexp" + "\n")
        file.write("    %s\n" % calcstring)
        file.write("}\n\n")

        concstring = "i, ".join(cs)
        if len(cs) > 0:
            concstring = ", " + concstring
            concstring += "i"

        file.write("INITIAL {\n")
        file.write("    rates(v%s)\n" % concstring)
        for var in sv:
            file.write("    %s = %s_inf\n" % (var, var))
        file.write("}\n\n")

        file.write("DERIVATIVE states {\n")
        file.write("    rates(v%s)\n" % concstring)
        for var in sv:
            file.write("    %s' = (%s_inf - %s) /  tau_%s \n" % (var, var, var, var))
        file.write("}\n\n")

        # substitution for common neuron names
        repl_pairs = [(str(c), str(c) + "i") for c in self.conc]

        assignment_lines = []
        local_vars = []
        temp_counter = 0
        for var, svar in zip(sv, self.ordered_statevars):
            vi_lines, temp_counter, vi_locals = _nmodl_assignment_lines(
                f"{var}_inf", self.varinf[svar], temp_counter
            )
            ti_lines, temp_counter, ti_locals = _nmodl_assignment_lines(
                f"tau_{var}", self.tauinf[svar], temp_counter
            )
            local_vars.extend(vi_locals)
            local_vars.extend(ti_locals)
            assignment_lines.extend(vi_lines)
            assignment_lines.extend(ti_lines)

        for ii, line in enumerate(assignment_lines):
            for repl_pair in repl_pairs:
                line = line.replace(*repl_pair)
            assignment_lines[ii] = line

        file.write("PROCEDURE rates(v%s) {\n" % concstring)
        if local_vars:
            file.write("    LOCAL %s\n" % ", ".join(dict.fromkeys(local_vars)))
        file.write("    %s = celsius\n" % str(self.sp_t))
        for line in assignment_lines:
            file.write(f"    {line}\n")
        file.write("}\n\n")

        file.close()

        return modname

    def _create_nestml_funcstr(self, code_str, n_spaces=0, indent=8):
        """
        This function is used to recursively expand if... else... statements
        across multiple lines, as by default the single line version is printed
        by `sympy.pycode()` and `ast.unparse()`
        """
        tree = ast.parse(code_str)
        iev = IfExpVisitor()
        ifexp = iev.find_IfExp_node(tree)

        if ifexp is not None:
            # sanity check
            assert iev.find_IfExp_node(ifexp.test) is None
            # if test is True
            cond_1_str = self._create_nestml_funcstr(
                ast.unparse(ifexp.body), n_spaces=n_spaces, indent=n_spaces + indent
            )
            # if test is False
            cond_0_str = self._create_nestml_funcstr(
                ast.unparse(ifexp.orelse), n_spaces=n_spaces, indent=n_spaces + indent
            )
            code_str = (
                " " * indent
                + f"if {ast.unparse(ifexp.test)}:\n"
                + f"{cond_1_str}"
                + " " * indent
                + f"else:\n"
                + f"{cond_0_str}"
            )
        else:
            try:
                code_str = (
                    " " * indent + f"val = {sp.printing.ccode(sp.sympify(code_str))}\n"
                )
            except TypeError as e:
                print(e)
        return code_str

    def write_nestml_blocks(
        self,
        blocks=["state", "parameters", "equations", "function"],
        v_comp=-75.0,
        g=0.0,
        e=None,
    ):
        cname = self.__class__.__name__
        sv = [str(svar) for svar in self.ordered_statevars]
        cs = [str(conc) for conc in self.conc]
        sv_suff = [sv_ + "_" + cname for sv_ in sv]
        e = self._get_reversal(e)
        sv_init = self.compute_varinf(v_comp)

        blocks_dict = {block: "" for block in blocks}

        func_call_args = ["v_comp real"]
        for ckey, cval in self.conc.items():
            func_call_args.append(f"{ckey} real")
        func_call_args = ", ".join(func_call_args)

        func_args = ["v_comp"]
        for ckey, cval in self.conc.items():
            func_args.append(f"c_{ckey}")
        func_args = ", ".join(func_args)

        if "state" in blocks:
            state_str = "\n" + "        # state variables %s\n" % cname
            for sv_, sv_key in zip(sv_suff, sv):
                state_str += "        %s real = %.8f\n" % (sv_, sv_init[sv_key])

            blocks_dict["state"] += state_str

        if "parameters" in blocks:
            param_str = (
                "\n"
                + "        # parameters %s\n" % cname
                + "        gbar_%s real = %.2f\n" % (cname, g)
                + "        e_%s real = %.2f\n" % (cname, e)
            )

            blocks_dict["parameters"] += param_str

        if "equations" in blocks:
            # reformulate open probability in terms of suffixed variables
            p_open_ = self.p_open
            for svar, sv_ in zip(self.ordered_statevars, sv_suff):
                p_open_ = p_open_.subs(svar, sp.symbols(sv_))
                p_open_ = p_open_.subs(
                    self.sp_v, sp.UnevaluatedExpr(sp.symbols("v_comp"))
                )

            eq_str = (
                "\n"
                + "        # equation %s\n" % cname
                + "        inline i_%s real = gbar_%s * (%s) * (e_%s - v_comp) @mechanism::channel\n"
                % (cname, cname, str(p_open_), cname)
            )

            for var, var_suff, svar in zip(sv, sv_suff, self.ordered_statevars):
                vi = sp.printing.ccode(self.varinf[svar])
                ti = sp.printing.ccode(self.tauinf[svar])

                eq_str += f"        {var_suff}' = ( {var}_inf_{cname}( {func_args} ) - {var_suff} ) / ( tau_{var}_{cname}( {func_args} ) * 1s )\n"

            eq_str += "\n"
            blocks_dict["equations"] += eq_str

        def _customsimplify(expr):
            return sp.logcombine(sp.powsimp(sp.expand(expr)))

        if "function" in blocks:
            func_str = "\n" + "    # functions %s\n" % cname
            for svar, sv_, sv_suff_ in zip(self.ordered_statevars, sv, sv_suff):
                # substitute possible default values and concentrations
                varinf_func = self._substitute_defaults(self.varinf[svar])
                func_args = ["v_comp real"]
                for ckey, cval in self.conc.items():
                    func_args.append(f"{ckey} real")
                func_args = ", ".join(func_args)
                #     varinf_func = varinf_func.subs(ckey, cval)
                # print activation function to nestml file
                varinf_func = varinf_func.subs(
                    svar, sp.UnevaluatedExpr(sp.symbols(sv_suff_))
                )
                varinf_func = varinf_func.subs(
                    self.sp_v, sp.UnevaluatedExpr(sp.symbols("v_comp"))
                )

                code_str = sp.pycode(varinf_func, fully_qualified_modules=False)
                func_str += (
                    f"    function {sv_}_inf_{cname} ({func_call_args}) real:\n"
                    f"        val real\n"
                    f"{self._create_nestml_funcstr(code_str, n_spaces=4, indent=8)}"
                    f"        return val\n\n"
                )

                # substitute possible default values and concentrations
                tauinf_func = self._substitute_defaults(self.tauinf[svar])
                for ckey, cval in self.conc.items():
                    tauinf_func = tauinf_func.subs(ckey, cval)

                tauinf_func = tauinf_func.subs(
                    svar, sp.UnevaluatedExpr(sp.symbols(sv_suff_))
                )
                tauinf_func = tauinf_func.subs(
                    self.sp_v, sp.UnevaluatedExpr(sp.symbols("v_comp"))
                )

                code_str = sp.pycode(tauinf_func, fully_qualified_modules=False)
                func_str += (
                    f"\n    function tau_{sv_}_{cname} ({func_call_args}) real:\n"
                    f"        val real\n"
                    f"{self._create_nestml_funcstr(code_str, n_spaces=4, indent=8)}"
                    f"        return val\n\n"
                )

            blocks_dict["function"] += func_str

        return blocks_dict

    def write_cpp_code(self, path):
        """
        Concentration dependent ion channels get constant concentrations
        substituted for c++ simulation
        """
        c_name = self.__class__.__name__
        svs = [str(svar) for svar in self.ordered_statevars]
        # rewrite open probabilities
        p_open_m = self.p_open
        p_open_m_inf = self.p_open
        for svar in self.ordered_statevars:
            p_open_m = p_open_m.subs(svar, sp.symbols("m_" + str(svar)))
            p_open_m_inf = p_open_m_inf.subs(
                svar, sp.symbols("m_" + str(svar) + "_inf")
            )

        # substitue concentrations in expression
        def _replaceConc(expr_str, prefix="", suffix=""):
            for ion, conc in self.conc.items():
                expr_str = expr_str.replace(str(ion), prefix + str(ion) + suffix)
            return expr_str

        # open header and cc files
        fcc = open(os.path.join(path, "Ionchannels.cc"), "a")
        fh = open(os.path.join(path, "Ionchannels.h"), "a")

        # define class and functions in header file
        fh.write("class %s: public IonChannel{\n" % c_name)
        fh.write("private:" + "\n")
        for svar in self.ordered_statevars:
            sv = sp.printing.ccode(svar)
            fh.write("    double m_%s;\n" % sv)
            fh.write("    double m_%s_inf, m_tau_%s;\n" % (sv, sv))
            fh.write("    double m_v_%s = 10000.;\n" % sv)
        fh.write("    double m_p_open_eq = 0.0, m_p_open = 0.0;\n")
        # hardcode default concentrations
        for ion, conc in self.conc.items():
            fh.write("    double m_%s = %.8f;\n" % (ion, conc))
        fh.write("public:" + "\n")
        fh.write("    void calcFunStatevar(double v) override;" + "\n")
        fh.write("    double calcPOpen() override;" + "\n")
        fh.write("    void setPOpen() override;" + "\n")
        fh.write("    void setPOpenEQ(double v) override;" + "\n")
        fh.write("    void advance(double dt) override;" + "\n")
        fh.write("    double getCond() override;" + "\n")
        fh.write("    double getCondNewton() override;" + "\n")
        fh.write("    double f(double v) override;" + "\n")
        fh.write("    double DfDv(double v) override;" + "\n")
        fh.write("    void setfNewtonConstant(double* vs, int v_size) override;" + "\n")
        fh.write("    double fNewton(double v) override;" + "\n")
        fh.write("    double DfDvNewton(double v) override;" + "\n")
        fh.write("};" + "\n")

        # function in cc file
        fcc.write("void %s::calcFunStatevar(double v){\n" % c_name)
        for svar in self.ordered_statevars:
            varinf = self._substitute_defaults(self.varinf[svar])
            tauinf = self._substitute_defaults(self.tauinf[svar])
            sv = str(svar)
            vi = _replaceConc(sp.printing.ccode(varinf), prefix="m_")
            ti = _replaceConc(sp.printing.ccode(tauinf), prefix="m_")
            fcc.write("    m_%s_inf = %s;\n" % (sv, vi))
            # if self.varinf.shape[1] == 2 and ind == (0,0):
            if sv == "m":
                # instantaneous approximation possible if statevar is activation (denoted by 'm')
                fcc.write("    if(m_instantaneous)" + "\n")
                fcc.write(
                    "        m_tau_%s = %s;\n" % (sv, sp.printing.ccode(sp.Float(1e-5)))
                )
                fcc.write("    else" + "\n")
                fcc.write("        m_tau_%s = %s;\n" % (sv, ti))
            else:
                fcc.write("    m_tau_%s = %s;\n" % (sv, ti))
        fcc.write("}\n")

        fcc.write("double %s::calcPOpen(){\n" % c_name)
        fcc.write("    return %s;\n" % sp.printing.ccode(p_open_m))
        fcc.write("}\n")

        fcc.write("void %s::setPOpen(){\n" % c_name)
        fcc.write("    m_p_open = calcPOpen();\n")
        fcc.write("}\n")

        fcc.write("void %s::setPOpenEQ(double v){\n" % c_name)
        fcc.write("    calcFunStatevar(v);\n")
        fcc.write("\n")
        for sv in svs:
            fcc.write("    m_%s = m_%s_inf;\n" % (sv, sv))
        fcc.write("    m_p_open_eq = %s;\n" % sp.printing.ccode(p_open_m_inf))
        fcc.write("}\n")

        fcc.write("void %s::advance(double dt){\n" % c_name)
        for sv in svs:
            fcc.write("    double p0_%s = exp(-dt / m_tau_%s);\n" % (sv, sv))
            fcc.write("    m_%s *= p0_%s ;\n" % (sv, sv))
            fcc.write("    m_%s += (1. - p0_%s) *  m_%s_inf;\n" % (sv, sv, sv))
        fcc.write("}\n")

        fcc.write("double %s::getCond(){\n" % c_name)
        fcc.write("    return m_g_bar * (m_p_open - m_p_open_eq);\n")
        fcc.write("}\n")

        fcc.write("double %s::getCondNewton(){\n" % c_name)
        fcc.write("    return m_g_bar;\n")
        fcc.write("}\n")

        # function for temporal integration
        fcc.write("double %s::f(double v){\n" % c_name)
        fcc.write("    return (m_e_rev - v);\n")
        fcc.write("}\n")

        fcc.write("double %s::DfDv(double v){\n" % c_name)
        fcc.write("    return -1.;\n")
        fcc.write("}\n")

        # set voltage values to evaluate at constant voltage during newton iteration
        fcc.write("void %s::setfNewtonConstant(double* vs, int v_size){\n" % c_name)
        fcc.write("    if(v_size != %d)" % len(self.ordered_statevars) + "\n")
        fcc.write(
            '        cerr << "input arg [vs] has incorrect size, '
            + 'should have same size as number of channel state variables" << endl'
            + ";\n"
        )
        for ii, svar in enumerate(self.ordered_statevars):
            fcc.write("    m_v_%s = vs[%d];\n" % (str(svar), ii))
        fcc.write("}\n")

        # functions for solving Newton iteration
        fcc.write("double %s::fNewton(double v){\n" % c_name)
        p_o = self.p_open
        for svar in self.ordered_statevars:
            sv = "v_" + str(svar)
            # substitute default parameters
            vi = self._substitute_defaults(self.varinf[svar])
            # write ccode and substitute variable names
            vi_ccode = sp.printing.ccode(vi)
            vi_ccode = vi_ccode.replace(str(self.sp_v), sv)
            vi_ccode = _replaceConc(vi_ccode, prefix="m_")
            # assign dynamic or fixed voltage to the activation
            fcc.write("    double %s;\n" % (sv))
            fcc.write("    if(m_%s > 1000.){\n" % sv)
            fcc.write("        %s = v;\n" % (sv))
            fcc.write("    } else{\n")
            fcc.write("        %s = m_%s;\n" % (sv, sv))
            fcc.write("    }" + "\n")
            fcc.write("    double %s = %s;\n" % (str(svar), vi_ccode))

        fcc.write(
            "    return (m_e_rev - v) * (%s - m_p_open_eq);\n"
            % sp.printing.ccode(self.p_open)
        )
        fcc.write("}\n")

        fcc.write("double %s::DfDvNewton(double v){\n" % c_name)
        dp_o = {svar: sp.diff(self.p_open, svar, 1) for svar in self.ordered_statevars}

        # print derivatives
        for svar in self.ordered_statevars:
            sv = "v_" + str(svar)
            v_var = sp.symbols(sv)

            # substitute default parameters
            vi = self._substitute_defaults(self.varinf[svar])
            # write ccode and substitute variable names
            vi_ccode = sp.printing.ccode(vi)
            vi_ccode = vi_ccode.replace(str(self.sp_v), sv)
            vi_ccode = _replaceConc(vi_ccode, prefix="m_")
            # compute voltage derivatives
            dvi_dv = sp.diff(vi, self.sp_v, 1)
            dvi_dv_ccode = sp.printing.ccode(dvi_dv)
            dvi_dv_ccode = dvi_dv_ccode.replace(str(self.sp_v), sv)
            dvi_dv_ccode = _replaceConc(dvi_dv_ccode, prefix="m_")

            # compute derivative
            fcc.write("    double %s;\n" % sv)
            fcc.write("    double d%s_dv;\n" % str(svar))
            fcc.write("    if(m_%s > 1000.){\n" % sv)
            fcc.write("        %s = v;\n" % sv)
            fcc.write("        d%s_dv = %s;\n" % (str(svar), dvi_dv_ccode))
            fcc.write("    } else{\n")
            fcc.write("        %s = m_%s;\n" % (sv, sv))
            fcc.write("        d%s_dv = 0;\n" % str(svar))
            fcc.write("    }\n")
            fcc.write("    double %s = %s;\n" % (str(svar), vi_ccode))

        expr_str = " + ".join(
            [
                "%s * d%s_dv" % (sp.printing.ccode(dp_o[svar]), str(svar))
                for svar in self.ordered_statevars
            ]
        )

        fcc.write(
            "    return -1. * (%s - m_p_open_eq) + (%s) * (m_e_rev - v);\n"
            % (sp.printing.ccode(self.p_open), expr_str)
        )
        fcc.write("}\n")

        fh.write("\n")
        fcc.write("\n")

        fh.close()
        fcc.close()
