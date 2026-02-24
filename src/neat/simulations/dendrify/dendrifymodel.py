import numpy as np
import sympy as sp
import warnings
from typing import Dict, List, Optional, Tuple

from ...trees.compartmenttree import CompartmentTree, CompartmentNode
from .brian2model import _sympy_to_brian2_str


# ── Brian2 / Dendrify imports (optional at import time for type-checking) ──
try:
    from brian2.units import pF, nS, mV, uS, uF
    _BRIAN2_AVAILABLE = True
except ImportError:
    _BRIAN2_AVAILABLE = False
    warnings.warn(
        "brian2 is not installed. DendrifyCompartmentTree requires brian2 "
        "and dendrify to export models.",
        ImportWarning,
    )

try:
    from dendrify import Soma, Dendrite, NeuronModel
    _DENDRIFY_AVAILABLE = True
except ImportError:
    _DENDRIFY_AVAILABLE = False
    warnings.warn(
        "dendrify is not installed. DendrifyCompartmentTree requires dendrify "
        "to export models.",
        ImportWarning,
    )

def ion_channel_dendrify_eqs(
    ion_channel,
    chan_name: str,
    comp_name: str,
    g_bar_uS: float,
    e_rev_mV: float,
) -> Tuple[List[str], dict, str]:
    """
    Build full Hodgkin-Huxley Brian2 equations from a NEAT IonChannel.

    Returns ``(equation_lines, params_dict, current_variable_name)``.

    Gate variables are namespaced as ``<gate>_<chan>_<comp>`` to avoid
    clashes when the same channel type appears on multiple compartments.
    """
    eqs: List[str] = []
    params: dict = {}

    V_var = f"V_{comp_name}"

    if hasattr(ion_channel, "alpha") and ion_channel.alpha:
        gates = list(ion_channel.alpha.keys())
        use_alpha_beta = True
    elif hasattr(ion_channel, "tauinf") and ion_channel.tauinf:
        gates = list(ion_channel.tauinf.keys())
        use_alpha_beta = False
    else:
        raise ValueError(
            f"IonChannel '{chan_name}' has neither alpha/beta nor tauinf/varinf definitions."
        )

    v_sym = sp.Symbol("v")
    v_brian = sp.Symbol(f"({V_var}/mV)")
    gate_subs = {
        sp.Symbol(str(gate)): sp.Symbol(f"{gate}_{chan_name}_{comp_name}")
        for gate in gates
    }

    def _to_brian2_str(sympy_expr, substitute_v=True) -> str:
        expr = sympy_expr.subs(gate_subs)
        if substitute_v:
            expr = expr.subs(v_sym, v_brian)
        return _sympy_to_brian2_str(expr)

    p_open_brian = _to_brian2_str(ion_channel.p_open, substitute_v=False)

    I_var = f"I_{chan_name}_{comp_name}"
    gbar_var = f"gbar_{chan_name}_{comp_name}"
    E_var = f"E_{chan_name}_{comp_name}"

    params[gbar_var] = g_bar_uS * uS
    params[E_var] = e_rev_mV * mV

    eqs.append(
        f"{I_var} = {gbar_var} * ({p_open_brian}) "
        f"* ({E_var} - {V_var}) : amp"
    )

    for gate in gates:
        g_ns = f"{gate}_{chan_name}_{comp_name}"

        if use_alpha_beta:
            alpha_str = _to_brian2_str(ion_channel.alpha[gate])
            beta_str = _to_brian2_str(ion_channel.beta[gate])
            alpha_var = f"alpha_{g_ns}"
            beta_var = f"beta_{g_ns}"

            eqs.append(
                f"d{g_ns}/dt = ({alpha_var} * (1 - {g_ns}) "
                f"- {beta_var} * {g_ns}) / ms : 1"
            )
            eqs.append(f"{alpha_var} = {alpha_str} : 1")
            eqs.append(f"{beta_var}  = {beta_str}  : 1")
        else:
            tauinf_str = _to_brian2_str(ion_channel.tauinf[gate])
            varinf_str = _to_brian2_str(ion_channel.varinf[gate])
            xinf_var = f"xinf_{g_ns}"
            taux_var = f"taux_{g_ns}"

            eqs.append(
                f"d{g_ns}/dt = ({xinf_var} - {g_ns}) "
                f"/ ({taux_var} * ms) : 1"
            )
            eqs.append(f"{xinf_var} = {varinf_str} : 1")
            eqs.append(f"{taux_var} = {tauinf_str} : 1")

    return eqs, params, I_var


class DendrifyCompartmentTree(CompartmentTree):
    """
    DendrifyCompartmentTree
    =======================

    A subclass of `neat.CompartmentTree` that exports reduced multi-compartmental
    neuron models — specified as electrical compartments with coupling conductances —
    to the Dendrify toolbox for simulation with Brian2.

    Dendrify is designed around *dimensionless* compartments (absolute capacitance
    and conductance), which maps onto the NEAT compartment representation
    where each node carries:

        node.ca       – capacitance  [uF]   (absolute, not specific)
        node.g_c      – coupling conductance to parent [uS]
        node.currents – dict  {channel_name: [g_bar (uS), e_rev (mV)]}

    Units map to Brian2 / Dendrify units:
        1 uF  -> 1e-6 * farad  -> 1e6 * pF
        1 uS  -> 1e-6 * siemens -> 1e3 * nS
        1 mV  -> 1e-3 * volt

    Ion channel kinetics translation
    ---------------------------------
    NEAT's IonChannel class specifies gating kinetics as Python string expressions
    in the variable ``v`` (membrane voltage in mV). Each channel defines:

        channel.p_open  – open-probability formula, e.g. ``'m**3 * h'``

    and *one* of the two rate parameterisations:

        channel.alpha / channel.beta
            dict {gate_var: rate_string(v)}  — transition rates in [1/ms]

        channel.tauinf / channel.varinf
            dict {gate_var: formula_string(v)}  — time constant [ms] and
            steady-state activation [dimensionless]

    This module translates either form into proper Brian2 ODEs.  Because Brian2
    works in SI units while NEAT uses mV for membrane voltage, every occurrence
    of the symbol ``v`` in a NEAT formula is substituted by ``(V_<comp>/mV)``
    so that the numeric mV value is correctly passed to the expression.

    When a channel's ``IonChannel`` object is available (via ``channel_storage``),
    full Hodgkin-Huxley kinetics are generated.  When only the fitted ``g_bar``
    and ``e_rev`` from the CompartmentTree are available (fallback), a simpler
    constant-conductance current is emitted with a UserWarning.

    Requirements
    ------------
        pip install nest-neat dendrify brian2

    Usage example
    -------------
        import neat
        from dendrify_compartment_tree import DendrifyCompartmentTree

        # Build/load a CompartmentTree the usual NEAT way, e.g.:
        ctree = neat.CompartmentTree(...)
        ...  # fit parameters

        # Wrap it — pass the channel_storage dict so full kinetics are emitted
        dtree = DendrifyCompartmentTree(ctree, channel_storage=ctree.channel_storage)

        # Get the Dendrify NeuronModel
        model = dtree.init_model(soma_model='leakyIF')

        # Build a Brian2 NeuronGroup
        neuron = model.make_neurongroup(
            N=1,
            threshold='V_soma > -40*mV',
            reset='V_soma = -70*mV',
            method='euler',
        )
    """

    def __init__(self, arg=None, channel_storage: Optional[Dict[str, "IonChannel"]] = None):
        super().__init__(arg)
        self._channel_storage = channel_storage or {}

    def get_compartment_objects(self, soma_model: str = "leakyIF"):
        """
        Build Dendrify :class:`~dendrify.Soma` / :class:`~dendrify.Dendrite`
        objects for every node in the tree.

        Parameters
        ----------
        soma_model : str, optional
            The soma model keyword passed to :class:`~dendrify.Soma`.
            Defaults to ``'leakyIF'``.  Other options provided by Dendrify
            include ``'adaptiveIF'`` and ``'adex'``.

        Returns
        -------
        compartments : dict[int, Soma | Dendrite]
            Mapping from node index to Dendrify compartment object.
        """
        compartments: dict = {}
        connections: list = []

        for node in self:          # iterates in tree order (root first)
            self._validate_node_params(node)
            name = self._node_name(node)

            if self.is_root(node):
                # Root node → soma
                comp = Soma(
                    name,
                    model=soma_model,
                    cm_abs=node.ca * uF, 
                    gl_abs=node.currents['L'][0] * uS, 
                    v_rest=node.currents['L'][1] * mV, 
                )
            else:
                # Non-root nodes → dendrites (passive by default)
                comp = Dendrite(
                    name,
                    model="passive",
                    cm_abs=node.ca * uF,
                    gl_abs=node.currents['L'][0] * uS, 
                    v_rest=node.currents['L'][1] * mV, 
                )
                connections.append((compartments[node.parent_node.index], comp, node.g_c * uS))

            # Add non-leak active channels as constant-conductance currents
            self._add_active_channels(comp, node, name)

            compartments[node.index] = comp

        return compartments, connections

    def init_model(
        self,
        soma_model: str = "leakyIF",
        v_rest: Optional[float] = None,
    ) -> "NeuronModel":
        """
        Construct and return a Dendrify :class:`~dendrify.NeuronModel` from
        the compartment tree.

        This is the primary convenience method.  It:

        1. Creates Dendrify :class:`~dendrify.Soma` / :class:`~dendrify.Dendrite`
           objects for every node.
        2. Wires them together using the coupling conductances stored in the tree.
        3. Wraps everything in a :class:`~dendrify.NeuronModel`.

        Parameters
        ----------
        soma_model : str, optional
            Soma model keyword (``'leakyIF'``, ``'adaptiveIF'``, ``'adex'``).
        v_rest : float, optional
            Global resting potential override in **mV**.  If ``None`` (default),
            each compartment uses its own ``node.e_eq`` value.

        Returns
        -------
        model : dendrify.NeuronModel
            Ready to call ``model.make_neurongroup(...)``.

        Example
        -------
        >>> dtree = DendrifyCompartmentTree(fitted_ctree)
        >>> model = dtree.init_model(soma_model='leakyIF')
        >>> neuron = model.make_neurongroup(
        ...     N=200,
        ...     threshold='V_soma > -40*mV',
        ...     reset='V_soma = -70*mV',
        ...     method='euler',
        ... )
        """
        compartments, connections = self.get_compartment_objects(soma_model=soma_model)
        print(connections)


        kwargs = {}
        if v_rest is not None:
            kwargs["v_rest"] = _mV(v_rest)

        model = NeuronModel(connections, **kwargs)
        return model

    def print_summary(self):
        """
        Print a human-readable summary of each compartment's parameters as
        they will be exported to Dendrify.
        """
        print(f"{'node':>5}  {'name':>15}  {'Ca (pF)':>10}  "
              f"{'gl (nS)':>10}  {'e_eq (mV)':>10}  {'g_c (nS)':>10}  "
              f"active channels")
        print("-" * 80)
        for node in self:
            name = self._node_name(node)
            ca_pF = node.ca * 1e6
            gl_nS = node.g_l * 1e3
            gc_nS = node.g_c * 1e3
            active = [k for k in node.currents if k != "L"]
            print(
                f"{node.index:>5}  {name:>15}  {ca_pF:>10.3f}  "
                f"{gl_nS:>10.3f}  {node.e_eq:>10.2f}  {gc_nS:>10.3f}  "
                f"{active or '—'}"
            )

    def _node_name(self, node: "CompartmentNode") -> str:
        """
        Return a clean, Dendrify-compatible name for a node.

        Dendrify uses the compartment name as a suffix in variable names
        (e.g., ``V_soma``, ``C_dend``), so it must be a valid Python
        identifier and must be unique within the model.

        The root node is always named ``'soma'`` so that Dendrify's
        ``V_soma`` threshold/reset strings work out of the box.
        """
        if node.index == self.root.index:
            return "soma"
        if node.loc_idx is not None:
            return f"dend{node.index}"
        return f"dend{node.index}"

    @staticmethod
    def _validate_node_params(node: "CompartmentNode"):
        """
        Raise a descriptive ValueError if a node's fitted parameters are
        unphysical.  This guards against accidentally passing an unfitted
        tree, where ``ca``, ``g_l``, or ``e_eq`` may be zero, negative,
        or NaN — any of which would cause Dendrify to silently fall back
        to its morphological (length/diameter) pathway and raise a
        confusing ``DimensionlessCompartmentError`` at connection time.
        """
        errs = []
        if not np.isfinite(node.ca) or node.ca <= 0:
            errs.append(f"ca={node.ca} µF (must be finite and > 0)")
        if not np.isfinite(node.currents['L'][0]) or node.currents['L'][0] < 0:
            errs.append(f"g_l={node.currents['L'][0]} µS (must be finite and ≥ 0)")
        if not np.isfinite(node.e_eq):
            errs.append(f"e_eq={node.e_eq} mV (must be finite)")
        if errs:
            raise ValueError(
                f"Node {node.index} has unphysical parameters — is the tree "
                f"fully fitted?\n  " + "\n  ".join(errs)
            )

    def _add_active_channels(
        self,
        comp,
        node: "CompartmentNode",
        comp_name: str,
    ):
        """
        Translate every non-leak active channel on *node* into Brian2 equations
        and inject them into the Dendrify compartment *comp*.

        For each channel, look up the :class:`~neat.IonChannel` object in
        ``self._channel_storage`` and emit full HH kinetics (gate ODEs +
        conductance-based current).
        """
        active_channels = {
            k: v
            for k, v in node.currents.items()
            if k != "L"
        }
        if not active_channels:
            return

        all_eqs: List[str] = []
        all_params: Dict[str, object] = {}
        current_terms: List[str] = []
        for chan_name, (g_bar_uS, e_rev_mV) in active_channels.items():
            ion_channel = self._channel_storage.get(chan_name)
            if ion_channel is None:
                raise KeyError(
                    f"IonChannel object for '{chan_name}' not found in channel_storage. "
                    f"Pass channel_storage=ctree.channel_storage to DendrifyCompartmentTree."
                )

            eqs, params, i_term = ion_channel_dendrify_eqs(
                ion_channel, chan_name, comp_name, g_bar_uS, e_rev_mV,
            )

            all_eqs.extend(eqs)
            all_params.update(params)
            current_terms.append(i_term)

        if all_eqs:
            comp.add_equations("\n".join(all_eqs))

            # Wire all channel currents into the compartment's total current.
            I_total = f"I_{comp_name}"
            I_new_terms = " + ".join(current_terms)
            try:
                comp.replace_equations(
                    f"{I_total} = I_ext_{comp.name}",
                    f"{I_total} = I_ext_{comp.name} + {I_new_terms}",
                )
            except Exception:
                warnings.warn(
                    f"Could not automatically wire channel currents "
                    f"({list(active_channels)}) into the total current of "
                    f"compartment '{comp_name}'.  You may need to do this manually.",
                    UserWarning,
                )

            if hasattr(comp, "add_params"):
                comp.add_params(all_params)
            else:
                for k, v in all_params.items():
                    setattr(comp, k, v)

