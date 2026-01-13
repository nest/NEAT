import numpy as np

import os
import sys
import copy
import importlib

from ...trees.phystree import PhysTree, PhysNode
from ...trees.morphtree import MorphTree, MorphLoc
from ...trees.compartmenttree import CompartmentTree
from ...factorydefaults import DefaultPhysiology

try:
    import jax.numpy as jnp
    import jaxley as jx
    from jaxley.channels import Leak

except ModuleNotFoundError:
    warnings.warn(
        "Jaxley not available, importing non-functional jx module only for doc generation",
        UserWarning,
    )

    # universal iterable mock object
    class JX(object):
        def __init__(self):
            pass

        def __getattr__(self, attr):
            try:
                return super(H, self).__getattr__(attr)
            except AttributeError:
                return self.__global_handler

        def __global_handler(self, *args, **kwargs):
            return JX()

        def __iter__(self):  # make iterable
            return self

        def __next__(self):
            raise StopIteration

        def __mul__(self, other):  # make multipliable
            return 1.0

        def __rmul__(self, other):
            return self * other

        def __call__(self, *args, **kwargs):  # make callable
            return JX()

    jx = JX()
    np_array = np.array

    def array(*args, **kwargs):
        if isinstance(args[0], JX):
            print(args)
            print(kwargs)
            return np.eye(2)
        else:
            return np_array(*args, **kwargs)

    np.array = array

JX_MECH = {}

def load_jaxley_model(name):
    jx_path = os.path.join(
        os.path.dirname(__file__),
        f"tmp/",
    )
    if os.path.exists(jx_path + f"{name}.py"):
        if jx_path not in sys.path:
            sys.path.insert(0, jx_path)
        JX_MECH[name] = importlib.import_module(name)
    else:
        raise FileNotFoundError(
            f"The Jaxley model named '{name}' is not installed. "
            f"Run 'neatmodels -h' in the terminal for help on "
            f"installing new Jaxley models with NEAT. "
            f"Installed models will be in '{jx_path}'."
        )


class JaxleySimNode(PhysNode):
    """
    Subclass of `neat.PhysNode` that implements functionality to instantiate a
    cylindrical `neuron.h.Section` based on its physiological and geometrical
    parameters.
    """

    def __init__(self, index, p3d=None):
        super().__init__(index, p3d)

    def _make_branch(self, jx_channels, dx_max=15, factorlambda=1.0, pprint=False):
        if dx_max is None:
            n_comp = factorlambda

        if self.index == 1:
            n_comp = 1
            l_comp = 2.0 * self.R

        else:
            n_comp = int(self.L / dx_max) + 1
            l_comp = self.L / n_comp

        compartments = []
        for _ in range(n_comp):
            jx_comp = jx.Compartment()
            jx_comp.set("length", l_comp)
            jx_comp.set("radius", self.R)
            jx_comp.set("axial_resistivity", self.r_a * 1e6)  # MOhm*cm --> Ohm*cm
            jx_comp.set("capacitance", self.c_m) # uF/cm^2

            for key, current in self.currents.items():
                jx_channel = jx_channels[key]
                if current[0] > 1e-10:
                    prefix = jx_channel.prefix if key != 'L' else 'Leak_'
                    suffix = key if key != 'L' else 'Leak'
                    jx_comp.insert(jx_channel)
                    jx_comp.set(f"{prefix}g{suffix}", current[0]*1e-6) # uS/cm^2 --> S/cm^2
                    jx_comp.set(f"{prefix}e{suffix}", current[1]) # mV

            compartments.append(jx_comp)

        branch = jx.Branch(compartments)
        return branch


class JaxleySimTree(PhysTree):
    """
    Tree class to define NEURON (Carnevale & Hines, 2004) based on `neat.PhysTree`.

    Attributes
    ----------
    sections: dict of hoc sections
        Storage for hoc sections. Keys are node indices.
    shunts: list of hoc mechanisms
        Storage container for shunts
    syns: list of hoc mechanisms
        Storage container for synapses
    iclamps: list of hoc mechanisms
        Storage container for current clamps
    vclamps: lis of hoc mechanisms
        Storage container for voltage clamps
    vecstims: list of hoc mechanisms
        Storage container for vecstim objects
    netcons: list of hoc mechanisms
        Storage container for netcon objects
    vecs: list of hoc vectors
        Storage container for hoc spike vectors
    dt: float
        timestep of the simulator ``[ms]``
    t_calibrate: float
        Time for the model to equilibrate``[ms]``. Not counted as part of the
        simulation.
    factor_lambda : int or float
        If int, the number of segments per section. If float, multiplies the
        number of segments given by the standard lambda rule (Carnevale, 2004)
        to give the number of compartments simulated (default value 1. gives
        the number given by the lambda rule)
    v_init: float
        The initial voltage at which the model is initialized ``[mV]``

    A `NeuronSimTree` can be extended easily with custom point process mechanisms.
    Just make sure that you store the point process in an existing appropriate
    storage container or in a custom storage container, since if all references
    to the hocobject disappear, the object itself will be deleted as well.

    .. code-block:: python
        class CustomSimTree(NeuronSimTree):
            def addCustomPointProcessMech(self, loc, **kwargs):
                loc = MorphLoc(loc, self)

                # create the point process
                pp = h.custom_point_process(self.sections[loc['node']](loc['x']))
                pp.arg1 = kwargs['arg1']
                pp.arg2 = kwargs['arg2']
                ...

                self.storage_container_for_point_process.append(pp)

    If you define a custom storage container, make sure that you overwrite the
    `__init__()` and `delete_model()` functions to make sure it is created and
    deleted properly.
    """

    def __init__(self, arg=None, types=[1, 3, 4]):
        self.cell = None
        self.pre_dummies = {}       
        self.syn_locs = []
        self.syn_models = []
        # simulation parameters
        self.dt = 0.1  # ms
        self.t_calibrate = 0.0  # ms
        self.factor_lambda = 1.0
        self.v_init = -75.0  # mV
        self.indstart = 0
        # initialize the tree structure
        super().__init__(arg=arg, types=types)

    def create_corresponding_node(self, node_index, p3d=None):
        """
        Creates a node with the given index corresponding to the tree class.

        Parameters
        ----------
            node_index: int
                index of the new node
        """
        return JaxleySimNode(node_index, p3d=p3d)

    def init_model(
        self, model_name, dt=0.025, t_calibrate=0.0, t_max=100., v_init=-75.0, factor_lambda=1.0, pprint=False
    ):
        """
        Initialize hoc-objects to simulate the neuron model implemented by this
        tree.

        Parameters
        ----------
        dt: float (default is ``.025`` ms)
            Timestep of the simulation
        t_calibrate: float (default ``0.`` ms)
            The calibration time; time model runs without input to reach its
            equilibrium state before the true simulation starts
        v_init: float (default ``-75.`` mV)
            The initial voltage at which the model is initialized
        factor_lambda: float or int (default 1.)
            If int, the number of segments per section. If float, multiplies the
            number of segments given by the standard lambda rule (Carnevale, 2004)
            to give the number of compartments simulated (default value 1. gives
            the number given by the lambda rule)
        pprint: bool (default ``False``)
            Whether or not to print info on the NEURON model's creation
        """
        # simulation control
        self.sim_control = {
            'dt': dt,
            't_calibrate': t_calibrate,
            't_max': t_max,
            't_sim': t_calibrate + t_max,
        }
        self.model_name = model_name
        self.pre_dummies = []
        self.syn_locs = []
        self.syn_models = []
        
        # make the Jaxley channel current mechanisms in this model available
        jx_channels = {
            chan: eval(f"JX_MECH['{model_name}'].{chan}()") for chan in self.channel_storage
        }
        jx_channels['L'] = Leak()

        # create Jaxley branches for all nodes
        branches = [node._make_branch(jx_channels) for node in self]

        # create index map for mapping branches to parents
        self.index_map = {node.index: ii for ii, node in enumerate(self)}
        parents = [self.index_map[node.parent_node.index] if not self.is_root(node) else -1 for node in self]

        self.cell = jx.Cell(branches, parents)

        return self.cell
    
    def delete_model(self):
        """
        Deletes all Jaxley objects created to simulate the model
        implemented by this tree.
        """
        self.cell = None
        self.pre_dummies = []
        self.syn_locs = []
        self.syn_models = []

    def _append_synapse(self, loc, synapse):
        self.syn_locs.append(MorphLoc(loc, self))
        self.syn_models.append(synapse)
        self.pre_dummies.append(None)
    
    def add_AMPA_synapse(self, loc, tau_r_AMPA=.2, tau_d_AMPA=3., e_r_AMPA=0.):
        synapse = JX_MECH[f'{self.model_name}'].AMPASynapse(tau_r_AMPA=tau_r_AMPA, tau_d_AMPA=tau_d_AMPA, e_r_AMPA=e_r_AMPA)
        self._append_synapse(loc, synapse)

    def add_GABA_synapse(self, loc, tau_r_GABA=.2, tau_d_GABA=10., e_r_GABA=-80.):
        synapse = JX_MECH[f'{self.model_name}'].GABASynapse(tau_r_GABA=tau_r_GABA, tau_d_GABA=tau_d_GABA, e_r_GABA=e_r_GABA)
        self._append_synapse(loc, synapse)

    def set_spiketrain(self, syn_index, syn_weight, spike_times):
        spike_times = jnp.array(spike_times) + self.sim_control['t_calibrate']
        # create the pre synaptic dummy voltage representing weighted spikes
        spks_v = jnp.zeros(int(self.sim_control['t_sim'] / self.sim_control['dt']) + 1)
        spks_v = spks_v.at[(spike_times / self.sim_control['dt']).astype(int)].set(syn_weight)
        # store the dummy cell and its 'spike' voltage
        self.pre_dummies[syn_index] = (jx.Cell(), spks_v)

    def run(self):
        # if there are synapses that don't receive a spiketrain, create empty ones
        for ii, pd in enumerate(self.pre_dummies):
            if pd is None:
                self.set_spiketrain(ii, 0.0, [])

        # create the network consisting of the pre dummies and the cell
        net = jx.Network(
            [
                pre_dummy[0] for pre_dummy in self.pre_dummies
            ] + [
                self.cell
            ]
        )
        # connect the synapses
        cell_id = len(self.pre_dummies)
        for ii, (loc, syn) in enumerate(zip(self.syn_locs, self.syn_models)):
            # post cell synapse at correct location
            post = net.cell(cell_id).branch(
                self.index_map[loc['node']]
            ).loc(
                loc['x']
            )
            # clamp the pre dummy
            net.cell(ii).clamp("v", self.pre_dummies[ii][1])
            # select correct pre dummy
            pre = net.cell(ii).branch(0).loc(0.0)
            jx.connect(pre, post, syn)

        net.delete_recordings()
        net.cell(cell_id).branch(0).loc(0.0).record()

        current = jx.step_current(
            i_delay=10.+self.sim_control['t_calibrate'], i_dur=10.0, i_amp=1.0, 
            delta_t=self.sim_control['dt'], 
            t_max=self.sim_control['t_sim'],
        )
        net.cell(cell_id).branch(0).loc(0.5).stimulate(current)

        res_raw = jx.integrate(net, delta_t=self.sim_control['dt'])
        
        i0 = int(self.sim_control['t_calibrate'] / self.sim_control['dt'])
        # breakpoint()
        res = {
            't': jnp.arange(0, res_raw[0,i0:].shape[0]) * self.sim_control['dt'],
            'v_m': res_raw[:,i0:],
        }
        return res 
    

class JaxleyCompartmentNode(JaxleySimNode):
    """
    Subclass of `NeuronSimNode` that defines a cylinder with fake geometry in
    NEURON to result in the effective simulation as a single compartment.
    """

    def __init__(self, index):
        super().__init__(index)

    def get_child_nodes(self, skip_inds=[]):
        return super().get_child_nodes(skip_inds=skip_inds)

    def _make_branch(self, pprint=False):
        return super()._make_branch(dx_max=None, factorlambda=1, pprint=False)


class JaxleyCompartmentTree(JaxleySimTree):
    """
    Creates a `neat.NeuronCompartmentTree` to simulate reduced compartmentment
    models from a `neat.CompartmentTree`.

    Parameters
    ----------
    ctree: `neat.CompartmentTree`
        The tree containing the parameters of the reduced compartmental model
        to be simulated
    fake_c_m: float
        Fake value for the membrance capacitance density, rescales cylinder
        surface
    fake_r_a: float
        Fake value for the axial resistance, rescales cylinder length

    Attributes
    ----------
    equivalent_locs: list of tuples
        'Fake' locations corresponding to each compartment, which are
        used to insert hoc point process at the compartments using
        the same functions definitions as for as for a morphological
        `neat.NeuronSimTree`.

    Notes
    -----
    - Note that this class inherits from `neat.NeuronSimTree` and *not* from
    `neat.CompartmentTree`. This is because NEAT defines a fake morphology to
    implement the compartment model in NEURON, and also to reuse the functionality
    implemented by `neat.NeuronSimTree`. Any function that is not explicitly
    redefined from `neat.NeuronSimTree` can be called in the same way for this
    compartment model.
    - Locations to this class can be provided either as fake morphology locations
    -- i.e. a tuple `(node.index, x-location in [0,1])` -- where the value of the
    x-location is ignored since the nodes here are single compartments, as in
    the `neat.CompartmentTree`, and not cylinders, as in `neat.MorphTree` or
    subclasses, or as location indices, where the index corresponds to the location
    in the original list of locations from which the `neat.CompartmentTree` was
    derived.
    """

    def __init__(self, ctree, fake_c_m=1.0, fake_r_a=100.0 * 1e-6, method=2):

        try:
            assert issubclass(ctree.__class__, CompartmentTree)
        except AssertionError as e:
            raise ValueError(
                "`neat.NeuronCompartmentTree` can only be instantiated "
                "from a `neat.CompartmentTree` or derived class"
            )
        super().__init__(ctree, types=[1, 3, 4])
        self.equivalent_locs = ctree.get_equivalent_locs()
        self._create_reduced_jaxley_model(
            ctree,
            fake_c_m=fake_c_m,
            fake_r_a=fake_r_a,
        )

    def _create_reduced_jaxley_model(self, ctree, fake_c_m=1.0, fake_r_a=100.0 * 1e-6):
        arg1, arg2 = ctree.compute_fake_geometry(
            fake_c_m=fake_c_m,
            fake_r_a=fake_r_a,
            factor_r_a=1e-6,
            delta=1e-10,
            method=2,
        )
        lengths = arg1
        radii = arg2
        surfaces = 2.0 * np.pi * radii * lengths
        for ii, comp_node in enumerate(ctree):
            sim_node = self.__getitem__(comp_node.index, skip_inds=[])
            if self.is_root(sim_node):
                sim_node.set_p3d(np.array([0.0, 0.0, 0.0]), radii[ii] * 1e4, 1)
            else:
                sim_node.set_p3d(
                    np.array(
                        [sim_node.parent_node.xyz[0] + lengths[ii] * 1e4, 0.0, 0.0]
                    ),
                    radii[ii] * 1e4,
                    3,
                )

        # fill the tree with the currents
        for ii, sim_node in enumerate(self):
            comp_node = ctree[ii]
            sim_node.currents = {
                chan: [g / surfaces[comp_node.index], e]
                for chan, (g, e) in comp_node.currents.items()
            }
            sim_node.concmechs = copy.deepcopy(comp_node.concmechs)
            for concmech in sim_node.concmechs.values():
                concmech.gamma *= surfaces[comp_node.index] * 1e6
            sim_node.c_m = fake_c_m
            sim_node.r_a = fake_r_a
            sim_node.R = radii[comp_node.index] * 1e4  # convert to [um]
            sim_node.L = lengths[comp_node.index] * 1e4  # convert to [um]
