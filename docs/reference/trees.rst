**************
Abstract Trees
**************

.. automodule:: neat.trees
.. currentmodule:: neat


Basic tree
==========

.. autoclass:: neat.STree

.. autosummary::
   :toctree: generated/

   STree.__getitem__
   STree.__len__
   STree.__iter__
   STree.__str__
   STree.__copy__
   STree.__repr__
   STree.__hash__
   STree.unique_hash
   STree.check_ordered
   STree.get_nodes
   STree.nodes
   STree.gather_nodes
   STree.get_leafs
   STree.leafs
   STree.is_leaf
   STree.root
   STree.is_root
   STree.create_corresponding_node
   STree.add_node_with_parent_from_index
   STree.add_node_with_parent
   STree.soft_remove_node
   STree.remove_node
   STree.remove_single_node
   STree.insert_node
   STree.reset_indices
   STree.get_sub_tree
   STree.depth_of_node
   STree.degree_of_node
   STree.order_of_node
   STree.path_to_root
   STree.path_between_nodes
   STree.path_between_nodes_depth_first
   STree.get_nodes_in_subtree
   STree.sister_leafs
   STree.find_bifurcation_node_to_root
   STree.find_bifurcation_node_from_root
   STree.find_in_between_bifurcation_nodes
   STree.get_nearest_neighbours


.. autoclass:: neat.SNode
   :toctree: generated/

   SNode.__getitem__
   SNode.__setitem__
   SNode.make_empty
   SNode.remove_child

Compartment Tree
================

.. autoclass:: neat.CompartmentTree

.. autosummary::
   :toctree: generated/

   CompartmentTree.create_corresponding_node
   CompartmentTree.get_nodes_from_loc_idxs
   CompartmentTree.permute_to_tree_idxs
   CompartmentTree.permute_to_locs_idxs
   CompartmentTree.get_equivalent_locs
   CompartmentTree.add_channel_current
   CompartmentTree.add_conc_mech
   CompartmentTree.set_expansion_points
   CompartmentTree.remove_expansion_points
   CompartmentTree.set_e_eq
   CompartmentTree.get_e_eq
   CompartmentTree.set_conc_eq
   CompartmentTree.get_conc_eq
   CompartmentTree.fit_e_leak
   CompartmentTree.calc_impedance_matrix
   CompartmentTree.calc_impulse_response_matrix
   CompartmentTree.calc_conductance_matrix
   CompartmentTree.calc_system_matrix
   CompartmentTree.calc_eigenvalues
   CompartmentTree.compute_gmc
   CompartmentTree.compute_g_channels
   CompartmentTree.compute_g_single_channel
   CompartmentTree.compute_c
   CompartmentTree.reset_fit_data
   CompartmentTree.run_fit
   CompartmentTree.compute_fake_geometry
   CompartmentTree.plot_dendrogram

.. autoclass:: neat.CompartmentNode
   :toctree: generated/

   CompartmentNode.set_conc_eq
   CompartmentNode.set_expansion_point
   CompartmentNode.calc_membrane_conductance_terms
   CompartmentNode.calc_membrane_concentration_terms
   CompartmentNode.calc_g_tot
   CompartmentNode.calc_i_tot
   CompartmentNode.calc_linear_statevar_terms


Neural Evaluation Tree
======================

.. autoclass:: neat.NET

.. autosummary::
   :toctree: generated/

   NET.create_corresponding_node
   NET.get_loc_idxs
   NET.get_leaf_loc_node
   NET.set_new_loc_idxs
   NET.get_reduced_tree
   NET.calc_total_impedance
   NET.calc_total_kernel
   NET.calc_i_z
   NET.calc_i_z_matrix
   NET.calc_impedance_matrix
   NET.calc_compartmentalization
   NET.compute_cond_rescale
   NET.improve_input_resistance
   NET.plot_dendrogram


.. autoclass:: neat.NETNode


.. autoclass:: neat.Kernel

.. autosummary::
   :toctree: generated/

   Kernel.k_bar
   Kernel.t
   Kernel.ft
   Kernel.diff
   Kernel.fit_c


*******************
Morphological Trees
*******************


Morphology Tree
===============

.. autoclass:: neat.MorphTree

Read a morphology from an SWC file

.. autosummary::
   :toctree: generated/

   MorphTree.read_swc_tree_from_file
   MorphTree.determine_soma_type

Note that in the '.swc' format,
nodes 2 and 3 contain extra information on the soma geometry. By default, these
nodes are skipped in the `neat.MorphTree` iteration and getitem functions.

.. autosummary::
   :toctree: generated/

   MorphTree.__getitem__
   MorphTree.__iter__
   MorphTree.__copy__
   MorphTree.reset_indices

Get specific nodes or sets of nodes from the tree. 

.. autosummary::
   :toctree: generated/

   MorphTree.create_corresponding_node
   MorphTree.root
   MorphTree.get_nodes
   MorphTree.nodes
   MorphTree.get_leafs
   MorphTree.leafs
   MorphTree.get_nodes_in_basal_subtree
   MorphTree.get_nodes_in_apical_subtree
   MorphTree.get_nodes_in_axonal_subtree
   MorphTree.convert_node_arg_to_nodes

Relating to the computational tree.

.. autosummary::
   :toctree: generated/

   MorphTree.set_default_tree
   MorphTree.as_computational_tree
   MorphTree.as_original_tree
   MorphTree.check_computational_tree_active
   MorphTree.set_comp_tree
   MorphTree._evaluate_comp_criteria
   MorphTree.remove_comp_tree

Storing locations, interacting with stored locations and distributing
locations

.. autosummary::
   :toctree: generated/

   MorphTree.convert_loc_arg_to_locs
   MorphTree.store_locs
   MorphTree.add_loc
   MorphTree.clear_locs
   MorphTree.remove_locs
   MorphTree._try_name
   MorphTree.get_locs
   MorphTree.get_node_indices
   MorphTree.get_x_coords
   MorphTree.get_loc_idxs_on_node
   MorphTree.get_loc_idxs_on_nodes
   MorphTree.get_loc_idxs_on_path
   MorphTree.get_nearest_loc_idxs
   MorphTree.get_nearest_neighbour_loc_idxs
   MorphTree.get_leaf_loc_idxs
   MorphTree.distances_to_soma
   MorphTree.distances_to_bifurcation
   MorphTree.distribute_locs_at_d2s
   MorphTree.distribute_locs_uniform
   MorphTree.distribute_locs_random
   MorphTree.distribute_locs_finite_diff
   MorphTree.extend_with_bifurcation_locs
   MorphTree.unique_locs
   MorphTree.path_length

Plotting on a 1D axis.

.. autosummary::
   :toctree: generated/

   MorphTree.make_x_axis
   MorphTree.set_node_colors
   MorphTree.get_x_values
   MorphTree.plot_1d
   MorphTree.plot_true_d2s
   MorphTree.color_x_axis

Plotting the morphology in 2D.

.. autosummary::
   :toctree: generated/

   MorphTree.plot_2d_morphology
   MorphTree.plot_morphology_interactive

Creating new trees from the existing tree.

.. autosummary::
   :toctree: generated/

   MorphTree.find_common_root
   MorphTree.create_new_tree
   MorphTree.create_compartment_tree
   MorphTree.__copy__


.. autoclass:: neat.MorphNode

.. autosummary::
   :toctree: generated/

   MorphNode.set_p3d
   MorphNode.child_nodes

.. autoclass:: neat.MorphLoc


Physiology Tree
===============

.. autoclass:: neat.PhysTree

.. autosummary::
   :toctree: generated/

   PhysTree.as_passive_membrane
   PhysTree.set_v_ep
   PhysTree.set_conc_ep
   PhysTree.set_physiology
   PhysTree.set_leak_current
   PhysTree.add_channel_current
   PhysTree.add_conc_mech
   PhysTree.fit_leak_current
   PhysTree.get_channels_in_tree

Pertaining to the computational tree create, which for the `neat.PhysTree`
also takes the physiological parameters into account.

.. autosummary::
   :toctree: generated/
   PhysTree._evaluate_comp_criteria

NEAT implements the functionality to construct a `neat.CompartmentTree` 
whose parameters represent the 2nd order finite difference approximation 
to the cable equation.

.. autosummary::
   :toctree: generated/
   PhysTree.create_new_tree
   PhysTree.create_finite_difference_tree

.. autoclass:: neat.PhysNode

.. autosummary::
   :toctree: generated/
   PhysNode.calc_g_tot
   PhysNode.calc_i_tot
   PhysNode.as_passive_membrane

Separation of Variables Tree
============================

.. autoclass:: neat.SOVTree

.. autosummary::
   :toctree: generated/

   SOVTree.create_corresponding_node
   SOVTree.calc_sov_equations
   SOVTree.get_sov_matrices
   SOVTree.get_mode_importance
   SOVTree.get_important_modes
   SOVTree.get_kernels
   SOVTree.calc_zf
   SOVTree.calc_zt
   SOVTree.calc_impedance_matrix
   SOVTree.calc_impulse_response_matrix
   SOVTree.construct_net
   SOVTree.compute_lin_terms

.. autoclass:: neat.SOVNode


Greens Tree
===========

.. autoclass:: neat.GreensTree

.. autosummary::
   :toctree: generated/

   GreensTree.create_corresponding_node
   GreensTree.remove_expansion_points
   GreensTree.set_impedance
   GreensTree.calc_zf
   GreensTree.calc_impedance_matrix
   GreensTree.calc_channel_response_f
   GreensTree.calc_channel_response_matrix

.. autoclass:: neat.GreensNode

.. autosummary::
   :toctree: generated/

    GreensNode.set_expansion_point


Greens Tree Time
================

.. autoclass:: neat.GreensTreeTime

.. autosummary::
   :toctree: generated/

   GreensTreeTime.set_impedance
   GreensTreeTime.calc_zt
   GreensTreeTime.calc_impulse_response_matrix
   GreensTreeTime.calc_channel_response_t
   GreensTreeTime.calc_channel_response_matrix


**********
CacheTrees
**********

Compute equilibrium potentials and concentrations
=================================================

.. autoclass:: neat.EquilibriumTree

.. autosummary::
   :toctree: generated/

   EquilibriumTree.calc_e_eq
   EquilibriumTree.set_e_eq


Caching the Greens function and separation of variables expansion
==================================================================

.. autoclass:: neat.CachedTree

.. autosummary::
   :toctree: generated/

   CachedTree.get_cache_defaults
   CachedTree.set_cache_params
   CachedTree.get_cache_params
   CachedTree.maybe_execute_funcs

.. autoclass:: neat.CachedGreensTree

.. autosummary::
   :toctree: generated/

   CachedGreensTree.set_impedances_in_tree
   CachedGreensTree.calc_net_steadystate

.. autoclass:: neat.CachedGreensTreeTime

.. autosummary::
   :toctree: generated/

   CachedGreensTreeTime.set_impedances_in_tree

.. autoclass:: neat.CachedSOVTree

.. autosummary::
   :toctree: generated/

   CachedSOVTree.set_sov_in_tree


Fitting reduced models
======================

.. autoclass:: neat.CompartmentFitter

To implement the default methodology.

.. autosummary::
   :toctree: generated/
   CompartmentFitter.fit_model

To get stored fit results and associated location lists

.. autosummary::
   :toctree: generated/

   CompartmentFitter.convert_fit_arg
   CompartmentFitter.remove_fit

To check the faithfulness of the passive reduction, the following functions
implement visualization of impedance kernels and SOV quantities.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.check_passive
   CompartmentFitter.get_kernels
   CompartmentFitter.plot_kernels
   CompartmentFitter.plot_sov

Individual fit functions.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.set_ctree
   CompartmentFitter.create_tree_gf
   CompartmentFitter.create_tree_sov
   CompartmentFitter.fit_leak_only
   CompartmentFitter.fit_passive
   CompartmentFitter.fit_channels
   CompartmentFitter.fit_concentration
   CompartmentFitter.fit_capacitance
   CompartmentFitter.fit_e_eq

`neat.CompartmentFitter` can also compute conductance rescale values for synapses
at sites on the original morphology, when they are shifted to compartment locations
on the reduced morphology. For this, the average conductances of each synapses need
to be known.

.. autosummary::
   :toctree: generated/

   CompartmentFitter.get_net
   CompartmentFitter.recalc_impedance_matrix
   CompartmentFitter.assign_locs_to_comps
   CompartmentFitter.fit_syn_rescale


**********************************
Simulating full and reduced models
**********************************


Simulate full models in NEURON
==============================

.. autoclass:: neat.NeuronSimTree

.. autosummary::
   :toctree: generated/

   NeuronSimTree.create_corresponding_node
   NeuronSimTree.set_simulation_parameters
   NeuronSimTree.init_model
   NeuronSimTree.delete_model
   NeuronSimTree.set_rec_locs
   NeuronSimTree.add_shunt
   NeuronSimTree.add_double_exp_current
   NeuronSimTree.add_exp_synapse
   NeuronSimTree.add_double_exp_synapse
   NeuronSimTree.add_nmda_synapse
   NeuronSimTree.add_double_exp_nmda_synapse
   NeuronSimTree.add_i_clamp
   NeuronSimTree.add_sin_clamp
   NeuronSimTree.add_ou_clamp
   NeuronSimTree.add_ou_conductance
   NeuronSimTree.add_ou_reversal
   NeuronSimTree.add_v_clamp
   NeuronSimTree.set_spiketrain
   NeuronSimTree.run
   NeuronSimTree.calc_e_eq
   NeuronSimTree.calc_impedance_matrix
   NeuronSimTree.calc_zt
   NeuronSimTree.calc_impulse_response_matrix

.. autoclass:: neat.NeuronSimNode


Simulate reduced compartmental models in NEURON
===============================================

.. autoclass:: neat.NeuronCompartmentTree

.. autosummary::
   :toctree: generated/

   NeuronCompartmentTree.create_corresponding_node
   NeuronCompartmentTree.set_rec_locs
   NeuronCompartmentTree.add_shunt
   NeuronCompartmentTree.add_double_exp_current
   NeuronCompartmentTree.add_exp_synapse
   NeuronCompartmentTree.add_double_exp_synapse
   NeuronCompartmentTree.add_nmda_synapse
   NeuronCompartmentTree.add_double_exp_nmda_synapse
   NeuronCompartmentTree.add_i_clamp
   NeuronCompartmentTree.add_sin_clamp
   NeuronCompartmentTree.add_ou_clamp
   NeuronCompartmentTree.add_ou_conductance
   NeuronCompartmentTree.add_ou_reversal
   NeuronCompartmentTree.add_v_clamp

.. autoclass:: neat.NeuronCompartmentNode


Simulate reduced compartmental models in Brian2
===============================================

.. autoclass:: neat.Brian2CompartmentTree

.. autosummary::
   :toctree: generated/

   Brian2CompartmentTree.create_corresponding_node
   Brian2CompartmentTree.get_compartment_index
   Brian2CompartmentTree.morpho_access_string
   Brian2CompartmentTree.add_double_exp_synapse
   Brian2CompartmentTree.get_on_pre
   Brian2CompartmentTree.initialise_at_veq
   Brian2CompartmentTree.init_model

.. autoclass:: neat.Brian2CompartmentNode

.. autosummary::
   :toctree: generated/

   Brian2CompartmentNode.fake_surface


Simulate reduced compartmental models in NEST
=============================================

.. autoclass:: neat.NestCompartmentTree

.. autosummary::
   :toctree: generated/

   NestCompartmentTree.create_corresponding_node
   NestCompartmentTree.init_model

.. autoclass:: neat.NestCompartmentNode


Neural evaluation tree simulator
================================

.. autoclass:: neat.netsim.NETSim


*************
Miscellaneous
*************

Defining concentration mechanisms
=================================

.. autoclass:: neat.ExpConcMech

.. autosummary::
   :toctree: generated/

   ExpConcMech.items
   ExpConcMech.compute_linear
   ExpConcMech.compute_lin
   ExpConcMech.compute_lin_tau_fit
   ExpConcMech.write_nestml_blocks


Defining ion channels
=====================

.. autoclass:: neat.IonChannel

.. autosummary::
   :toctree: generated/

   IonChannel.set_default_params
   IonChannel.ordered_statevars
   IonChannel.compute_p_open
   IonChannel.compute_derivatives
   IonChannel.compute_derivativesConc
   IonChannel.compute_varinf
   IonChannel.compute_tauinf
   IonChannel.compute_lin_statevar_response
   IonChannel.compute_linear
   IonChannel.compute_linear_conc
   IonChannel.compute_lin_sum
   IonChannel.compute_lin_conc
   IonChannel.write_mod_file
   IonChannel.write_cpp_code


Compute Fourier transforms
===========================

.. autoclass:: neat.FourierQuadrature

.. autosummary::
   :toctree: generated/

   FourierQuadrature.__call__
   FourierQuadrature.ft
   FourierQuadrature.ft_inv

.. autoclass:: neat.FourierTools

.. autosummary::
   :toctree: generated/

   FourierTools.inverse_fourier
