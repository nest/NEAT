from .channels import channels_branco

from neat import PhysTree


def getL23PyramidPas():
    """
    Return a passive model of the L2/3 pyramidal cell
    """
    cm = 1.0  # uF / cm^2
    rm = 10000.0  # Ohm * cm^2
    ri = 150.0  # Ohm * cm
    el = -75.0  # mV

    phys_tree = PhysTree("models/morphologies/L23PyrBranco.swc", types=[1, 2, 3, 4])

    # set specific membrane capacitance and axial resistance
    phys_tree.set_physiology(
        cm,  # Cm [uF/cm^2]
        ri / 1e6,  # Ra[MOhm*cm]
    )

    # passive membrane conductance
    phys_tree.set_leak_current(1e6 / 10000.0, el)
    phys_tree.set_leak_current(0.02 * 1e6, el, node_arg="axonal")

    return phys_tree


def getL23PyramidNaK():
    """
    Return a model of the L2/3 pyramidal cell with somatic and basal Na- and
    K-channels
    """
    cm = 1.0  # uF / cm^2
    rm = 10000.0  # Ohm * cm^2
    ri = 150.0  # Ohm * cm
    el = -75.0  # mV
    tadj = 3.21

    phys_tree = PhysTree("models/morphologies/L23PyrBranco.swc")

    phys_tree.set_physiology(
        cm,  # Cm [uF/cm^2]
        ri / 1e6,  # Ra[MOhm*cm]
    )

    # channels
    Na = channels_branco.Na()
    K_v = channels_branco.K_v()

    # somatic channels
    phys_tree.add_channel_current(
        Na, tadj * 1500.0 * 1e2, 60.0, node_arg=[phys_tree[1]]
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        K_v, tadj * 200.0 * 1e2, -90.0, node_arg=[phys_tree[1]]
    )  # pS/um^2 -> uS/cm^2
    phys_tree.set_leak_current(1.0 / rm * 1e6, el, node_arg=[phys_tree[1]])

    # basal channels
    phys_tree.add_channel_current(
        Na, tadj * 40.0 * 1e2, 60.0, node_arg="basal"
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        K_v, tadj * 30.0 * 1e2, -90.0, node_arg="basal"
    )  # pS/um^2 -> uS/cm^2
    phys_tree.set_leak_current(1.0 / rm * 1e6, el, node_arg="basal")

    # passive apical dendrite
    phys_tree.set_leak_current(1.0 / rm * 1e6, el, node_arg="apical")

    return phys_tree


def getL23Pyramid():
    """
    Return a model of the L2/3 pyramidal cell with somatic and basal Na-, K- and
    Ca-channels
    """
    cm = 1.0  # uF / cm^2
    rm = 10000.0  # Ohm * cm^2
    ri = 150.0  # Ohm * cm
    el = -75.0  # mV
    tadj = 3.21

    phys_tree = PhysTree("models/morphologies/L23PyrBranco.swc")

    phys_tree.set_physiology(
        cm,  # Cm [uF/cm^2]
        ri / 1e6,  # Ra[MOhm*cm]
    )

    # channels
    Na = channels_branco.Na()
    K_v = channels_branco.K_v()
    K_m = channels_branco.K_m()
    K_ca = channels_branco.K_ca()
    Ca_H = channels_branco.Ca_H()
    Ca_T = channels_branco.Ca_T()

    # somatic channels
    phys_tree.add_channel_current(
        Na, tadj * 1500.0 * 1e2, 60.0, node_arg=[phys_tree[1]]
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        K_v, tadj * 200.0 * 1e2, -90.0, node_arg=[phys_tree[1]]
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        K_m, tadj * 2.2 * 1e2, -90.0, node_arg=[phys_tree[1]]
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        K_ca, tadj * 2.5 * 1e2, -90.0, node_arg=[phys_tree[1]]
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        Ca_H, tadj * 0.5 * 1e2, 140.0, node_arg=[phys_tree[1]]
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        Ca_T, 0.0003 * 1e6, 60.0, node_arg=[phys_tree[1]]
    )  # mho/cm^2 -> uS/cm^2
    phys_tree.set_leak_current(1.0 / rm * 1e6, el, node_arg=[phys_tree[1]])

    # basal channels
    phys_tree.add_channel_current(
        Na, tadj * 40.0 * 1e2, 60.0, node_arg="basal"
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        K_v, tadj * 30.0 * 1e2, -90.0, node_arg="basal"
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        K_m, tadj * 0.05 * 1e2, -90.0, node_arg="basal"
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        K_ca, tadj * 2.5 * 1e2, -90.0, node_arg="basal"
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        Ca_H, tadj * 0.5 * 1e2, 140.0, node_arg="basal"
    )  # pS/um^2 -> uS/cm^2
    phys_tree.add_channel_current(
        Ca_T, 0.0006 * 1e6, 60.0, node_arg="basal"
    )  # mho/cm^2 -> mho/cm^2
    phys_tree.set_leak_current(1.0 / rm * 1e6, el, node_arg="basal")

    # passive apical dendrite
    phys_tree.set_leak_current(1.0 / rm * 1e6, el, node_arg="apical")

    return phys_tree
