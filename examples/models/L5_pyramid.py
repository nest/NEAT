import numpy as np

from .channels import channels_hay

from neat import PhysTree


def getL5Pyramid():
    """
    Return a minimal model of the L5 pyramid for BAC-firing
    """
    # load the morphology
    phys_tree = PhysTree("models/morphologies/cell1_simplified.swc")

    # set specific membrane capacitance and axial resistance
    phys_tree.set_physiology(
        1.0, 100.0 / 1e6, node_arg=[phys_tree[1]]  # Cm [uF/cm^2]  # Ra[MOhm*cm]
    )
    # set specific membrane capacitance and axial resistance
    phys_tree.set_physiology(
        2.0,  # Cm [uF/cm^2]
        100.0 / 1e6,  # Ra[MOhm*cm]
        node_arg=[n for n in phys_tree if not phys_tree.is_root(n)],
    )

    # channels present in tree
    Kv3_1 = channels_hay.Kv3_1()
    Na_Ta = channels_hay.Na_Ta()
    Ca_LVA = channels_hay.Ca_LVA()
    Ca_HVA = channels_hay.Ca_HVA()
    h_HAY = channels_hay.h_HAY()

    # soma ion channels [uS/cm^2]
    phys_tree.add_channel_current(Kv3_1, 0.766 * 1e6, -85.0, node_arg=[phys_tree[1]])
    phys_tree.add_channel_current(Na_Ta, 1.71 * 1e6, 50.0, node_arg=[phys_tree[1]])
    phys_tree.add_channel_current(Ca_LVA, 0.00432 * 1e6, 50.0, node_arg=[phys_tree[1]])
    phys_tree.add_channel_current(Ca_HVA, 0.000567 * 1e6, 50.0, node_arg=[phys_tree[1]])
    phys_tree.add_channel_current(h_HAY, 0.0002 * 1e6, -45.0, node_arg=[phys_tree[1]])
    phys_tree.set_leak_current(0.0000344 * 1e6, -90.0, node_arg=[phys_tree[1]])

    # basal ion channels [uS/cm^2]
    phys_tree.add_channel_current(h_HAY, 0.0002 * 1e6, -45.0, node_arg="basal")
    phys_tree.set_leak_current(0.0000535 * 1e6, -90.0, node_arg="basal")

    # apical ion channels [uS/cm^2]
    phys_tree.add_channel_current(Kv3_1, 0.000298 * 1e6, -85.0, node_arg="apical")
    phys_tree.add_channel_current(Na_Ta, 0.0211 * 1e6, 50.0, node_arg="apical")
    phys_tree.add_channel_current(
        Ca_LVA,
        lambda x: 0.0198 * 1e6 if (x > 685.0 and x < 885.0) else 0.0198 * 1e-2 * 1e6,
        50.0,
        node_arg="apical",
    )
    phys_tree.add_channel_current(
        Ca_HVA,
        lambda x: (
            0.000437 * 1e6 if (x > 685.0 and x < 885.0) else 0.000437 * 1e-1 * 1e6
        ),
        50.0,
        node_arg="apical",
    )
    phys_tree.add_channel_current(
        h_HAY,
        lambda x: 0.0002 * 1e6 * (-0.8696 + 2.0870 * np.exp(x / 323.0)),
        -45.0,
        node_arg="apical",
    )
    phys_tree.set_leak_current(0.0000447 * 1e6, -90.0, node_arg="apical")

    return phys_tree
