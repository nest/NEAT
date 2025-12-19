import numpy as np
import jaxley as jx

import os

from neat import PhysTree, JaxleySimTree, JaxleyCompartmentTree, CompartmentFitter

import channelcollection_for_tests as channelcollection
import channel_installer

channel_installer.load_or_install_neuron_test_channels()
channel_installer.load_or_install_jaxley_test_channels()


MORPHOLOGIES_PATH_PREFIX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_morphologies")
)


class TestJaxley:
    def load_ball(self):
        """
        Load point neuron model
        """
        self.tree = PhysTree(os.path.join(MORPHOLOGIES_PATH_PREFIX, "ball.swc"))
        # capacitance and axial resistance
        self.tree.set_physiology(0.8, 100.0 / 1e6)
        # ion channels
        self.k_chan = channelcollection.Kv3_1()
        self.tree.add_channel_current(self.k_chan, 0.766 * 1e6, -85.0)
        self.na_chan = channelcollection.NaTa_t()
        self.tree.add_channel_current(self.na_chan, 1.71 * 1e6, 50.0)
        # fit leak current
        self.tree.fit_leak_current(-75.0, 10.0)
        # set equilibirum potententials
        self.tree.set_v_ep(-75.0)
        # set computational tree
        self.tree.set_comp_tree()

        cfit = CompartmentFitter(self.tree, save_cache=False, recompute_cache=True)
        self.ctree, _ = cfit.fit_model([(1, 0.5)])


    def create_jaxley_model(self):
        self.load_ball()

        jt = JaxleySimTree(self.tree)
        jcell = jt.init_model("multichannel_test", t_max=1000)

        jt.add_AMPA_synapse((1,.5))
        jt.set_spiketrain(0, 1., [200.])
        res = jt.run()

        import matplotlib.pyplot as pl
        pl.plot(res.T)
        pl.show()

        jct = JaxleyCompartmentTree(self.ctree)
        jcompcell = jt.init_model("multichannel_test")

    # def test_jaxley_channels(self):
    #     self.load_ball()

    #     channelstr_k = self.k_chan.write_jaxley_code()
    #     channelstr_na = self.na_chan.write_jaxley_code()
    #     with open("jtest", "w") as file:
    #         file.write(channelstr_k + "\n\n\n")
    #         file.write(channelstr_na)
        


if __name__ == "__main__":
    tjx = TestJaxley()
    tjx.create_jaxley_model()
    # tjx.test_jaxley_channels()
