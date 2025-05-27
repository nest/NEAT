import sympy as sp

from neat import IonChannel


class h_HAY(IonChannel):
    """
    Hcn channel from (Kole, Hallermann and Stuart, 2006)
    Used in (Hay, 2011)
    """

    def define(self):
        self.p_open = "m"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "0.001 * 6.43 * (v + 154.9) / (exp((v + 154.9) / 11.9) - 1.)"
        self.beta["m"] = "0.001 * 193. * exp(v / 33.1)"

        self.e = -43.0


class Na_Ta(IonChannel):
    """
    (Colbert and Pan, 2002)

    Used in (Hay, 2011)
    """

    def define(self):
        self.ion = "na"
        self.p_open = "h * m ** 3"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "0.182 * (v + 38.) / (1. - exp(-(v + 38.) / 6.))"  # 1/ms
        self.beta["m"] = "-0.124 * (v + 38.) / (1. - exp( (v + 38.) / 6.))"  # 1/ms
        self.alpha["h"] = "-0.015 * (v + 66.) / (1. - exp( (v + 66.) / 6.))"  # 1/ms
        self.beta["h"] = "0.015 * (v + 66.) / (1. - exp(-(v + 66.) / 6.))"  # 1/ms

        self.q10 = 2.95


class Na_p(IonChannel):
    """
    Derived by (Hay, 2011) from (Magistretti and Alonso, 1999)
    Used in (Hay, 2011)
    """

    def define(self):
        self.ion = "na"
        self.p_open = "m**3 * h"
        # activation functions
        self.varinf = {
            "m": "1. / (1. + exp(-(v + 52.6) / 4.6))",
            "h": "1. / (1. + exp( (v + 48.8) / 10.))",
        }
        # non-standard time-scale definition
        v = sp.symbols("v")
        alpha, beta = {}, {}
        alpha["m"] = (
            0.182 * (v + 38.0) / (1.0 - sp.exp(-(v + 38.0) / 6.0, evaluate=False))
        )  # 1/ms
        beta["m"] = (
            -0.124 * (v + 38.0) / (1.0 - sp.exp((v + 38.0) / 6.0, evaluate=False))
        )  # 1/ms
        alpha["h"] = (
            -2.88e-6 * (v + 17.0) / (1.0 - sp.exp((v + 17.0) / 4.63, evaluate=False))
        )  # 1/ms
        beta["h"] = (
            6.94e-6 * (v + 64.4) / (1.0 - sp.exp(-(v + 64.4) / 2.63, evaluate=False))
        )  # 1/ms
        self.tauinf = {
            "m": 6.0 / (alpha["m"] + beta["m"]),
            "h": 1.0 / (alpha["h"] + beta["h"]),
        }

        self.q10 = 2.95


class Kv3_1(IonChannel):
    """
    Shaw-related potassium channel (Rettig et al., 1992)
    Used in (Hay et al., 2011)
    """

    def define(self):
        self.ion = "k"
        self.p_open = "m"
        # activation functions
        self.varinf = {"m": "1. / (1. + exp(-(v - 18.70) /  9.70))"}
        self.tauinf = {"m": "4. / (1. + exp(-(v + 46.56) / 44.14))"}  # ms


class Kpst(IonChannel):
    """
    Persistent Potassium channel (Korngreen and Sakmann, 2000)
    Used in (Hay, 2011)
    """

    def define(self):
        self.ion = "k"
        self.p_open = "m**2 * h"
        # activation functions
        self.varinf = {
            "m": "1. / (1. + exp(-(v + 11.) / 12.))",
            "h": "1. / (1. + exp( (v + 64.) / 11.))",
        }
        self.tauinf = {
            "m": "(3.04 + 17.3 * exp(-((v + 60.) / 15.9)**2) + 25.2 * exp(-((v + 60.) / 57.4)**2)) / 2.95",
            "h": "(360. + (1010. + 24. * (v + 65.)) * exp(-((v + 85.) / 48.)**2)) / 2.95",
        }  # ms


class Ktst(IonChannel):
    """
    Transient Potassium channel (Korngreen and Sakmann, 2000)
    Used in (Hay, 2011)
    """

    def define(self):
        self.ion = "k"
        self.p_open = "m**2 * h"
        # activation functions
        self.varinf = {
            "m": "1. / (1. + exp(-(v + 10.) / 19.))",
            "h": "1. / (1. + exp( (v + 76.) / 10.))",
        }
        self.tauinf = {
            "m": "(0.34 + 0.92 * exp(-((v + 81.) / 59.)**2)) / 2.95",
            "h": "(8.   + 49.  * exp(-((v + 83.) / 23.)**2)) / 2.95",
        }  # ms


class m(IonChannel):
    """
    M-type potassium current (Adams, 1982)
    Used in (Hay, 2011)
    """

    def define(self):
        self.ion = "k"
        self.p_open = "m"
        # activation functions
        self.alpha = {"m": "3.3e-3 * exp( 2.5 * 0.04 * (v + 35.))"}
        self.beta = {"m": "3.3e-3 * exp(-2.5 * 0.04 * (v + 35.))"}

        self.q10 = 2.95


class SK(IonChannel):
    """
    SK-type calcium-activated potassium current (Kohler et al., 1996)
    used in (Hay et al., 2011)
    """

    def define(self):
        self.ion = "k"
        self.conc = ["ca"]
        self.p_open = "z"
        # activation functions
        self.varinf = {"z": "1. / (1. + (0.00043 / ca)**4.8)"}
        self.tauinf = {"z": "1."}  # ms


class Ca_LVA(IonChannel):
    """
    LVA calcium channel (Avery and Johnston, 1996; tau from Randall, 1997)
    Used in (Hay, 2011)
    """

    def define(self):
        self.ion = "ca"
        self.p_open = "m**2 * h"
        # activation functions
        self.varinf = {
            "m": "1. / (1. + exp(-(v + 40.)/6.))",
            "h": "1. / (1. + exp((v + 90.)/6.4))",
        }
        self.tauinf = {
            "m": " 5. + 20./(1. + exp((v + 35.) / 5.)) / 2.95",
            "h": "20. + 50./(1. + exp((v + 50.) / 7.)) / 2.95",
        }  # ms


class Ca_HVA(IonChannel):
    """
    High voltage-activated calcium channel (Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993)
    Used in (Hay, 2011)
    """

    def define(self):
        self.ion = "ca"
        self.p_open = "m**2 * h"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "-0.055 * (27. + v) / (exp(-(27. + v)/3.8) - 1.)"  # 1/ms
        self.beta["m"] = "0.94 * exp(-(75. + v)/17.)"  # 1/ms
        self.alpha["h"] = "0.000457 * exp(-(13. + v)/50.)"  # 1/ms
        self.beta["h"] = "0.0065 / (exp(-(v + 15.)/28.) + 1.)"  # 1/ms
