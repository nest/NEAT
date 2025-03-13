from neat import IonChannel


class CaE(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "ca"
        self.p_open = "m*h"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "2.6    / (1 + exp((v+7.)  / (-8.) ))"
        self.beta["m"] = "0.18   / (1 + exp((v+26.) / 4.    ))"
        self.alpha["h"] = "0.0025 / (1 + exp((v+32.) / 8.    ))"
        self.beta["h"] = "0.19   / (1 + exp((v+42.) / (-10.)))"


class CaP(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "ca"
        self.p_open = "m*h"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "8.5    / (1. + exp(-(v - 8. ) / 12.5))"  # 1/ms
        self.beta["m"] = "35.    / (1. + exp( (v + 74.) / 14.5))"  # 1/ms
        self.alpha["h"] = "0.0015 / (1. + exp( (v + 29.) / 8.  ))"  # 1/ms
        self.beta["h"] = "0.0055 / (1. + exp(-(v + 23.) / 8.  ))"  # 1/ms


class CaP2(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "ca"
        self.p_open = "m"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "8.5 / (1. + exp((v-8. ) / (-12.5)))"
        self.beta["m"] = "35. / (1. + exp((v+74.) / 14.5   ))"


class CaT(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "ca"
        self.p_open = "m*h"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "2.6    / (1. + exp((v+21.) / (-8.) ))"
        self.beta["m"] = "0.18   / (1. + exp((v+40.) / 4.    ))"
        self.alpha["h"] = "0.0025 / (1. + exp((v+40.) / 8.    ))"
        self.beta["h"] = "0.19   / (1. + exp((v+50.) / (-10.)))"


class NaF(IonChannel):
    def define(self):
        """
        Purkinje Cell (Miyasho et al., 2001)
        """
        self.ion = "na"
        self.p_open = "m**3 * h"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "35.   / exp((v+5.)  / (-10.))"
        self.beta["m"] = "7.    / exp((v+65.) / 20.   )"
        self.alpha["h"] = "0.225 / (1. + exp((v+80.) /10.))"
        self.beta["h"] = "7.5   / exp((v-3.)  / (-18.))"


class NaP(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "na"
        self.p_open = "m**3"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "200. / (1. + exp((v-18.) / (-16.)))"
        self.beta["m"] = "25.  / (1. + exp((v+58.) / 8.    ))"


class K23(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "k"
        self.concentrations = ["ca"]
        self.p_open = "m * z**2"
        # activation functions
        self.varinf, self.tauinf = {}, {}
        self.varinf["z"] = "1. / (1. + 0.020 / ca)"
        self.tauinf["z"] = "10."
        self.varinf["m"] = "25. / (25. + (0.075 / exp((v + 5.) / 10.)))"
        self.tauinf["m"] = "1.  / (25. + (0.075 / exp((v + 5.) / 10.)))"


class KC3(IonChannel):
    def define(self):
        """
        Purkinje Cell (Miyasho et al., 2001)
        """
        self.ion = "k"
        self.concentrations = ["ca"]
        self.p_open = "m * z**2"
        # activation functions
        self.varinf, self.tauinf = {}, {}
        self.varinf["z"] = "1. / (1. + 0.4 / ca)"
        self.tauinf["z"] = "10."
        self.varinf["m"] = "7.5 / (7.5 + (0.11 / exp((v - 35.) / 14.9)))"
        self.tauinf["m"] = "1.  / (7.5 + (0.11 / exp((v - 35.) / 14.9)))"


class KA(IonChannel):
    def define(self):
        """
        Purkinje Cell (Miyasho et al., 2001)
        """
        self.ion = "k"
        self.p_open = "m**4 * h"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "1.4    / (1. + exp((v+27.) / (-12.)))"
        self.beta["m"] = "0.49   / (1. + exp((v+30.) / 4.    ))"
        self.alpha["h"] = "0.0175 / (1. + exp((v+50.) / 8.    ))"
        self.beta["h"] = "1.3    / (1. + exp((v+13.) / (-10.)))"


class KD(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "k"
        self.p_open = "m * h"
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha["m"] = "8.5    / (1. + exp((v+17.) / (-12.5)))"
        self.beta["m"] = "35.    / (1. + exp((v+99.) / 14.5   ))"
        self.alpha["h"] = "0.0015 / (1. + exp((v+89.) / 8.     ))"
        self.beta["h"] = "0.0055 / (1. + exp((v+83.) / (-8.)  ))"


class KM(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "k"
        self.p_open = "m"
        # activation functions
        self.varinf = {"m": "1.0 / (1. + exp(-(v+35.) / 10.))"}
        self.tauinf = {
            "m": "1.  / (3.3 * (exp((v+35.) / 40. ) + exp(-(v+35.) / 20.)) / 200.)"
        }  # ms


class Kh(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "k"
        self.p_open = ".8 * m + .2 * n"
        # activation functions
        expr = "1. / (1. + exp((v+78.) / 7.))"
        self.varinf = {"m": expr, "n": expr}
        self.tauinf = {"m": "38.", "n": "319."}


class Khh(IonChannel):
    """
    Purkinje Cell (Miyasho et al., 2001)
    """

    def define(self):
        self.ion = "k"
        self.p_open = "n**4"
        # activation functions
        self.alpha = {"n": ".01 * -(v + 55.1234) / (exp(-(v + 55.1234) / 10.) - 1.)"}
        self.beta = {"n": ".125 * exp(-(v + 65.) / 80.)"}
