import numpy as np
import sympy as sp

from ionchannels import IonChannel


def sp_exp(x):
    return sp.exp(x, evaluate=False)

# dictionary with suggested reversal potential for each channel
E_REV_DICT = {
                'TestChannel': -23.,
                'h': -43.,
                'h_HAY': -45.,
                'Na_Ta': 50.,
                'Na_p': 50.,
                'Kv3_1': -85.,
                'Kpst': -85.,
                'Ktst': -85.,
                'm': -85.,
                'Ca_LVA': 50.,
                'Ca_HVA': 50.,
                'L': -75.,
             }


class TestChannel(IonChannel):
    '''
    Simple channel to test basic functionality
    '''
    def __init__(self):
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # basic factors to construct channel open probability
        self.factors = np.array([5.,1.])
        self.powers = np.array([[3,3,1],
                                [2,2,1]])
        self.varnames = np.array([['a00', 'a01', 'a02'],
                                  ['a10', 'a11', 'a12']])
        # asomptotic state variable functions
        self.varinf = np.array([[1./(1.+sp_exp((self.sp_v-30.)/100.)), 1./(1.+sp_exp((-self.sp_v+30.)/100.)), -10.],
                                [2./(1.+sp_exp((self.sp_v-30.)/100.)), 2./(1.+sp_exp((-self.sp_v+30.)/100.)), -30.]])
        # state variable relaxation time scale
        self.tauinf = np.array([[1., 2., 1.],
                                [2., 2., 3.]])
        # base class instructor
        super(TestChannel, self).__init__()


class h(IonChannel):
    def __init__(self, ratio=0.2):
        '''
        Hcn channel from (Bal and Oertel, 2000)
        '''
        self.ratio = ratio
        self.tauf = 40. # ms
        self.taus = 300. # ms
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # basic factors to construct channel open probability
        self.varnames = np.array([['hf'],
                                  ['hs']])
        self.powers = np.array([[1],
                                [1]], dtype=int)
        self.factors = np.array([1.-self.ratio, self.ratio])
        # asomptotic state variable functions
        self.varinf = np.array([[1./(1.+sp_exp((self.sp_v+82.)/7.))],
                                [1./(1.+sp_exp((self.sp_v+82.)/7.))]])
        # state variable relaxation time scales
        self.tauinf = np.array([[sp.Float(self.tauf)],
                                [sp.Float(self.taus)]])
        # base class constructor
        super(h, self).__init__()


class h_HAY(IonChannel):
    def __init__(self):
        '''
        Hcn channel from (Kole, Hallermann and Stuart, 2006)

        Used in (Hay, 2011)
        '''
        self.ion = ''
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m']])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 0.001 * 6.43 * (self.sp_v + 154.9) / (sp_exp((self.sp_v + 154.9) / 11.9) - 1.)
        spbetam  = 0.001 * 193. * sp_exp(self.sp_v / 33.1)
        self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam)]])
        # base class constructor
        super(h_HAY, self).__init__()


class Na_Ta(IonChannel):
    def __init__(self):
        '''
        (Colbert and Pan, 2002)

        Used in (Hay, 2011)
        '''
        self.ion = 'na'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham =   0.182 * (self.sp_v + 38.) / (1. - sp_exp(-(self.sp_v + 38.) / 6.)) # 1/ms
        spbetam  = - 0.124 * (self.sp_v + 38.) / (1. - sp_exp( (self.sp_v + 38.) / 6.)) # 1/ms
        spalphah = - 0.015 * (self.sp_v + 66.) / (1. - sp_exp( (self.sp_v + 66.) / 6.)) # 1/ms
        spbetah  =   0.015 * (self.sp_v + 66.) / (1. - sp_exp(-(self.sp_v + 66.) / 6.)) # 1/ms
        self.varinf = np.array([[spalpham / (spalpham + spbetam),
                                 spalphah / (spalphah + spbetah) ]])
        self.tauinf = np.array([[(1./2.95) / (spalpham + spbetam),
                                 (1./2.95) / (spalphah + spbetah)]]) # 1/ms
        # base class constructor
        super(Na_Ta, self).__init__()


class Na_p(IonChannel):
    def __init__(self):
        '''
        Derived by (Hay, 2011) from (Magistretti and Alonso, 1999)

        Used in (Hay, 2011)

        !!! Does not work !!!
        '''
        self.ion = 'na'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[3,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham =   0.182   * (self.sp_v + 38. ) / (1. - sp_exp(-(self.sp_v + 38. ) / 6.  ))  #1/ms
        spbetam  = - 0.124   * (self.sp_v + 38. ) / (1. - sp_exp( (self.sp_v + 38. ) / 6.  ))  #1/ms
        spalphah = - 2.88e-6 * (self.sp_v + 17. ) / (1. - sp_exp( (self.sp_v + 17. ) / 4.63))   #1/ms
        spbetah  =   6.94e-6 * (self.sp_v + 64.4) / (1. - sp_exp(-(self.sp_v + 64.4) / 2.63))   #1/ms
        self.varinf = np.array([[   1. / (1. + sp_exp(-(self.sp_v + 52.6) / 4.6)) ,
                                    1. / (1. + sp_exp( (self.sp_v + 48.8) / 10.)) ]])
        self.tauinf = np.array([[(6./2.95) / (spalpham + spbetam), (1./2.95) / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(Na_p, self).__init__()


class Kv3_1(IonChannel):
    def __init__(self):
        '''
        Shaw-related potassium channel (Rettig et al., 1992)

        Used in (Hay et al., 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m']])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        self.varinf = np.array([[1. / (1. + sp_exp(-(self.sp_v - 18.7) / 9.7))]])
        self.tauinf = np.array([[4. / (1. + sp_exp(-(self.sp_v + 46.56) / 44.14))]]) # ms
        # base class constructor
        super(Kv3_1, self).__init__()


class Kpst(IonChannel):
    def __init__(self):
        '''
        Persistent Potassium channel (Korngreen and Sakmann, 2000)

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[2, 1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        self.varinf = np.array([[1. / (1. + sp_exp(-(self.sp_v + 11.) / 12.)) ,
                                 1. / (1. + sp_exp( (self.sp_v + 64.) / 11.)) ]])
        self.tauinf = np.array([[(3.04 + 17.3 * sp_exp(-((self.sp_v + 60.) / 15.9)**2) + 25.2 * sp_exp(-((self.sp_v + 60.) / 57.4)**2)) / 2.95, \
                                 (360. + (1010. + 24. * (self.sp_v + 65.)) * sp_exp(-((self.sp_v + 85.) / 48.)**2)) / 2.95]]) # ms
        # base class constructor
        super(Kpst, self).__init__()


class Ktst(IonChannel):
    def __init__(self):
        '''
        Transient Potassium channel (Korngreen and Sakmann, 2000)

        Used in (Hay, 2011)
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[2, 1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        self.varinf = np.array([[1. / (1. + sp_exp(-(self.sp_v + 10.) / 19.)) ,
                                 1. / (1. + sp_exp( (self.sp_v + 76.) / 10.)) ]])
        self.tauinf = np.array([[(0.34 + 0.92 * sp_exp(-((self.sp_v + 81.) / 59.)**2)) / 2.95 ,
                                 (8.   + 49.  * sp_exp(-((self.sp_v + 83.) / 23.)**2)) / 2.95]]) # ms
        # base class constructor
        super(Ktst, self).__init__()


class m(IonChannel):
    def __init__(self):
        '''
        M-type potassium current (Adams, 1982)

        Used in (Hay, 2011)

        !!! does not work when e > V0 !!!
        '''
        self.ion = 'k'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m']])
        self.powers = np.array([[1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = 3.3e-3 * sp_exp( 2.5 * 0.04 * (self.sp_v + 35.))
        spbetam = 3.3e-3 * sp_exp(-2.5 * 0.04 * (self.sp_v + 35.))
        self.varinf = np.array([[spalpham / (spalpham + spbetam)]])
        self.tauinf = np.array([[(1. / (spalpham + spbetam)) / 2.95]])
        # base class constructor
        super(m, self).__init__()


class Ca_LVA(IonChannel):
    def __init__(self):
        '''
        LVA calcium channel (Avery and Johnston, 1996; tau from Randall, 1997)

        Used in (Hay, 2011)
        '''
        self.ion = 'ca'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[2,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        self.varinf = np.array([[1. / (1. + sp_exp(-(self.sp_v + 40.)/6.)), \
                                 1. / (1. + sp_exp((self.sp_v + 90.)/6.4))]])
        self.tauinf = np.array([[5. + 20./(1. + sp_exp((self.sp_v  + 35.)/5.))/2.95,
                                 20. + 50./(1. + sp_exp((self.sp_v + 50.)/7.))/2.95]]) # ms
        # base class constructor
        super(Ca_LVA, self).__init__()


class Ca_HVA(IonChannel):
    def __init__(self):
        '''
        High voltage-activated calcium channel (Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993)

        Used in (Hay, 2011)
        '''
        self.ion = 'ca'
        self.concentrations = []
        # always include this line, to define a sympy voltage symbol
        self.sp_v = sp.symbols('v')
        # state variables
        self.varnames = np.array([['m', 'h']])
        self.powers = np.array([[2,1]], dtype=int)
        self.factors = np.array([1.])
        # activation functions
        spalpham = -0.055 * (27. + self.sp_v) / (sp_exp(-(27. + self.sp_v)/3.8) - 1.)  #1/ms
        spbetam = 0.94 * sp_exp(-(75. + self.sp_v)/17.)  #1/ms
        spalphah = 0.000457 * sp_exp(-(13. + self.sp_v)/50.)   #1/ms
        spbetah = 0.0065 / (sp_exp(-(self.sp_v + 15.)/28.) + 1.)   #1/ms
        self.varinf = np.array([[spalpham / (spalpham + spbetam), spalphah / (spalphah + spbetah)]])
        self.tauinf = np.array([[1. / (spalpham + spbetam), 1. / (spalphah + spbetah)]]) # ms
        # base class constructor
        super(Ca_HVA, self).__init__()

