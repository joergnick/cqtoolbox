import unittest
import numpy as np
import sys
sys.path.append('../cqToolbox')
from cqtoolbox import CQModel

## Problems with analytic solutions 

class NonlinearScatModel(CQModel):
    def precomputing(self,s):
        return s**1
    def harmonicForward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,history = None):
        return 3*t**2+t**9
    def nonlinearity(self,x,t,phi):
        return x**3
    def ex_sol(self,ts):
        return ts**3

## Test cases, two for each of the predefined models
## above.
class TestCQMethods(unittest.TestCase):
    def test_nonlinear_RadauIIA_1(self):
        modelN       = NonlinearScatModel()
        m = 1
        N = 5
        T = 1
        sol,counters = modelN.simulate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),0.5)


    def test_nonlinear_RadauIIA_2(self):
        modelN       = NonlinearScatModel()
        m = 2
        N = 11
        T = 2
        sol,counters = modelN.simulate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-3))

    def test_nonlinear_RadauIIA_3(self):
        modelN       = NonlinearScatModel()
        m = 3
        N = 7
        T = 2
        sol,counters = modelN.simulate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-7))


if __name__ == '__main__':
    unittest.main()

