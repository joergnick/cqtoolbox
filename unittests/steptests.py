import unittest
import numpy as np
import sys

sys.path.append('cq-core')
sys.path.append('../cq-core')
from cqtoolbox import CQModel
from newtonStepper import NewtonIntegrator
from rkmethods     import RKMethod
## Problems with analytic solutions 
class LinearScatModel(NewtonIntegrator):
    def precomputing(self,s):
        return s**2
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,time_index,history = None):
        return 8*7*t**6
    def nonlinearity(self,x,t,time_index):
        return 0*x
    def ex_sol(self,ts):
        return ts**8
class LinearScatModelSimple(NewtonIntegrator):
    def precomputing(self,s):
        return s**1
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,time_index,history = None):
        return 3*t**2
    def nonlinearity(self,x,t,time_index):
        return 0*x
    def ex_sol(self,ts):
        return ts**3

class LinearScatModelInt(NewtonIntegrator):
    def precomputing(self,s):
        return s**(-1)
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,time_index,history = None):
        return t**3
    def nonlinearity(self,x,t,time_index):
        return 0*x
    def ex_sol(self,ts):
        return 3*ts**2
class LinearScatModelInt2(NewtonIntegrator):
    def precomputing(self,s):
        return s**(-2)
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,time_index,history = None):
        return 1.0/5*t**5
    def nonlinearity(self,x,t,time_index):
        return 0*x
    def ex_sol(self,ts):
        return 4*ts**3

class NonlinearScatModel(NewtonIntegrator):
    def precomputing(self,s):
        return s**1
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,time_index,history = None):
        return 3*t**2+t**9
    def nonlinearity(self,x,t,time_index):
        return x**3
    def ex_sol(self,ts):
        return ts**3

class ScissorModel(NewtonIntegrator):
    def precomputing(self,s):
        return np.array([s**1,s**(-1)])
    def harmonic_forward(self,s,b,precomp = None):
        return np.array([precomp[0]*b[0],precomp[1]*b[1]])
    def righthandside(self,t,time_index,history = None):
        return np.array([3*t**2 , 1.0/3*t**3])
    def nonlinearity(self,x,t,time_index):
        return np.array([ 0 , 0 ])
    def ex_sol(self,ts):
        return np.array([ts**3,ts**2])


class NonlinearScatModel2Components(NewtonIntegrator):
    def precomputing(self,s):
        return np.array([s**1,s**2])
    def harmonic_forward(self,s,b,precomp = None):
        return np.array([precomp[0]*b[0],precomp[1]*b[1]])
    def righthandside(self,t,time_index,history=None):
        return np.array([3*t**2+t**9 +t**4,4*3*t**2+t**4])
    def nonlinearity(self,x,t,time_index):
        return np.array([x[0]**3+x[1],x[1]])
    def ex_sol(self,ts):
        return np.array([ts**3,ts**4])

class NonlinearScatModelCustomGradient(NewtonIntegrator):
    def precomputing(self,s):
        return np.array([s**1,s**2])
    def harmonic_forward(self,s,b,precomp = None):
        return np.array([precomp[0]*b[0],precomp[1]*b[1]])
    def righthandside(self,t,time_index,history=None):
        return np.array([3*t**2+t**9 +t**4,4*3*t**2+t**4])
    def nonlinearity(self,x,t,time_index):
        return np.array([x[0]**3+x[1],x[1]])
    def calc_jacobian(self,x,t,time_index):
        return np.array([[3*x[0]**2,1],[0,1]])
    def apply_jacobian(self,jacobian,x):
        return jacobian.dot(x)
    def ex_sol(self,ts):
        return np.array([ts**3,ts**4])




## Test cases, two for each of the predefined models
## above.
class TestCQMethods(unittest.TestCase):
    def test_linear_RadauIIA_1Simple(self):
        modelL       = LinearScatModelSimple()
        m = 1
        N = 100
        T = 1
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = np.max(np.max(np.abs(sol[0,::m]-exSol)))
        self.assertLess(np.abs(err),10**(-1))
    def test_scissor(self):
        modelS = ScissorModel()
        m = 3
        N = 21
        T = 4
        sol,counters = modelS.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelS.ex_sol(np.linspace(0,T,N+1))
#        import matplotlib
#        matplotlib.use('Agg')
#        import matplotlib.pyplot as plt
#        plt.plot(sol[1,::m])
#        plt.plot(exSol[1,:],linestyle='dashed')
#        plt.savefig('temp.png')
        err          = np.max(np.max(np.abs(sol[:,::m]-exSol)))
        self.assertLess(np.abs(err),10**(-6))

    def test_linear_RadauIIA_2SimpleInt(self):
        modelL       = LinearScatModelInt()
        m = 2
        N = 20
        T = 4
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-1))

    def test_linear_RadauIIA_3SimpleInt(self):
        modelL       = LinearScatModelInt()
        m = 3
        N = 10
        T = 4
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-5))

    def test_linear_RadauIIA_3SquaredInt(self):
        modelL       = LinearScatModelInt2()
        m = 3
        N = 60
        T = 4
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-1))

    def test_linear_RadauIIA_3SquaredIntmhigh(self):
        modelL       = LinearScatModelInt2()
        m = 7
        N = 47
        T = 4
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-4))


    def test_linear_RadauIIA_2(self):
        modelL       = LinearScatModel()
        m = 2
        N = 8
        T = 1
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-2))

    def test_linear_RadauIIA_3(self):
        modelL       = LinearScatModel()
        m = 3
        N = 9
        T = 1
        sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-3))

    def test_nonlinear_RadauIIA_2(self):
        modelN       = NonlinearScatModel()
        m = 2
        N = 40
        T = 2
        sol,counters = modelN.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),2*10**(-3))


    def test_nonlinear_RadauIIA_3(self):
        modelN       = NonlinearScatModel()
        m = 3
        N = 30
        T = 2
        sol,counters = modelN.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = max(np.abs(sol[0,::m]-exSol))
        self.assertLess(np.abs(err),10**(-2))
    def test_nonlinear_RadauIIA_2_components(self):
        modelN       = NonlinearScatModel2Components()
        m = 2
        N = 16

        T = 2
        sol,counters = modelN.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = np.max(np.max(np.abs(sol[:,::m]-exSol)))
        self.assertLess(np.abs(err),5*10**(-2))
    def test_nonlinear_RadauIIA_2_custom_gradient(self):
        modelN       = NonlinearScatModelCustomGradient()
        m = 2
        N = 50
        T = 2
        sol,counters = modelN.integrate(T,N,method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = np.max(np.max(np.abs(sol[:,::m]-exSol)))
        self.assertLess(np.abs(err),10**(-3))
    def test_nonlinear_RadauIIA_3_custom_gradient(self):
        modelN       = NonlinearScatModelCustomGradient()
        m = 3
        N = 31
        T = 2
        sol,counters = modelN.integrate(T,N,tolsolver=10**(-8),method = "RadauIIA-"+str(m))
        exSol        = modelN.ex_sol(np.linspace(0,T,N+1))
        err          = np.max(np.max(np.abs(sol[:,::m]-exSol)))
        self.assertLess(np.abs(err),10**(-2))
    def test_nonlinear_inhom_RadauIIA_2(self):
        N       = 100
        T       = 2
        m       = 2
        tau     = T*1.0/N
        rk      = RKMethod("RadauIIA-3",tau)
        inHom   = rk.get_time_points(T)**1

        class NonlinearInhomModel(NewtonIntegrator):
            def precomputing(self,s):
                return np.array([s**1,s**2])
            def harmonic_forward(self,s,b,precomp = None):
                return np.array([precomp[0]*b[0],precomp[1]*b[1]])
            def righthandside(self,t,time_index=None,history = None):
                return np.array([3*t**2+(t**3+t**1)**3 +t**4,4*3*t**2+t**4])
            def nonlinearity(self,x,t,time_index):
                return np.array([(x[0]+inHom[time_index])**3+x[1],x[1]])
            def ex_sol(self,ts):
                return np.array([ts**3,ts**3])
        #modelNI = NonlinearInhomModel()
        modelNI = NonlinearScatModel2Components()
        sol,counters = modelNI.integrate(T,N,tolsolver=10**(-10),method = "RadauIIA-"+str(m))
        exSol        = modelNI.ex_sol(np.linspace(0,T,N+1))
        err          = np.max(np.max(np.abs(sol[:,::m]-exSol)))
        self.assertLess(np.abs(err),5*10**(-4))
    def test_nonlinear_inhom_RadauIIA_3(self):
        N       = 60
        T       = 1.5
        m       = 3
        tau     = T*1.0/N
        rk      = RKMethod("RadauIIA-3",tau)
        inHom   = rk.get_time_points(T)**0.5

        class NonlinearInhomModel(NewtonIntegrator):
            def precomputing(self,s):
                return np.array([s**1,s**2])
            def harmonic_forward(self,s,b,precomp = None):
                return np.array([precomp[0]*b[0],precomp[1]*b[1]])
            def righthandside(self,t,history = None):
                return np.array([3*t**2+(t**3+t**0.5)**3 +t**4,4*3*t**2+t**4])
            def nonlinearity(self,x,t,time_index):
                return np.array([(x[0]+inHom[time_index])**3+x[1],x[1]])
            def ex_sol(self,ts):
                return np.array([ts**3,ts**4])
        modelNI = NonlinearScatModel2Components()
        sol,counters = modelNI.integrate(T,N,tolsolver=10**(-10),method = "RadauIIA-"+str(m))
        exSol        = modelNI.ex_sol(np.linspace(0,T,N+1))
        err          = np.max(np.max(np.abs(sol[:,::m]-exSol)))
        self.assertLess(np.abs(err),10**(-4))

    def test_extrapolation_p1(self):
        modelN       = NonlinearScatModel()
        self.assertTrue((modelN.extrapol_coefficients(1)==[-1,2]).all())
    def test_extrapolation_p2(self):
        modelN       = NonlinearScatModel()
        self.assertTrue((modelN.extrapol_coefficients(2)==[1,-3,3]).all())
if __name__ == '__main__':
    unittest.main()

