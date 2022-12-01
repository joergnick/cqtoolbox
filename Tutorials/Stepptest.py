

class LinearScatModelInt2(NewtonIntegrator):
    def precomputing(self,s):
        return s**(-2)
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def righthandside(self,t,history = None):
        return 1.0/5*t**5
    def nonlinearity(self,x,t,time_index):
        return 0*x
    def ex_sol(self,ts):
        return 4*ts**3

modelL       = LinearScatModelInt2()
m = 7
N = 500
T = 4
sol,counters = modelL.integrate(T,N,method = "RadauIIA-"+str(m))
exSol        = modelL.ex_sol(np.linspace(0,T,N+1))
err          = max(np.abs(sol[0,::m]-exSol))