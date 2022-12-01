import numpy as np
class DirectCQ:
    tol = 10**(-15)
    N       = -1
    weights = []
    def __init__(self,laplace_evals):
        self.laplace_evals = laplace_evals
    def delta(self,zeta):
        return 1-zeta
        #return 1.5-2.0*zeta+0.5*zeta**2
    def calc_weights(self,N,T):
        tau = T*1.0/N
        L = 2*(N+1)
        rho = self.tol**(1.0/(2*L))
        weights = 1j*np.zeros(N+1) 
        freqs  = rho*np.exp(-1j*2*np.pi*(np.linspace(0,L-1,L)/(L)))
        lap_evals = [self.laplace_evals(self.delta(f)/tau) for f in freqs]
        for n in range(N+1):
            for l in range(L):
                weights[n] += lap_evals[l]*np.exp(l*n*1j*2*np.pi/(L))
            weights[n] = rho**(-n)/L*weights[n]
        if (np.abs(np.imag(weights))>10**(-10)).any():
            print("Warning, nontrivial imaginary part, max |imag(weights)| = "+str(max(np.abs(np.imag(weights)))))
        return np.real(weights)

    def forward_convolution(self,weights,g):
        if( len(weights) != len(g)):
            raise ValueError('Lengths are different.')
        N = len(weights) - 1
        w_star_g = np.zeros(N+1) 
        for j in range(N+1):
            for i in range(j+1):
                w_star_g[j] += weights[i]*g[j-i]
        return w_star_g


    def time_stepping(self,T,g):
        N = len(g)-1
        weights = self.calc_weights(N,T)
        u = np.zeros(N+1)
        for n in range(1,N+1):
            b = g[n]-sum([weights[n-j]*u[j] for j in range(n)])
            u[n] = weights[0]**(-1)*b
        return u
def laplace_evals(s):
    return s**(-2)
cq = DirectCQ(laplace_evals)
T = 1
Am = 1
errs = np.zeros(Am)
Ns = np.zeros(Am)
for j in range(Am):
    N = 2*2**(j)
    Ns[j] = N
    g = 1.0/5*np.linspace(0,T,N+1)**5
    u = cq.time_stepping(T,g)
    w = cq.calc_weights(N,T)
    print("weights = ",w)
    errs[j] = np.linalg.norm(u-4*np.linspace(0,T,N+1)**3)
    print(Ns)
    print(errs)
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#plt.plot(u)
#plt.plot(4*np.linspace(0,T,N+1)**3,linestyle='dashed',color = 'red')
#
#plt.savefig('temp.png')