from cqStepper import AbstractIntegrator
from cqStepperSparse import AbstractIntegratorSparse
from cqDirectStepper import AbstractIntegratorDirect
from linearcq import Conv_Operator
import numpy as np
from rkmethods import RKMethod
import math
import random
import warnings
warnings.filterwarnings("error")
T  = 1
class ImplicitEuler(AbstractIntegratorSparse):
    def time_step(self,W0,j,rk,history,w_star_sol_j,tolsolver = 0):
        tau = rk.tau
        t   = tau*j
        rhs = np.zeros((1,m))
        for stageInd in range(m):
            rhs[:,stageInd] = 1.0/9*((t+tau*rk.c[stageInd]))**9-w_star_sol_j[:,stageInd]
        rhs = rk.diagonalize(rhs)
        sol = 1j*np.zeros((1,m))
        for stageInd in range(m):
            sol[:,stageInd] = W0[stageInd]**(-1)*(rhs[:,stageInd])
        sol = np.real(rk.reverse_diagonalize(sol))
        return np.real(sol)
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def precomputing(self,s):
        return s**(-2)*(1+0*10**(-10)*random.uniform(-1,1))

class ImplicitEulerDirect(AbstractIntegratorDirect):
    def time_step(self,W0,j,rk,history,w_star_sol_j,tolsolver = 0):
        tau = rk.tau
        t   = tau*j
        rhs = np.zeros((1,m))
        for stageInd in range(m):
            rhs[:,stageInd] = 1.0/9*((t+tau*rk.c[stageInd]))**9-w_star_sol_j[:,stageInd]
        rhs = rk.diagonalize(rhs)
        sol = 1j*np.zeros((1,m))
        for stageInd in range(m):
            sol[:,stageInd] = W0[stageInd]**(-1)*(rhs[:,stageInd])
        sol = np.real(rk.reverse_diagonalize(sol))
        return np.real(sol)
    def harmonic_forward(self,s,b,precomp = None):
        return precomp*b
    def precomputing(self,s):
        return s**(-2)*(1+0*10**(-10)*random.uniform(-1,1))



def th_der(s,b):
    return s**2*b
forward_der = Conv_Operator(th_der)

int_der = ImplicitEuler()
int_der2 = ImplicitEulerDirect()
Am = 1
m  = 1
err = np.zeros(Am)
err_fw = np.zeros(Am)
err_fw2 = np.zeros(Am)
Ns  = np.zeros(Am)
for j in range(Am):
    #N   = 10000
    N   = 1000*2**j
    #N = 2
    Ns[j] = N
    rk = RKMethod("RadauIIA-"+str(m),T*1.0/N)
    rhs = 1.0/9*(rk.get_time_points(T))**9
    #rhs = np.concatenate((np.zeros(8),rhs))
    print(rhs)
    print("lenRHS = ",len(rhs))
    #rhs = 1.0/5*(rk.get_time_points(T))**5
    print("First integration.")
    sol,counters = int_der.integrate(T,N,method = rk.method_name,factor_laplace_evaluations = 2,max_evals_saved = 100000)
#    print("Second integration.")
#    sol2,counters = int_der2.integrate(T,N,method = rk.method_name,factor_laplace_evaluations = 2,max_evals_saved = 100000)
#    print("Forward integration.")
    sol_fw = forward_der.apply_RKconvol(rhs[1::],T,show_progress=False,method="RadauIIA-"+str(m))
#    
#    #ex_sol = 4*np.linspace(0,T,N+1)**3
#   # err[j] = np.max(np.abs(sol-sol2))
#    #err_fw[j] = max(np.abs(sol_fw[0,0::m]-sol[0,0::m]))
#    #err_fw[j] = max(np.abs(sol_fw[0,0::m]-sol[0,1::m]))
#    #err[j] = max(np.abs(sol[0,::m]-4*np.linspace(0,T,N+1)**3))
#    #print(Ns)
#    #print(err)
    print("||fw_sol|| = ",np.max(np.abs(sol_fw[:,::m])))
    print("||sol|| = ",np.max(np.abs(sol[:,::m])))
#   # print("err = ",err)
#   # print("err_fw = ",err_fw)
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#plt.plot(sol[0,::m])
#plt.plot(4*np.linspace(0,T,N+1)**3,linestyle='dashed',color='red')
#plt.semilogy(np.abs(sol[0,::]-4*rk.get_time_points(T)**3))
#plt.semilogy(np.abs(sol[0,::m]-4*np.linspace(0,T,N+1)**3))
#print(Ns)
#print(err)
#
#plt.loglog(Ns**(-1),Ns**(-(1)),linestyle='dashed')
#plt.loglog(Ns**(-1),err_fw,marker='o')
#plt.loglog(Ns**(-1),err,marker='d')
#
#plt.savefig('temp.png')
###plt.plot(sol[0])
###plt.plot(np.linspace(0,1,N+1)**3,linestyle='dashed')
##plt.show()
#