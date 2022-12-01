import sys
sys.path.append('..')
sys.path.append('cqToolbox')
import numpy as np
import math
from linearcq import Conv_Operator
from rkmethods import RKMethod

def create_timepoints(method,N,T):
    rk = RKMethod(method,1)
    m = rk.m
    #if (method=="RadauIIA-1"):
    #    m = 1
    #    c_RK=np.array([1])
    #if (method=="RadauIIA-2"):
    #    m = 2
    #    c_RK=np.array([1.0/3,1])
    #if (method=="RadauIIA-3"):
    #    m = 3
    #    c_RK=np.array([2.0/5-math.sqrt(6)/10,2.0/5+math.sqrt(6)/10,1])
    c_RK = rk.c
    time_points=np.zeros((1,m*N))
    for j in range(m):
        time_points[0,j:m*N:m]=c_RK[j]*1.0/N*np.ones((1,N))+np.linspace(0,1-1.0/N,N)
    return T*time_points

## Creating right-hand side
T=1
## Frequency - domain operator defined
def freq_int(s,b):
    #return s**1*np.exp(-1*s)*b
    #return s**(-0.5)*b
    return s**(-3)*b
def freq_der(s,b):
    return s*b
ScatOperator=Conv_Operator(freq_int)
Deriv       =Conv_Operator(freq_der)
Am = 15
m = 5
taus = np.zeros(Am)
errRK = np.zeros(Am)
errRK2 = np.zeros(Am)
errRK3 = np.zeros(Am)
for j in range(Am):
    N=int(np.round(4*1.5**j))
    taus[j] = T*1.0/N
    tt = np.linspace(0,T,N+1)
    ex_sol = tt**10
    #### RK  solution
    time_points=create_timepoints("RadauIIA-"+str(m),N,T)
    rhs = 10*9*8*time_points**7
    ex_stages = time_points**10
    num_solStages = ScatOperator.apply_RKconvol(rhs,T,cutoff=10**(-15),show_progress=False,method="RadauIIA-"+str(m))
    err_Stages = num_solStages-ex_stages
    num_solStages2 = Deriv.apply_RKconvol(err_Stages,T,cutoff=10**(-15),show_progress=False,method="RadauIIA-"+str(m))
    solRK=np.zeros(N+1)
    solRK2=np.zeros(N+1)
    solRK[1:N+1]=num_solStages[0,m-1:N*m:m]
    solRK2[1:N+1]=num_solStages2[0,m-1:N*m:m]
    #errRK[j] = max(np.abs(err_Stages[0,:]))
    errRK[j] = max(np.abs(solRK-ex_sol))
    errRK2[j] = max(np.abs(solRK2))
import matplotlib.pyplot as plt
#plt.loglog(taus,110*taus**2,linestyle='dashed')
plt.loglog(taus,errRK)
plt.loglog(taus,errRK2)
#plt.loglog(taus,10*taus**3,linestyle='dashed')
#plt.loglog(taus,taus**(3),linestyle='dashed')
#plt.loglog(taus,taus**(5),linestyle='dashed')
plt.loglog(taus,taus**(m+3),linestyle='dashed')
plt.loglog(taus,taus**(m),linestyle='dashed')
plt.show()


#plt.plot(tt,sol_ref)
#plt.plot(tt,32.0/35*np.sqrt(np.pi)**(-1)*tt**3.5, linestyle='dashed')
#plt.semilogy(tt,np.abs(sol_ref-32.0/(35*np.sqrt(np.pi))*tt**(2.5)))
#### Multistep:
#
##plt.plot(np.linspace(0,T,N+1),solBDF)
##
##plt.plot(tt,32.0/35*np.sqrt(np.pi)**(-1)*tt**3.5, linestyle='dashed')
##plt.show() 
#






#m=3
#Am_time=8
#tau_s=np.zeros(Am_time)
#errors=np.zeros(Am_time)
#for ixTime in range(Am_time):
#   N=8*2**(ixTime)
#   tau_s[ixTime]=T*1.0/N
#   ## Rescaling reference solution:        
#   tt=np.linspace(0,T,N+1)
#   speed=N_ref/N
#   resc_ref=np.zeros((3,N+1))
#   for j in range(N+1):
#       resc_ref[:,j]      = sol_ref[:,j*speed]
#   ## Numerical Solution :
#
#   rhs=create_rhs(N,T,m)
#   num_sol  = deriv_solution(N,T,m)
#   errors[ixTime]=np.max(np.abs(resc_ref-num_sol))
#
#import matplotlib.pyplot as plt
#plt.loglog(tau_s,errors)
#plt.loglog(tau_s,tau_s**3,linestyle='dashed')
#plt.loglog(tau_s,tau_s**2,linestyle='dashed')
#plt.loglog(tau_s,tau_s**1,linestyle='dashed')
#plt.show()
