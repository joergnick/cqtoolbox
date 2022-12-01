import sys
sys.path.append('../cq-core')
sys.path.append('cq-core')
import numpy as np
import math
from linearcq import Conv_Operator
from rkmethods import RKMethod
#from conv_op import ConvOperatoralt

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
T=20
## Frequency - domain operator defined

sigma = 0
def freq_int(s,b):
    #return s**1*np.exp(-1*s)*b
    #return s**(-0.5)*b
    s = s+sigma
    #return s**(-1)*b
    return s**(-1)*b
ScatOperator=Conv_Operator(freq_int)
Am = 15
m = 7
taus = np.zeros(Am)
errRK = np.zeros(Am)
for j in range(Am):
    N=int(np.round(4*1.5**j))
    taus[j] = T*1.0/N
    tt=np.linspace(0,T,N+1)
    #ex_sol = 1.0/21*tt**21
    ex_sol = np.array([0.5*math.sqrt(np.pi)*(math.erf(10)-math.erf(10-t)+math.erf(12)-math.erf(12-t)) for t in tt])
    #ex_sol = 0.776469257929*tt**(12.1)
    #ex_sol = 0.280029125779*tt**(12.5)
    #ex_sol = 0.806824245138*tt**(8.1)
    #ex_sol = 32.0/35.0*np.sqrt(np.pi)**(-1)*tt**(3.5)
    #### RK  solution
    time_points=create_timepoints("RadauIIA-"+str(m),N,T)
    rhs=np.exp(-sigma*time_points)*(np.exp(-(time_points-10)**2)+np.exp(-(time_points-12)**2))
    #rhs=np.exp(-sigma*time_points)*time_points**20
    num_solStages = ScatOperator.apply_RKconvol(rhs,T,cutoff=10**(-15),show_progress=False,method="RadauIIA-"+str(m))
    solRK=np.zeros(N+1)
    solRK[1:N+1]=np.real(num_solStages[0,m-1:N*m:m])
    solRK = np.exp(sigma*tt)*solRK
    errRK[j] = np.sqrt(taus[j]*sum((np.abs(solRK-ex_sol))**2))

print(errRK)
import matplotlib.pyplot as plt
####plt.loglog(taus,110*taus**2,linestyle='dashed')
#plt.loglog(taus,errRK,marker='o')
##plt.loglog(taus,10*taus**3,linestyle='dashed')
#plt.loglog(taus,0.000000001*taus**(min(m+1+1,2*m-1)),linestyle='dashed')
#plt.loglog(taus,0.00000000001*taus**(2*m-1),linestyle='dashed')
#plt.show()

#
#plt.plot(tt,solRK)
#plt.plot(tt,32.0/35*np.sqrt(np.pi)**(-1)*tt**3.5, linestyle='dashed')
#plt.show()
##plt.semilogy(tt,np.abs(sol_ref-32.0/(35*np.sqrt(np.pi))*tt**(2.5)))
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
