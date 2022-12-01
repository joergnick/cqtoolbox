import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')

import numpy as np
import math
from linearcq import Conv_Operator
from rkmethods import RKMethod
#from conv_op import ConvOperatoralt

def create_timepoints(method,N,T):
    rk = RKMethod(method,T*1.0/N)
    m = rk.m
    return rk.get_time_points(T)


## Creating right-hand side
T=20
## Frequency - domain operator defined

sigma = 0.1
#sigma = 0.05
param = 2
def freq_int(s,b):
    #return s**1*np.exp(-1*s)*b
    #return s**(-0.5)*b
    s = s+sigma
    #return s**(-1)*b
    #return (s**(1)+1+s**(-1))**(-1)*b
    return (s**(param))*b
def rhs_func(t):
#    return t**30
   # centres = [ 5,6,7,8,9]
   # factors = [ 10,1,1,1,0.1]
    centres = [15]
    factors = [1]
    rhs = 0*t
    for j in range(len(centres)):
        rhs += np.exp(-factors[j]*(t-centres[j])**2)
    return rhs

ScatOperator=Conv_Operator(freq_int)
Am = 8
m = 2

###### Calculate reference solution
N=int(np.round(8*2**(Am+2)))
print("REFERENCE N = ",N)
tt=np.linspace(0,T,N+1)
#### RK  solution
time_points=create_timepoints("RadauIIA-"+str(m),N,T)
rhs= np.exp(-sigma*time_points)*rhs_func(time_points)
#rhs=np.exp(-sigma*time_points)*(np.exp(-(time_points-13)**2)+np.exp(-(time_points-11)**2)+np.exp(-(time_points-10)**2)+np.exp(-(time_points-12)**2))  
#rhs=np.exp(-sigma*time_points)*time_points**20
num_solStages = ScatOperator.apply_RKconvol(rhs[1:],T,cutoff=10**(-15),show_progress=False,method="RadauIIA-"+str(m))
solref=np.zeros(N)
print(num_solStages.shape)
solref[:N]=np.real(num_solStages[m-1:N*m:m])
solref = np.exp(sigma*tt[1:])*solref
Nref = N 

taus = np.zeros(Am)
errRK = np.zeros(Am)
for j in range(Am):
    N=int(np.round(8*2**j))
    speed = Nref/N
    taus[j] = T*1.0/N
    tt=np.linspace(0,T,N+1)
    #### RK  solution
    time_points=create_timepoints("RadauIIA-"+str(m),N,T)
    rhs= np.exp(-sigma*time_points)*rhs_func(time_points)
    #rhs=np.exp(-sigma*time_points)*(np.exp(-(time_points-13)**2)+np.exp(-(time_points-11)**2)+np.exp(-(time_points-10)**2)+np.exp(-(time_points-12)**2))
    #rhs=np.exp(-sigma*time_points)*time_points**20
    num_solStages = ScatOperator.apply_RKconvol(rhs[1:],T,cutoff=10**(-15),show_progress=False,method="RadauIIA-"+str(m))
    solRK=np.zeros(N)
    solRK=np.real(num_solStages[0,m-1:N*m:m])
    solRK = np.exp(sigma*tt[1:])*solRK
    #errRK[j] = np.sqrt(taus[j]*sum((np.abs(solRK[0:N+1]-solref[0:Nref+1:speed]))**2))
print(errRK)
##print(solRK[-1])
#import matplotlib.pyplot as plt
#####plt.loglog(taus,110*taus**2,linestyle='dashed')
#plt.loglog(taus,errRK,marker='o')
##plt.loglog(taus,10*taus**3,linestyle='dashed')
#refline1 = errRK[0]*taus**(min(m+1-param,2*m-1))/(taus[0]**(min(m+1-param,2*m-1)))
#plt.loglog(taus,refline1,linestyle='dashed')
##refline2 = errRK[0]*taus**(2*m-1)/(taus[0]**(2*m-1))
##plt.loglog(taus,refline2,linestyle='dashed')
##plt.ylim([10**(-15),10**(-2)])
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
