import sys
sys.path.append('../cq-core')
sys.path.append('cq-core')
import numpy as np
import math
from linearcq import Conv_Operator
from rkmethods import RKMethod
#from conv_op import ConvOperatoralt

## Creating right-hand side
T=20
## Frequency - domain operator defined

sigma = 1.0/T
def freq_int(s,b):
    #return s**1*np.exp(-1*s)*b
    #return s**(-0.5)*b
    s = s+sigma
    #return s**(-1)*b
    return s**(-1)*b
ScatOperator=Conv_Operator(freq_int)
Am = 15
m = 5
taus = np.zeros(Am)
errRK = np.zeros(Am)
errRK2 = np.zeros(Am)
for j in range(Am):
    N=int(np.round(4*1.5**j))
    taus[j] = T*1.0/N
    tt=np.linspace(0,T,N+1)
    ex_sol = np.array([0.5*math.sqrt(np.pi)*(math.erf(10)-math.erf(10-t)+math.erf(12)-math.erf(12-t)) for t in tt])
    #### RK  solution
    rk = RKMethod("RadauIIA-"+str(m),taus[j])
    time_points = rk.get_time_points(T,initial_point=False)
    rhs=np.exp(-sigma*time_points)*(np.exp(-(time_points-10)**2)+np.exp(-(time_points-12)**2))
    num_solStages = ScatOperator.apply_RKconvol(rhs,T,cutoff=10**(-15),show_progress=False,method="RadauIIA-"+str(m),first_value_is_t0=False)
    solRK=np.zeros(N+1)
    solRK[1:N+1]=np.real(num_solStages[m-1:N*m:m])
    solRK = np.exp(sigma*tt)*solRK
    errRK[j] = np.sqrt(taus[j]*sum((np.abs(solRK-ex_sol))**2))

print(errRK)
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
####plt.loglog(taus,110*taus**2,linestyle='dashed')
plt.loglog(taus,errRK,marker='o')
#plt.loglog(taus,10*taus**3,linestyle='dashed')
plt.loglog(taus[:8],0.000000001*taus[:8]**(min(m+1+1,2*m-1)),linestyle='dashed')
plt.loglog(taus[:6],0.00000000001*taus[:6]**(2*m-1),linestyle='dashed')
#plt.show()
plt.savefig("test")


#plt.plot(tt,solRK)
#plt.savefig("test")
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
