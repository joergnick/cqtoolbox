import sys
sys.path.append('cq-core')
sys.path.append('../cq-core')
import numpy as np
import math
from linearcq import Conv_Operator
from rkmethods import RKMethod
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
    rk = RKMethod("RadauIIA-"+str(m),taus[j])
    time_points = rk.get_time_points(T,initial_point=False)
    rhs = 10*9*8*time_points**7
    ex_stages = time_points**10
    num_solStages = ScatOperator.apply_RKconvol(rhs,T,cutoff=10**(-15),show_progress=False,method="RadauIIA-"+str(m),first_value_is_t0=False)
    err_Stages = num_solStages-ex_stages
    num_solStages2 = Deriv.apply_RKconvol(err_Stages,T,cutoff=10**(-15),show_progress=False,method="RadauIIA-"+str(m),first_value_is_t0=False)
    solRK=np.zeros(N+1)
    solRK2=np.zeros(N+1)
    solRK[1:N+1]=np.real(num_solStages[m-1:N*m:m])
    solRK2[1:N+1]=np.real(num_solStages2[m-1:N*m:m])
    #errRK[j] = max(np.abs(err_Stages[0,:]))
    errRK[j] = max(np.abs(solRK-ex_sol))
    errRK2[j] = max(np.abs(solRK2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.loglog(taus,110*taus**2,linestyle='dashed')
plt.loglog(taus,errRK,marker='o')
plt.loglog(taus,errRK2,marker='s')
#plt.loglog(taus,10*taus**3,linestyle='dashed')
#plt.loglog(taus,taus**(3),linestyle='dashed')
#plt.loglog(taus,taus**(5),linestyle='dashed')
plt.loglog(taus,taus**(m+3),linestyle='dashed')
plt.loglog(taus,taus**(m),linestyle='dashed')
plt.savefig("test")
