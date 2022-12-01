import sys
sys.path.append("..")
from cqtoolbox import CQModel
import numpy as np
class ScatModel(CQModel):
    def precomputing(self,s):
        return np.array([[s**(1),0],[s**(-1),s**(-1)]])
        #return np.array([[s**(1),0],[0,s**(1)]])
    def harmonicForward(self,s,b,precomp = None):
        #return precomp.dot(b)
        return precomp.dot(b)
    def righthandside(self,t,history = None):
        #return 1.0/4*t**4+10000*t**9
        return [5*t**4+2**(-0.25)*t**(2.5), 2.0/6*t**6+2**(-0.25)*t**(2.5)]
        #return 5*t**4+t**(2.5)
        #return 1.0/6*t**6+t**2.5
    def nonlinearity(self,x,t,rhsInhom):
    #   print("X IN NONLINEARITY: ",x," RESULT : ",abs(x)**(-0.5)*x)
        val = np.linalg.norm(x)**(-0.5)*x
        nanindizes = np.isnan(val)  
        val[nanindizes] = 0
        return val
    def calcJacobian(self,x,t,rhsInhom):
        return -0.5*np.linalg.norm(x)**(-2.5)*np.outer(x,x)+np.linalg.norm(x)**(-0.5)*np.eye(2)
    def applyJacobian(self,jacob,b):
        return jacob.dot(b)
       # raise NotImplementedError("Hi")
    def nonlinearityInverse(self,x):
        #val = np.linalg.norm(x)**(1)*x
        val = np.linalg.norm(x)**(1)*x
        return val
        #return 0*x
        #return np.array([x[0]**1+x[1]**2,x[0]**3+x[1]**(1)])

model = ScatModel()
#print(model.nonlinearity(model.nonlinearityInverse(np.array([-12312,123123]))))
T  = 1
Am = 1
m  = 2
taus = np.zeros(Am)
err1 = np.zeros(Am)
err2 = np.zeros(Am)
Ns  = np.zeros(Am)
comp = np.zeros(Am)
#print(np.kron(np.array([[1,2],[3,4]]),np.identity(3)))
for j in range(Am):
    print("NEW SIMULATION")
    N = int(np.round(4*2**(j)))
    #N = 2*(2*2**j-1)
    Ns[j] = N
    tau = T*1.0/N
    taus[j] =tau 
    tt = np.linspace(0,T,N+1)
    #ex_sol = 1.0/30*tt**6
    ex_sol = tt**5
    #ex_sol = 4*3*tt**2
    #method = "BDF-"+str(m)
    method = "RadauIIA-"+str(m)
    sol,counters = model.simulate(T,N,method = method)
    #err[j] = np.abs(sol[0,2]-tau**3)
    #print(err)
    err1[j] = max(np.abs(sol[0,::m]-ex_sol))
    err2[j] = max(np.abs(sol[1,::m]-ex_sol))
    #err[j] = np.abs(sol[0,-1])
#    import matplotlib.pyplot as plt
##   print(sol)
#    plt.semilogy(sol[0,::m])
#    #plt.plot(sol[1,::m])
#    plt.semilogy(ex_sol,linestyle='dashed')
#    plt.show()
#print(err1)
#print(err2)
##print(sol)
##import matplotlib.pyplot as plt
##plt.plot(counters)
##plt.show()
####plt.plot(sol[0,::m])
####plt.plot(sol[1,::m])
####plt.plot(ex_sol,linestyle='dashed')
####plt.show()
#####
###
##plt.loglog(taus,err1)
##plt.loglog(taus,err2)
##
##plt.loglog(taus,taus**1,linestyle = 'dashed')
##plt.loglog(taus,taus**2,linestyle = 'dashed')
##plt.loglog(taus,taus**3,linestyle = 'dashed')
##plt.show()
#####    
