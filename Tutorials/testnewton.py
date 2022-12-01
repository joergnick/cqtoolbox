from cqRK import CQModel
class ScatModel(CQModel):
	def precomputing(self,s):
		return np.array([[s,1],[1,s]])
	def harmonicForward(self,s,b,precomp = None):
		return precomp.dot(b)
	def righthandside(self,t,history = None):
		return np.array([3*t**2+t**4+t**3+t**8,5*t**3+t**9+t**4])
	def nonlinearity(self,x):
		#return 0*x
		return np.array([x[0]**1+x[1]**2,x[0]**3+x[1]**(1)])
	
model = ScatModel()
T = 1
Am = 1
m=3
import numpy as np
taus = np.zeros(Am)
err = np.zeros(Am)
Ns  = np.zeros(Am)
comp = np.zeros(Am)
#print(np.kron(np.array([[1,2],[3,4]]),np.identity(3)))
for j in range(Am):
	N = 2**14*2**j
	#N = 2*(2*2**j-1)
	Ns[j] = N
	tau = T*1.0/N
	taus[j] =tau 
	tt = np.linspace(0,T,N+1)
	#ex_sol = 1.0/30*tt**6
	ex_sol = tt**3
	#ex_sol = 4*3*tt**2
	#method = "BDF-"+str(m)
	method = "RadauIIA-"+str(m)
	sol = model.simulate(T,N,method = method)

	#err[j] = np.abs(sol[0,2]-tau**3)
	#print(err)
	err[j] = max(np.abs(sol[0,::m]-ex_sol))
	#err[j] = np.abs(sol[0,-1])
#
#	import matplotlib.pyplot as plt
##	print(sol)
#	plt.plot(sol[0,::m])
#	#plt.plot(sol[1,::m])
#	plt.plot(ex_sol,linestyle='dashed')
#	plt.show()
print(err)
#print("NUMERICAL SOLUTION;")
#print(sol)
##print(sol[::2])
#print("EXACT SOLUTION:")
#print(ex_sol)
#print("ENDING ",sol)
#print(taus**3)
#import matplotlib.pyplot as plt
#plt.plot(sol[0,::2])
##plt.plot(ex_sol,linestyle='dashed')
#plt.show()
#
#import matplotlib.pyplot as plt
#plt.loglog(taus,err)
#plt.loglog(taus,taus**2,linestyle = 'dashed')
#plt.show()
	#
