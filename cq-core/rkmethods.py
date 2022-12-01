import numpy as np
import math
class RKMethod():
    "Collects data and methods corresponding to a Runge-Kutta multistage method."
    method_name = ""
    c,A,b,m = 0,0,0,0
    delta_eigs,Tdiag,Tinv,delta_zero,tau  = 0,0,0,0,0
    def __init__(self,method,tau):
        if (method =="RadauIIA-1") or (method =="BDF-1") or (method == "Implicit Euler"):
            self.A=np.array([[1]])
            self.c=np.array([1])
            self.b=np.array([[1]])
        elif (method == "RadauIIA-2"):        
            self.A=np.array([[5.0/12,-1.0/12],
                           [3.0/4,1.0/4]])
            self.c=np.array([1.0/3,1])    
            self.b=np.array([[3.0/4,1.0/4]])
        elif (method == "RadauIIA-3"):
            self.A=np.array([[11.0/45-7*math.sqrt(6)/360, 37.0/225-169.0*math.sqrt(6)/1800 , -2.0/225+math.sqrt(6)/75],
                           [37.0/225+169.0*math.sqrt(6)/1800,11.0/45+7*math.sqrt(6)/360,-2.0/225-math.sqrt(6)/75],
                           [4.0/9-math.sqrt(6)/36,4.0/9+math.sqrt(6)/36,1.0/9]])
            self.c=np.array([2.0/5-math.sqrt(6)/10,2.0/5+math.sqrt(6)/10,1])
            self.b=np.array([[4.0/9-math.sqrt(6)/36,4.0/9+math.sqrt(6)/36,1.0/9]])
        elif (method == "RadauIIA-5"):
            m = 5
            self.c = np.array([0.5710419611451768219312e-01,0.2768430136381238276800e+00,0.5835904323689168200567e+00,0.8602401356562194478479e+00,1.0])
            self.A = self.construct_A(self.c)
            self.b = np.array([self.A[m-1,:]])
        elif (method == "RadauIIA-7"):
            m = 7
            self.c = np.array([0.2931642715978489197205e-01,0.1480785996684842918500,0.3369846902811542990971,0.5586715187715501320814,0.7692338620300545009169,0.9269456713197411148519,1.0])
            self.A = self.construct_A(self.c)
            self.b = np.array([self.A[m-1,:]])
        else:
            raise ValueError("Given method "+method+" not implemented.")
        self.method_name = method
        self.m    = len(self.c)
        self.tau  = tau
        self.delta_zero = np.linalg.inv(self.A)/tau
        self.delta_eigs,self.Tdiag  = np.linalg.eig(self.delta_zero)
        self.Tinv = np.linalg.inv(self.Tdiag)

    def construct_A(self,c):
        m  = len(c)
        CPm = np.ones((m,m))
        CQm = np.zeros((m,m))
        for j in range(m-1):
            CPm[:,j+1] = c**(j+1)
        for j in range(m):
            CQm[:,j] = c**(j+1)/(1.0*(j+1)) 
        return CQm.dot(np.linalg.inv(CPm))

    def diagonalize(self,x):
       # if len(x.shape)==1:
       #     if len(x) % self.m !=0:
       #         raise ValueError("Vector dimensions of Input does not allow diagonalization.")
       #     x.reshape((len(x)/self.m,self.m))
        return np.matmul(x,self.Tinv.T)

    def reverse_diagonalize(self,b_dof_x_m):
        return np.matmul(b_dof_x_m,self.Tdiag.T)
    def get_time_points(self,T):
        N  = int(np.round(T/self.tau))
        ts = np.zeros(self.m*N+1)
        for j in range(N):
            for k in range(self.m):
                ts[j*self.m+k+1] = (j+self.c[k])
        return self.tau*ts
class Extrapolator():
    "Provides functionality with regards to extrapolation."
    prolonged = 0
    coeff     = []
    p         = 0
    def extrapol_coefficients(self,p):
        coeffs = np.ones(p+1)
        for j in range(p+1):
                for m in range(p+1):
                        if m != j:
                                coeffs[j]=coeffs[j]*(p+1-m)*1.0/(j-m)
        self.coeff = coeffs
        self.p = p
        return coeffs

    def extrapol(self,u):
        p = self.p
        if len(u[0,:])<=p+1:
            u = np.concatenate((np.zeros((len(u[:,0]),p+1-len(u[0,:]))),u),axis=1)
        extrU = np.zeros(len(u[:,0]))
        gammas = self.coeff
        for j in range(p+1):
            extrU = extrU+gammas[j]*u[:,-p-1+j]
        return extrU
    def clamp_evals(self,x,speed=8):
        from scipy.special import comb
        x= x-x[0]
        if np.abs(x[-1])<10**(-14):
            x=10**(20)*x
            print("ODD STUFF AHEAD.")
        else:
            x = x/x[-1]
        x = 1-x
        result = np.zeros(len(x))
        for n in range(0,speed+1):
            result += comb(speed+n,n)*comb(2*speed+1,speed-n)*(-x)**n
        result *=x**(speed+1)
        return result

    def prolonge_towards_0(self,u,n_additional,rk,decay_speed=4):
        "Input dimensions are assumed to be a multiple of m."
        if n_additional == 0:
            return u
        dof = len(u[:,0])
        u_with_zeros = np.concatenate((np.zeros((dof,5)),u),axis = 1)
        additional_entries = np.zeros((dof,n_additional*rk.m))
        additional_times = np.array([])
        for j in range(n_additional):
            additional_times = np.append(additional_times,rk.tau*(j+2)+rk.tau*rk.c)

        interpolation_times = np.array([])
        interpolation_times = np.append(interpolation_times,rk.tau*rk.c)
        interpolation_times = np.append(interpolation_times,rk.tau+rk.tau*rk.c)
        interpolation_times = np.append(interpolation_times,2*rk.tau+rk.tau*rk.c)
        interpolation_times = np.append(interpolation_times,3*rk.tau+rk.tau*rk.c)
        interpolation_times = np.append(interpolation_times,4*rk.tau+rk.tau*rk.c)
        interpolation_times = np.append(interpolation_times,5*rk.tau+rk.tau*rk.c)
       # interpolation_times = interpolation_times[:min(len(interpolation_times),len(u[0,:]))]
        clamp_vals = self.clamp_evals(additional_times,speed=decay_speed)
        from scipy import interpolate

        index_min = min(len(interpolation_times),len(u_with_zeros[0,:]))
        for j in range(dof):
            #ipp = interpolate.interp1d(interpolation_times,u[j,-len(interpolation_times):],fill_value='extrapolate')
            #additional_entries[j,:] = ipp(additional_times)*clamp_vals
            ipp = interpolate.UnivariateSpline(interpolation_times[-index_min:],u_with_zeros[j,-index_min:],k=5)
            additional_entries[j,:] = ipp(additional_times)*clamp_vals
        self.prolonged = len(additional_entries[0,:])
        return np.concatenate((u,additional_entries),axis = 1)
    def cut_back(self,u):
        ret= u[:,:len(u[0,:])-self.prolonged]
        self.prolonged = 0
        return ret
#
#extr = Extrapolator()
#N = 10
#T=0.5#
#tau = T*1.0/N
#rk =RKMethod("RadauIIA-3",tau)
#import numpy as np
#print(1-rk.b.dot(np.linalg.inv(rk.A).dot(np.ones((3)))))
#u = np.zeros((2,N*rk.m+1))
#timepoints = rk.get_time_points(T)
#u[0,:] = timepoints**2
#u[1,:] = timepoints**3
#prolonge_by = 10
#print(timepoints)
#print(u[0,:])
#timepoints = rk.get_time_points(T+tau*prolonge_by)
#print(timepoints)
#print(u[0,:])
#u = extr.prolonge_towards_0(u,prolonge_by,rk,decay_speed = 8)
#print(u[0,:])
##import matplotlib.pyplot as plt
#plt.plot(timepoints,u[0,:])
#plt.plot(timepoints,u[1,:])
#
#plt.plot(timepoints,timepoints**2,linestyle='dashed')
#plt.plot(timepoints,timepoints**3,linestyle='dashed')
##plt.plot(vals)
#plt.show()
