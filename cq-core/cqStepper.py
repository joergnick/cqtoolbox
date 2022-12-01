#import psutil
import time
import math
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import numpy as np
from linearcq import Conv_Operator
from rkmethods import RKMethod
class AbstractIntegrator:
    def __init__(self):
        self.tdForward = Conv_Operator(self.forward_wrapper)
        self.freqObj = dict()
        self.freqUse = dict()
        self.countEv = 0
        
        ## Methods supplied by user:
    def time_step(self,s0,W0,t,history,conv_history,x0):
        raise NotImplementedError("No time stepping given.") 
    def harmonic_forward(self,s,b,precomp = None):
        raise NotImplementedError("No time-harmonic forward operator given.")
        ## Optional method supplied by user:
    def precomputing(self,s):
        raise NotImplementedError("No precomputing given.")
    def righthandside(self,t,index,history=None):
        raise NotImplementedError("Method righthandside not given.")
        ## Methods provided by class
    def forward_wrapper(self,s,b):
        ## Frequency was already seen and evaluation has been saved:           
        if s in self.freqObj:
            self.freqUse[s] = self.freqUse[s]+1
            return self.harmonic_forward(s,b,precomp=self.freqObj[s])
        ## Frequency has not been seen and we have saved less than the maximum
        self.countEv += 1
        if (self.count_saved_evals < self.max_evals_saved):
            self.freqObj[s] = self.precomputing(s)
            self.freqUse[s] = 1
            self.count_saved_evals +=1
            return self.harmonic_forward(s,b,precomp=self.freqObj[s])
        ## Frequency has not been seen and we have already saved the maximum amount of evaluations,
        ## Thus we simply apply the forward application without saving the objects.
        return self.harmonic_forward(s,b,precomp=self.precomputing(s))

    def integrate(self,T,N,method = "RadauIIA-2",tolsolver = 10**(-10),max_evals_saved=100000,factor_laplace_evaluations = 2,debug_mode=False,same_rho = False,same_L = False):
        self.tdForward.external_N = N 
        self.max_evals_saved   = max_evals_saved
        self.count_saved_evals = 0
        tau = T*1.0/N
        rk = RKMethod(method,tau)
        m = rk.m
        ## Initializing right-hand side:
        try:
            dof = len(self.righthandside(0,0))
        except:
            dof = 1
        ## Actual solving:
        W0 = []
        for j in range(m):
            try:
                W0.append(self.precomputing(rk.delta_eigs[j],is_W0 = True))
            except:
                W0.append(self.precomputing(rk.delta_eigs[j]))
        conv_hist = 1j*np.zeros((dof,m*N+1))
        sol = 1j*np.zeros((dof,m*N+1))
        counters = np.zeros(N)

        external_rho = None
        if same_rho:
            external_rho = 10**(-15*1.0/(4*N))
        external_L = None
        if same_L:
            external_L = 2*N
         
        for j in range(1,N+1):
            ## Calculating solution at timepoint tj
            start_ts = time.time()
            lconv = conv_hist[:,(j-1)*m+1:j*m+1]
            sol[:,(j-1)*m+1:j*m+1] = (self.time_step(W0,j-1,rk,sol[:,:rk.m*(j-1)+1],lconv,tolsolver=tolsolver))
            if np.isnan(sol[:,(j-1)*m+1:j*m+1]).any():
                print("NAN value in solution detected, norms until this point: ", np.linalg.norm(sol[:,:(j-1)*m+1],axis = 0))
                print("Time index = ",j-1, " N = ",N)
                raise ValueError("NAN is detected in solution.")
            end_ts   = time.time() 
            if debug_mode:
               print("Computed new step, relative progress: "+str((j-1)*1.0/N)+". Time taken: "+str(np.round((end_ts-start_ts)*1.0/60.0,decimals = 3))+" Min. ||x(t_j)|| = "+str(np.linalg.norm(sol[:,(j-1)*m+1:j*m+1])))
            ## Calculating Local History:
            currLen = math.gcd(2**j,j)
            localHist = np.concatenate((sol[:,m*j+1-m*currLen:m*j+1],np.zeros((dof,m*currLen))),axis=1)
            localconvHist = (self.tdForward.apply_RKconvol(localHist,(len(localHist[0,:]))*tau/m,method = method,factor_laplace_evaluations=factor_laplace_evaluations,external_rho=external_rho,external_L = external_L,prolonge_by=0,show_progress=False,first_value_is_t0=False))
            ## Updating Global History: 
            currLenCut = min(currLen,N-j)
            conv_hist[:,j*m+1:j*m+1+currLenCut*m] += localconvHist[:,currLen*m:currLen*m+currLenCut*m]
        #if debug_mode: 
            #print("N: ",N," m: ",rk.m," 2*m*N ",2*m*N, " Amount evaluations: ",self.countEv)
        #print("N: ",N," m: ",rk.m," 2*m*N ",2*m*N, " Amount evaluations: ",self.countEv)
        return np.real(sol) ,counters

