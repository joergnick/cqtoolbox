#import psutil
import time
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import numpy as np
from linearcq import Conv_Operator
from rkmethods import Extrapolator,RKMethod
class AbstractIntegratorSparse:
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
            dof = len(self.righthandside(0))
        except:
            dof = 1
        ## Actual solving:
        W0 = []
        for j in range(m):
            try:
                W0.append(self.precomputing(rk.delta_eigs[j],is_W0 = True))
            except:
                W0.append(self.precomputing(rk.delta_eigs[j]))
        conv_hist = np.zeros((dof,m*N+1))
        sol = np.zeros((dof,1+(m*N)))
        prolonged_history = np.zeros((dof,(m*N)))
        counters = np.zeros(N)
        lconv = np.zeros((dof,m))
        extr = Extrapolator()
        for j in range(0,N):
            ## Calculating solution at timepoint tj
            print("Timepoint : ",j)
            start_ts = time.time()
            sol[:,j*m+1:(j+1)*m+1] = self.time_step(W0,j,rk,sol[:,:rk.m*(j)+1],lconv,tolsolver=tolsolver)
            if j==N-1:
                break
            end_ts   = time.time() 
            if debug_mode:
                print("Computed new step, relative progress: "+str(j*1.0/N)+". Time taken: "+str(np.round((end_ts-start_ts)*1.0/60.0,decimals = 3))+" Min. ||x(t_j)|| = "+str(np.linalg.norm(sol[:,j*m+1:(j+1)*m+1])))
            ## Calculating Local History:
            history = sol[:,1:]
            prolonge_by = 1000
            if (N>=1): 
                n_end = min(m*N, m*(j+1)+prolonge_by)
                prolonged_history[:,:n_end]  = extr.prolonge_towards_0(history[:,:m*(j+1)],prolonge_by,rk,decay_speed=4)[:,:n_end]
            if j<50:
                cutoff = 10**(-15)
            else:
                cutoff = 10**(-15)
            show_progress = False
            if j % 100 == 0:
                show_progress = True
            globalconvHist = np.real(self.tdForward.apply_RKconvol(prolonged_history,T,cutoff = cutoff,method = method,factor_laplace_evaluations=factor_laplace_evaluations,prolonge_by=0,show_progress=show_progress))
            W0un = rk.diagonalize(prolonged_history[:,m*(j+1):m*(j+2)]) 
            for stage_ind in range(m):
                W0un[:,stage_ind] = self.harmonic_forward(rk.delta_eigs[stage_ind],W0un[:,stage_ind],precomp = W0[stage_ind])
            W0un = np.real(rk.reverse_diagonalize(W0un))

            lconv = globalconvHist[:,m*(j+1):m*(j+2)]-W0un
            #lconv = globalconvHist[:,m*(j+1)+1:m*(j+2)+1]
                #localconvHist = np.real(self.tdForward.apply_RKconvol(localHist,(len(localHist[0,:]))*tau/m,method = method,factor_laplace_evaluations=factor_laplace_evaluations,prolonge_by=0,show_progress=False))
            ## Updating Global History: 
        #if debug_mode: 
            #print("N: ",N," m: ",rk.m," 2*m*N ",2*m*N, " Amount evaluations: ",self.countEv)
        print("N: ",N," m: ",rk.m," 2*m*N ",2*m*N, " Amount evaluations: ",self.countEv)
        return sol[:,:m*N+1] ,counters