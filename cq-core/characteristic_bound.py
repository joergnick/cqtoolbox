from rkmethods import RKMethod
import numpy as np

N = 10000000
T = 1
tau = T*1.0/N
rk = RKMethod("RadauIIA-5",tau)

invA = np.linalg.inv(rk.A)
def Delta(zeta):
    return invA-zeta*invA.dot(np.ones((rk.m,1))).dot(rk.b).dot(invA)

def invDelta(zeta):
    return rk.A+zeta*1.0/(1-zeta)*np.ones((rk.m,1)).dot(rk.b)

e_m = np.zeros((1,rk.m))
e_m[0,rk.m-1] = 1
tt = np.linspace(-10,10,10000)
#tt = np.linspace(0,2*np.pi,10000)
sigma = 1/T
points_inner_circle = np.exp(-(sigma +1j*tt)*tau)
#points_inner_circle = 0.9999999999*np.exp(1j*tt)
evals = np.zeros(len(tt))
#B_diag = np.diag(rk.b[0,:]**(-0.5)*rk.c[:]**(-1))+1j*0*np.diag(rk.b[0,:])
#B_diag = np.diag(rk.b[0,:]*(np.ones(rk.m)-rk.c[:]))+1j*0*np.diag(rk.b[0,:])
B_diag = np.diag(rk.b[0,:])
#print(B_diag)
#B_inv12 = np.diag([B_diag[i,i]**(-0.5) for i in range(rk.m)])

#B_diag = np.diag(rk.b[0,:]*rk.c[:]**(-1))
for index,point in enumerate(points_inner_circle):
    #evals[index] = np.linalg.norm(  Delta(np.conj(point)).T.dot(invDelta(point)))
    #evals[index] = np.linalg.norm(  np.conj(invDelta(point)).T.dot(Delta(point)))
    #evals[index] = np.linalg.norm(np.conj(Delta(point)).T.dot(B_diag).dot(invDelta(point)))
    s = (sigma +1j*tt[index])
    e_diag = np.exp(s*(rk.c-1)*tau)
    #print("e_diag = ",e_diag)
    evals[index] = np.linalg.norm(e_m.dot(Delta(point).dot(Delta(point))/tau**2).dot(e_diag))
    #hermitian = 0.5*B_inv12.dot(B_diag.dot(Delta(point))+Delta(np.conj(point)).T.dot(B_diag)).dot(B_inv12)
    #evals[index] = min(np.real(np.linalg.eigvals(hermitian)))
point = 0
#print(0.5*B_inv12.dot(B_diag.dot(Delta(point))+Delta(np.conj(point)).T.dot(B_diag)).dot(B_inv12))
print("MAX EVALS: ",max(evals))
#print(Delta(0.99).dot(invDelta(0.99)))
#print(np.ones((rk.m,1)).dot(rk.b))
#print(rk.b.dot(invA))
#print(rk.b.dot(invA).dot( np.eye(rk.m)-1*np.ones((rk.m,1)).dot(rk.b).dot(invA) ) )
#print(np.ones((rk.m,1)).dot(rk.b).dot(invA))