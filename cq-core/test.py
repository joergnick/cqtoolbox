
import numpy as np
import math

A=np.array([[11.0/45-7*math.sqrt(6)/360, 37.0/225-169.0*math.sqrt(6)/1800 , -2.0/225+math.sqrt(6)/75],
          [37.0/225+169.0*math.sqrt(6)/1800,11.0/45+7*math.sqrt(6)/360,-2.0/225-math.sqrt(6)/75],
          [4.0/9-math.sqrt(6)/36,4.0/9+math.sqrt(6)/36,1.0/9]])
c=np.array([2.0/5-math.sqrt(6)/10,2.0/5+math.sqrt(6)/10,1])
m = len(c)
#b=np.array([[4.0/9-math.sqrt(6)/36,4.0/9+math.sqrt(6)/36,1.0/9]])
b= np.array([A[m-1,:]])

##########
c3 = c
CP3 = np.array([[ 1, c3[0], c3[0]**2],
                [ 1, c3[1], c3[1]**2],
                [ 1, c3[2], c3[2]**2],])
CQ3 = np.array([[c3[0], (c3[0]**2)/2.0, (c3[0]**3)/3.0],
                [c3[1], (c3[1]**2)/2.0, (c3[1]**3)/3.0],
                [c3[2], (c3[2]**2)/2.0, (c3[2]**3)/3.0],])

def construct_A(c):
    m  = len(c)
    CPm = np.ones((m,m))
    CQm = np.zeros((m,m))
    for j in range(m-1):
        CPm[:,j+1] = c**(j+1)
    for j in range(m):
        CQm[:,j] = c**(j+1)/(1.0*(j+1))
    return CQm.dot(np.linalg.inv(CPm))
#A3 = CQ3.dot(np.linalg.inv(CP3))
c = np.array([1.0/3,1])
A3 = construct_A(c)
print(np.array([[5.0/12,-1.0/12],[3.0/4,1.0/4]])) 
print(A3)


