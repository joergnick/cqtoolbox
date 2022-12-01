import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')

import numpy as np

from linearcq import Conv_Operator
from rkmethods import RKMethod
from linearcq import Conv_Operator

T = 1
N = 100
method = "RadauIIA-2"
rk = RKMethod(method,T*1.0/N)
rhs = rk.get_time_points(T)**8




def th_deriv(s,b):
    return s**(0.5)*b
deriv = Conv_Operator(th_deriv)

sol = deriv.apply_RKconvol(rhs,T,method = method)
ex  = 1.0/9*rk.get_time_points(T)**9
print(np.max(np.abs(sol-ex[1:])))
print(rhs)