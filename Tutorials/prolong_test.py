import numpy as np
import sys
sys.path.append('cqToolbox')
sys.path.append('../cqToolbox')
from rkmethods import Extrapolator,RKMethod
tau = 0.1
rk   = RKMethod("RadauIIA-1",tau)
extr = Extrapolator()
for j in range(1,2):
    x = np.ones((1,j*rk.m))
    prolonged_x  = extr.prolonge_towards_0(x,20,rk,decay_speed=10)
import matplotlib.pyplot as plt
plt.plot(prolonged_x[0,:])
plt.show()