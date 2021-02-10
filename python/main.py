import numpy as np
import matplotlib.pyplot as plt
from math import *

import global_params
import exact_solutions
from rrp_solvers import *

### MECHANICAL PARAMETERS ########
#lamb1 = 10e9 #Pa
#mu1 = 10e9 # Pa
#lamb2 = 20e9 #Pa
#mu2 = 20e9 # Pa
#lamb3 = 10e9 #Pa
#mu3 = 10e9 # Pa
#lamb4 = 20e9 #Pa
#mu4 = 20e9 # Pa

## CU-SN PARAMS
## Sn
lamb1 = 47.27e9 #Pa
mu1 = 18.38e9 # Pa
##Cu6Sn5
lamb2 = 63.16e9 #Pa
mu2 = 45.73e9 # Pa
##Cu3Sn
lamb3 = 97.06e9 #Pa
mu3 = 50e9 # Pa
##Cu
lamb4 = 103.08e9 #Pa
mu4 = 48.51e9 # Pa

lamb = [lamb1,lamb2,lamb3,lamb4]
mu = [mu1,mu2,mu3,mu4]

#p0= 0.#-10e6 # pressure, pa
k_coup = 10.0e-8
Temperature = global_params.Temperature

#eigenstrains
eig_12 = -0.00327 # dilatational eigenstrain
eig_23 = -0.0279  #
eig = [eig_12,eig_23]

eps_z = 0.0

## initial interface position
sx0 = 0   # 
sx1 = 10e-6  #Sn-Cu6Sn5 boundary
sx2 = 12e-6 # Cu6Sn5-Cu3Sn
sx3 = 14e-6  # Cu3Sn - Cu
sx4 = 20e-6  ##external boundary

sx=[sx0,sx1,sx2,sx3,sx4]
## concentration of Cu
c0 = 0.0
c1m = 0.0
c1p = 0.541
c2m = 0.549
c2p = 0.755
c3m = 0.765
c3p = 1.0
c4 = 1.0
c_ = [c0,c1m,c1p,c2m,c2p,c3m,c3p,c4]
R_gas = global_params.R_gas
# Values taken from [Mei, 1992]
D_01s = 6.41e-11
D_12s = 1.84e-9*exp(-53.92e3/(R_gas*(273+Temperature)))   #3.10e-15 ## m^2/s
D_23s = 5.48e-9*exp(-61.86e3/(R_gas*(273+Temperature)))##1.19e-15
D_34s= 2.07e-24


def D12_update(trace):
    Q12= 53.92e3 - k_coup*trace
    ans = 1.84e-9*exp(-(Q12)/(R_gas*(273+Temperature)))
    return ans
def D23_update(trace):
    Q23= 61.86e3 - k_coup*trace
    ans = 5.48e-9*exp(-(Q23)/(R_gas*(273+Temperature)))
    return ans


def test_mech(label):
    x_=[0.0,4.0,10.0,16.0,20.0]
    lamb_ = [1.0,10.0,2.0,1.0]
    mu_ = [0.5,5.0,4.0,0.5]
    eig_ =[1.0,2.0]
    p_external = 0.0

    
    if label == 'press':
        assert abs(p_external) > 10e-15 #"Pressure equals zero"
        msh_ex1, u_ex1 = exact_solutions.ExactPressure(p_external, lamb_ ,mu_,x_)
    if label == 'eig':
        msh_ex1, u_ex1 = exact_solutions.ExactEigenstrain(eig_, lamb_ ,mu_,x_)
    
    num_mesh, num_ur, num_S = Solver_Mec(x_, #coordinates 
                                    40,  #mesh number
                                    lamb_, #lamb
                                    mu_, #mu
                                    eig_, #eigenstrains 
                                    0.0, #epsilon_z initial
                                    p_external,
                                    iter = False)  #external pressure 

    plt.plot(msh_ex1, u_ex1, '-g', label = 'exact')
    plt.plot(num_mesh, num_ur,'-.b', label='num')
    ymin = num_ur.min()
    ymax = num_ur.max()

    plt.vlines(x_[1:-1], ymin, ymax, linestyles='dashed', color='red', label='interfaces')
    plt.grid()
    plt.legend()
    plt.show()
    return

#test_mech('eig')


