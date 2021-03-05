import numpy as np
import matplotlib.pyplot as plt
from math import *

import global_params
import exact_solutions
from rrp_solvers import *

### MECHANICAL PARAMETERS ########

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

p0= 0.0#-10e6 # pressure, pa
k_coup = 0.0#10.0e-8
Temperature = global_params.Temperature

#eigenstrains
eig_12 = -0.00327 # dilatational eigenstrain
eig_23 = -0.0279  #
eig = [eig_12,eig_23]

eps_z = 0.0

## initial interface position
sx0 = 0   # 
sx1 = 15e-6  #Sn-Cu6Sn5 boundary
sx2 = 17e-6 # Cu6Sn5-Cu3Sn
sx3 = 20e-6  # Cu3Sn - Cu
sx4 = 25e-6  ##external boundary
sx=[sx0,sx1,sx2,sx3,sx4]


## concentration of Cu
c0 = 0.0
c1m = 0.0
c1p = 0.541#0.4#0.541
c2m = 0.549#0.41#0.549
c2p = 0.755#0.605#0.755
c3m = 0.765#0.623#0.765
c3p = 0.993#1.0
c4 = 0.993#1.0
c_ = [c0,c1m,c1p,c2m,c2p,c3m,c3p,c4]

R_gas = global_params.R_gas
# Values taken from [Mei, 1992]
D_01s = 6.41e-11
D_12s = 1.84e-9*exp(-53.92e3/(R_gas*(273+Temperature)))  ## m^2/s
D_23s = 5.48e-9*exp(-61.86e3/(R_gas*(273+Temperature)))
D_34s= 2.07e-24
D_ = [D_01s, D_12s, D_23s, D_34s]



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

def animate_gif(mesh_total, ur_total, x1_new, x2_new, x3_new, sg ):

    #### animation
    counter = 0
    filenames = []
    for s in range(0, mesh_total[:,1].size, 10):
        hours = 10*60*s/3600

        interface0 = [sx0, sx1, sx1,  sx2, sx2,  sx3, sx3,  sx4 ]
        interface = [sx0, x1_new[s], x1_new[s],  x2_new[s], x2_new[s],  x3_new[s], x3_new[s],  sx4 ]
        profile = [c0, c1m,  c1p, c2m, c2p, c3m ,c3p, c4]

        plt.plot(interface0,profile, '-.r', label='Start')
        plt.plot(interface,profile, label='T=%4.2f hrs' % hours)
        plt.xlabel("$r, \quad [m]$")
        plt.ylabel("Concentration of Cu")
        plt.title('Concentration profile: \n Sn,       Cu6Sn5,       Cu3Sn,       Cu.')

        '''
        ymin = ur_total.min()
        ymax = ur_total.max()
        plt.plot(mesh_total[0,:],ur_total[0,:], '-b', label='initial',
        )
        plt.vlines([x1_new[0],x2_new[0],x3_new[0]], ymin, ymax, linestyles='dashed', color='green')    
        
        plt.vlines([x1_new[s],x2_new[s],x3_new[s]], ymin, ymax, linestyles='dashed', color='black', label='interfaces')
        
        
         plt.plot(mesh_total[s,:],ur_total[s,:], '-r', label='T=%4.2f hrs' % hours)
         '''
        
        counter += 1
        plt.legend()
        filename = 'img//tmp%04d.png' % counter
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)
        #time.sleep(0.2)

    import imageio
    import os
    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    # Remove files
    for filename in set(filenames):
        os.remove(filename)


def plot_profile(x1_new,x2_new,x3_new, x1_new2,x2_new2,x3_new2):
    interface0 = [sx0, sx1, sx1,  sx2, sx2,  sx3, sx3,  sx4 ]
    interface = [sx0, x1_new[-1], x1_new[-1],  x2_new[-1], x2_new[-1],  x3_new[-1], x3_new[-1],  sx4 ]
    interface2 = [sx0, x1_new2[-1], x1_new2[-1],  x2_new2[-1], x2_new2[-1],  x3_new2[-1], x3_new2[-1],  sx4 ]
    profile = [c0, c1m,  c1p, c2m, c2p, c3m ,c3p, c4]

    plt.plot(interface0,profile, '-.r', label='Start')
    plt.plot(interface,profile, label='T= 28 hours')
    plt.plot(interface2,profile,'-g', label='big dt')
    plt.xlabel("$r, \quad [m]$")
    plt.ylabel("Concentration of Cu")
    plt.title('Concentration profile: \n Sn,       Cu6Sn5,       Cu3Sn,       Cu.')
    plt.legend()
    plt.show()



def plot_growth(T, dt, x1_new,x2_new,x3_new):
    t_array = np.linspace(0, T, x1_new.size)
    power = 1.0
    plt.plot(np.power(t_array,power), x2_new - x1_new, '-.r', label='Cu6Sn5')
    plt.plot(np.power(t_array,power), x3_new - x2_new, label='Cu3Sn')
    plt.plot(np.power(t_array,power), x3_new - x1_new,'-g', label='Total')
    plt.xlabel('time to the power of %f' % power)
    plt.ylabel("Thickness")
    plt.legend()
    plt.grid()
    plt.show()



#####################################
#####################################
#####################################


mesh_total, ur_total, x1_new, x2_new, x3_new, sg = Solver(60, #dt
                                                        28*60*60, # T
                                                        20, #mesh_number 
                                                        p0, #pressure 
                                                        k_coup, #coupling coefficient
                                                        sx,  #interfaces
                                                        c_, #concentration
                                                        D_, #initial diff coeffs
                                                        lamb,# 
                                                        mu,  #
                                                        eig, #eigenstrains
                                                        0.0, #ez0
                                                        iter = True) #Iters for ez 


mesh_total2, ur_total2, x1_new2, x2_new2, x3_new2, sg2 = Solver(10*60, #dt
                                                        28*60*60, # T
                                                        20, #mesh_number 
                                                        p0, #pressure 
                                                        k_coup, #coupling coefficient
                                                        sx,  #interfaces
                                                        c_, #concentration
                                                        D_, #initial diff coeffs
                                                        lamb,# 
                                                        mu,  #
                                                        eig, #eigenstrains
                                                        0.0, #ez0
                                                        iter = True) #Iters for ez 


#animate_gif(mesh_total, ur_total, x1_new, x2_new, x3_new, sg)
#test_mech('eig')
#plot_profile(x1_new,x2_new,x3_new, x1_new2,x2_new2,x3_new2)
#print(x3_new[-1]-x1_new[-1])

plot_growth(10*60, 28*60*60 , x1_new,x2_new,x3_new)

##### Errors to fix

##### Computation of stresses
####division by zero when compute stresses


#print(B12)
#plt.plot(x1, J1, 'r')
#plt.plot(x2, J2, 'b')
#plt.ylim(0.54,0.55)
#plt.show()
