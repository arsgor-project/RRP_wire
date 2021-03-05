import numpy as np
import matplotlib.pyplot as plt
from math import *

Temperature = 215
R_gas = 8.317462 

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
c4 = 1.0#0.993#1.0
c_ = [c0,c1m,c1p,c2m,c2p,c3m,c3p,c4]

# Values taken from [Mei, 1992]
D_01s = 6.41e-11
D_12s = 1.84e-9*exp(-53.92e3/(R_gas*(273+Temperature)))  ## m^2/s
D_23s = 5.48e-9*exp(-61.86e3/(R_gas*(273+Temperature)))
D_34s= 2.07e-24
D_ = [D_01s, D_12s, D_23s, D_34s]

#print(D_12s, 1.5e7*1e-6/(24*60*60))

def Solver(dt_, T_, coordinates, concentration):

    sx0,sx1,sx2,sx3,sx4 = coordinates
    c0,c1m,c1p,c2m,c2p,c3m,c3p,c4 = concentration

    dt = dt_
    t_num = int(T_/dt)

    x1_new = np.zeros(t_num+1)
    x2_new = np.zeros(t_num+1)
    x3_new = np.zeros(t_num+1)
    
    x1_new[0] = sx1
    x2_new[0] = sx2
    x3_new[0] = sx3

    D_01 = D_01s
    D_12 = D_12s
    D_23 = D_23s
    D_34 = D_34s

    for t_i in range(t_num):

        from scipy import special

        x1_new[t_i+1] =  x1_new[t_i] + dt/(c1p-c1m)*(-D_12*(c2m-c1p))/(x2_new[t_i]-x1_new[t_i])
        x2_new[t_i+1] =  x2_new[t_i] + dt/(c2p-c2m)*(D_12*(c2m-c1p)/(x2_new[t_i]-x1_new[t_i]) - D_23*(c3m-c2p)/(x3_new[t_i]-x2_new[t_i]) )
        x3_new[t_i+1] =  x3_new[t_i] + dt/(c3p-c3m)*( D_23*(c3m-c2p)/(x3_new[t_i]-x2_new[t_i]) -D_34*(c4-c3p)/(sx4 - x3_new[t_i]))
        #check
        if (x1_new[t_i+1] - sx0)<0:
            print("ERROR   x1 < x0", t_i+1)
            break
        if (x2_new[t_i+1] - x1_new[t_i+1])<0:
            print("ERROR   x2 < x1", t_i+1)
            break
        if (x3_new[t_i+1] - x2_new[t_i+1])<0:
            print("ERROR   x3 < x2", t_i+1)
            break
        if (sx4 - x3_new[t_i+1])<0:
            print("ERROR   x4 < x3", t_i+1)
            break
            
        x0 = sx0
        x1 = x1_new[t_i+1]
        x2 = x2_new[t_i+1]
        x3 = x3_new[t_i+1]
        x4 = sx4
        
    return x1_new, x2_new, x3_new

def Solver2(dt_, T_, time_init,  coordinates, concentration):

    sx0,sx1,sx2,sx3,sx4 = coordinates
    c0,c1m,c1p,c2m,c2p,c3m,c3p,c4 = concentration

    dt = dt_
    t_num = int(T_/dt)

    x1_new = np.zeros(t_num+1)
    x2_new = np.zeros(t_num+1)
    x3_new = np.zeros(t_num+1)
    
    x1_new[0] = sx1
    x2_new[0] = sx2
    x3_new[0] = sx3

    D_01 = D_01s
    D_12 = D_12s
    D_23 = D_23s
    D_34 = D_34s

    for t_i in range(t_num):

        from scipy import special

        t0 = time_init + dt_*t_i
        #аx1 = np.linspace(sx1, sx2, 100)
        B12 = (c2m-c1p)/(special.erf(x1_new[t_i]/(2*sqrt(D_12s*t0))) - special.erf(x2_new[t_i]/(2*sqrt(D_12s*t0))) )
        A12 = c1p + B12*special.erf(x1_new[t_i]/(2*sqrt(D_12s*t0)))
        #y1 = A12 - B12*special.erf(x1/(2*sqrt(D_12s*t0)))
        J1 = -B12*np.exp(-x1_new[t_i]**2/(4*D_12s*t0))/(2*sqrt(D_12s*t0))
        J2 = -B12*np.exp(-x2_new[t_i]**2/(4*D_12s*t0))/(2*sqrt(D_12s*t0))


        #x2 = np.linspace(sx2, sx3, 100)
        B23 = (c3m-c2p)/(special.erf(x2_new[t_i]/(2*sqrt(D_23s*t0))) - special.erf(x3_new[t_i]/(2*sqrt(D_23s*t0))) )
        A23 = c2p + B23*special.erf(x2_new[t_i]/(2*sqrt(D_23s*t0)))
        #y2 = A23 - B23*special.erf(x2/(2*sqrt(D_23s*t0)))

        J3 = -B23*np.exp(-x2_new[t_i]**2/(4*D_23s*t0))/(2*sqrt(D_23s*t0))
        J4 = -B23*np.exp(-x3_new[t_i]**2/(4*D_23s*t0))/(2*sqrt(D_23s*t0))

        x1_new[t_i+1] =  x1_new[t_i] + dt/(c1p-c1m)*(-D_12*J1)
        x2_new[t_i+1] =  x2_new[t_i] + dt/(c2p-c2m)*(D_12*J2 - D_23*J3)
        x3_new[t_i+1] =  x3_new[t_i] + dt/(c3p-c3m)*(D_23*J4)
        
        #x1_new[t_i+1] =  x1_new[t_i] + dt/(c1p-c1m)*(-D_12*(c2m-c1p))/(x2_new[t_i]-x1_new[t_i])
        #x2_new[t_i+1] =  x2_new[t_i] + dt/(c2p-c2m)*(D_12*(c2m-c1p)/(x2_new[t_i]-x1_new[t_i]) - D_23*(c3m-c2p)/(x3_new[t_i]-x2_new[t_i]) )
        #x3_new[t_i+1] =  x3_new[t_i] + dt/(c3p-c3m)*( D_23*(c3m-c2p)/(x3_new[t_i]-x2_new[t_i]) )

        #check
        if (x1_new[t_i+1] - sx0)<0:
            print("ERROR   x1 < x0", t_i+1)
            break
        if (x2_new[t_i+1] - x1_new[t_i+1])<0:
            print("ERROR   x2 < x1", t_i+1)
            break
        if (x3_new[t_i+1] - x2_new[t_i+1])<0:
            print("ERROR   x3 < x2", t_i+1)
            break
        if (sx4 - x3_new[t_i+1])<0:
            print("ERROR   x4 < x3", t_i+1)
            break
            
        x0 = sx0
        x1 = x1_new[t_i+1]
        x2 = x2_new[t_i+1]
        x3 = x3_new[t_i+1]
        x4 = sx4
        
    return x1_new, x2_new, x3_new

def Solver3(dt_, T_,  coordinates, concentration):

    sx0,sx1,sx2,sx3,sx4 = coordinates
    c0,c1m,c1p,c2m,c2p,c3m,c3p,c4 = concentration

    dt = dt_
    t_num = int(T_/dt)

    x1_new = np.zeros(t_num+1)
    x2_new = np.zeros(t_num+1)
    x3_new = np.zeros(t_num+1)
    
    x1_new[0] = sx1
    x2_new[0] = sx2
    x3_new[0] = sx3

    D_01 = D_01s
    D_12 = D_12s
    D_23 = D_23s
    D_34 = D_34s

    for t_i in range(t_num):

        B12 = (c2m-c1p)/(log(x1_new[t_i]-log(x2_new[t_i])))
        
        J1 = D_12*B12/x1_new[t_i]
        J2 = D_12*B12/x2_new[t_i]

        B23 = (c3m-c2p)/(log(x2_new[t_i]-log(x3_new[t_i])))
        
        J3 = D_23*B23/x2_new[t_i]
        J4 = D_23*B23/x3_new[t_i]


        x1_new[t_i+1] =  x1_new[t_i] + dt/(c1p-c1m)*(-J1)
        x2_new[t_i+1] =  x2_new[t_i] + dt/(c2p-c2m)*(J2 - J3)
        x3_new[t_i+1] =  x3_new[t_i] + dt/(c3p-c3m)*(J4)
        
        #check
        if (x1_new[t_i+1] - sx0)<0:
            print("ERROR   x1 < x0", t_i+1)
            break
        if (x2_new[t_i+1] - x1_new[t_i+1])<0:
            print("ERROR   x2 < x1", t_i+1)
            break
        if (x3_new[t_i+1] - x2_new[t_i+1])<0:
            print("ERROR   x3 < x2", t_i+1)
            break
        if (sx4 - x3_new[t_i+1])<0:
            print("ERROR   x4 < x3", t_i+1)
            break
            
        x0 = sx0
        x1 = x1_new[t_i+1]
        x2 = x2_new[t_i+1]
        x3 = x3_new[t_i+1]
        x4 = sx4
        
    return x1_new, x2_new, x3_new


def animate_gif(time ,x1_new, x2_new, x3_new):

    #### animation
    counter = 0
    filenames = []
    for s in range(0, time[:].size, int(time[:].size/30)):
        hours = time[s]/(60*60)

        interface0 = [sx0, sx1, sx1,  sx2, sx2,  sx3, sx3,  sx4 ]
        interface = [sx0, x1_new[s], x1_new[s],  x2_new[s], x2_new[s],  x3_new[s], x3_new[s],  sx4 ]
        profile = [c0, c1m,  c1p, c2m, c2p, c3m ,c3p, c4]

        plt.plot(interface0,profile, '-.r', label='Start')
        plt.plot(interface,profile, label='T=%2.2f hrs' % hours)
        plt.xlabel("$r, \quad [m]$")
        plt.ylabel("Concentration of Cu")
        plt.title('Concentration profile: \n Sn,       Cu6Sn5,       Cu3Sn,       Cu.')

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
    #for filename in set(filenames):
    #    os.remove(filename)


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




T_max = 3.5*24*60*60
x_new = Solver(60, T_max, sx, c_)
x_new2 = Solver3(60, T_max, sx, c_)                    

#x_new2 = Solver2(60, T_max, 2*60*60, sx, c_)                    
#x_new3 = Solver2(60, 28*60*60, 4*60*60, sx, c_)
#x_new4 = Solver2(60, 28*60*60, 0.5*60*60, sx, c_)

#plot_growth(60, 28*60*60 , x_new[0],x_new[1],x_new[2])

t_array = np.linspace(0, T_max, x_new[0].size)
power = 1.0

'''
#plt.style.use('')
#plt.plot(np.power(t_array,power), x2_new - x1_new, '-.r', label='Cu6Sn5')
#plt.plot(np.power(t_array,power), x3_new - x2_new, label='Cu3Sn')
plt.plot(np.power(t_array/(24*60*60),power), x_new[2] - x_new[1], label='Linear')
#plt.plot(np.power(t_array,power), x_new2[2] - x_new2[1], label='Mei, 1 hour')
#plt.plot(np.power(t_array,power), x_new3[2] - x_new3[1],label='Mei, 2 hours')
#plt.plot(np.power(t_array,power), x_new4[2] - x_new4[1],label='Mei, 0.5 hours')

plt.xlabel('time to the power of %f' % power)
plt.ylabel("Thickness")
plt.legend()
plt.grid()
plt.show()
'''

#animate_gif(t_array, x_new[0], x_new[1], x_new[2])

interface0 = [sx0, sx1, sx1,  sx2, sx2,  sx3, sx3,  sx4 ]
interface = [sx0, x_new[0][-1], x_new[0][-1],  x_new[1][-1], x_new[1][-1],  x_new[2][-1], x_new[2][-1],  sx4 ]
interface2 = [sx0, x_new2[0][-1], x_new2[0][-1],  x_new2[1][-1], x_new2[1][-1],  x_new2[2][-1], x_new2[2][-1],  sx4 ]

profile = [c0, c1m,  c1p, c2m, c2p, c3m ,c3p, c4]

plt.plot(interface0,profile, '-.r', label='Start')

hours = t_array[-1]/(60*60)
plt.plot(interface,profile, label='T= %2.2f hours' % hours)
plt.plot(interface2,profile, label='T= %2.2f hours, Cylindrical' % hours)

plt.xlabel("$r, \quad [m]$")
plt.ylabel("Concentration of Cu")
plt.title('Concentration profile: \n Sn,       Cu6Sn5,       Cu3Sn,       Cu.')
plt.legend()
plt.savefig('fig.png', dpi=100)

