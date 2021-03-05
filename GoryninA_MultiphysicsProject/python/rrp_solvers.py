import numpy as np
from global_params import *
from math import *
    
def Solver_Mec(coordinates, mesh_number, lamb, mu, eig, ez0, p_, iter):
    p0=p_
    # number of point in each region, exluding endpoint point
    N1 = mesh_number
    N2 = mesh_number
    N3 = mesh_number
    N4 = mesh_number
    N = N1+N2+N3+N4

    x0 = coordinates[0]   # 
    x1 = coordinates[1]  #Sn-Cu6Sn5 boundary
    x2 = coordinates[2]  # Cu6Sn5-Cu3Sn
    x3 = coordinates[3]  # Cu3Sn - Cu
    x4 = coordinates[4]  ##external boundary

    mesh_1= np.linspace(x0,x1, N1, endpoint=False)
    mesh_2= np.linspace(x1,x2, N2, endpoint=False)
    mesh_3= np.linspace(x2,x3, N3, endpoint=False)
    mesh_4= np.linspace(x3,x4, N4)
    mesh  = np.concatenate((mesh_1,mesh_2, mesh_3, mesh_4))
    X_out = x4 #1e-6m 

    lamb1, lamb2, lamb3, lamb4 = lamb
    mu1,mu2,mu3,mu4 = mu
    eig_12, eig_23 = eig

    #size step in each zone
    dx1 = mesh_1[1]-mesh_1[0]
    dx2 = mesh_2[1]-mesh_2[0]
    dx3 = mesh_3[1]-mesh_3[0]
    dx4 = mesh_4[1]-mesh_4[0]

    # N-1 is the endpoint of the array
    ur = np.zeros(N)

    eps_z = ez0
    check = 1.0
    ## secondary variables
    bc1 = N1
    bc2 = N1+N2
    bc3 = N1+N2+N3

    while (check > 1e-5):
        # tridiagonal coefficients
        left = np.zeros(N)
        center = np.zeros(N)
        right = np.zeros(N) 
        b = np.zeros(N)  ## rhs vector

        for i in range(1,N-1):
            xc = mesh[i]
            dxp = mesh[i+1]-mesh[i]
            dxm = mesh[i] - mesh[i-1]

            left[i] = xc**2/(dxm*0.5*(dxm+dxp)) - xc/(dxm+dxp)
            center[i] = -xc**2/(0.5*(dxm+dxp))*(1./dxp+1./dxm) - 1.0
            right[i] = xc**2/(dxp*0.5*(dxm+dxp)) + xc/(dxm+dxp)

        center[0] = 1.0  ## ur(0)=0

        
        left[bc1] = -(lamb1+2*mu1)/(mesh[bc1]-mesh[bc1-1])
        center[bc1] = (lamb2+2*mu2)/(mesh[bc1+1]-mesh[bc1]) + (lamb1+2*mu1)/(mesh[bc1]-mesh[bc1-1]) + (lamb1-lamb2)/mesh[bc1]
        right[bc1] = -(lamb2+2*mu2)/(mesh[bc1+1]-mesh[bc1])
        b[bc1] = -(3*lamb2+2*mu2)*(eig_12) + (lamb2-lamb1)*eps_z


        left[bc2] = -(lamb2+2*mu2)/(mesh[bc2]-mesh[bc2-1])
        center[bc2] =  (lamb3+2*mu3)/(mesh[bc2+1]-mesh[bc2]) + (lamb2+2*mu2)/(mesh[bc2]-mesh[bc2-1])  + (lamb2-lamb3)/mesh[bc2]
        right[bc2] = -(lamb3+2*mu3)/(mesh[bc2+1]-mesh[bc2])
        b[bc2] = (3*lamb2+2*mu2)*eig_12 - (3*lamb3+2*mu3)*eig_23 + (lamb3-lamb2)*eps_z


        left[bc3] = -(lamb3+2*mu3)/(mesh[bc3]-mesh[bc3-1])
        center[bc3] = (lamb4+2*mu4)/(mesh[bc3+1]-mesh[bc3]) + (lamb3+2*mu3)/(mesh[bc3]-mesh[bc3-1])  + (lamb3-lamb4)/mesh[bc3]
        right[bc3] = -(lamb4+2*mu4)/(mesh[bc3+1]-mesh[bc3])
        b[bc3] = (3*lamb3+2*mu3)*(eig_23) + (lamb4-lamb3)*eps_z


        center[N-1] = (lamb4+2*mu4)/(mesh[N-1]-mesh[N-2]) + lamb4/mesh[N-1]
        left[N-1] = -(lamb4+2*mu4)/(mesh[N-1]-mesh[N-2])
        b[N-1] = p0 - lamb4*eps_z  

        ##прогонка
        # going down
        for i in range(1,N):
            m = left[i]/center[i-1]
            center[i] -= m*right[i-1]
            b[i] -= m*b[i-1] 

        ur[N-1] = b[N-1]/center[N-1]

        for i in range(N-2,-1,-1):
            ur[i] = (b[i] - right[i]*ur[i+1])/center[i]

        ## got a solution
        ## evaluate the integrals

        int1 = (lamb1)*(ur[bc1]*mesh[bc1] - ur[0]*mesh[0]) +\
               (lamb2)*(ur[bc2]*mesh[bc2] - ur[bc1]*mesh[bc1]) +\
               (lamb3)*(ur[bc3]*mesh[bc3] - ur[bc2]*mesh[bc2]) +\
               (lamb4)*(ur[N-1]*mesh[N-1] - ur[bc3]*mesh[bc3])
        int2 = (3*lamb2+2*mu2)*0.5*eig_12*(x2**2-x1**2) + (3*lamb3+2*mu3)*0.5*eig_23*(x3**2-x2**2)
        int3 = 0.5*(lamb4+2*mu4)*(x4**2 - x3**2) + 0.5*(lamb3+2*mu3)*(x3**2 - x2**2) +\
            0.5*(lamb2+2*mu2)*(x2**2 - x1**2) + 0.5*(lamb1+2*mu1)*(x1**2 - x0**2)

        Integral = (-int1 + int2)/int3
        check = abs(Integral-eps_z)
        eps_z = Integral
        
        if iter == False:
            check = 0.0
    

    ####postprocessing
    sg_r = np.zeros(N)
    sg_theta = np.zeros(N)
    sg_z = np.zeros(N)
    sg_trace = np.zeros(N) 

    for i in range(1,bc1,1):
        K1 = lamb1+2*mu1
        K2 = 3*lamb1+2*mu1
        du_rp = (ur[i+1]-ur[i])/(mesh[i+1]-mesh[i])
        sg_r[i] = K1*du_rp + lamb1*(ur[i]/mesh[i] + eps_z)
        sg_theta[i] = K1*ur[i]/mesh[i]+ lamb1*(du_rp + eps_z) 
        sg_z[i] = K1*eps_z + lamb1*(du_rp + ur[i]/mesh[i]) 

    for i in range(bc1,bc2,1):
        K1 = lamb2+2*mu2
        K2 = 3*lamb2+2*mu2
        du_rp = (ur[i+1]-ur[i])/(mesh[i+1]-mesh[i])
       
        sg_r[i] = K1*du_rp + lamb2*(ur[i]/mesh[i] + eps_z) - K2*eig_12  
        sg_theta[i] = K1*ur[i]/mesh[i]+ lamb2*(du_rp + eps_z) - K2*eig_12
        sg_z[i] = K1*eps_z + lamb2*(du_rp + ur[i]/mesh[i]) - K2*eig_12
    
    
    for i in range(bc2,bc3,1):
        K1 = lamb3+2*mu3
        K2 = 3*lamb3+2*mu3
        du_rp = (ur[i+1]-ur[i])/(mesh[i+1]-mesh[i])
        sg_r[i] = K1*du_rp + lamb3*(ur[i]/mesh[i] + eps_z) - K2*eig_23 
        sg_theta[i] = K1*ur[i]/mesh[i]+ lamb3*(du_rp + eps_z) - K2*eig_23
        sg_z[i] = K1*eps_z + lamb3*(du_rp + ur[i]/mesh[i]) - K2*eig_23
    
    for i in range(bc3,N-1,1):
        K1 = lamb4+2*mu4
        K2 = 3*lamb4+2*mu4
        du_rp = (ur[i+1]-ur[i])/(mesh[i+1]-mesh[i])
        sg_r[i] = K1*du_rp + lamb4*(ur[i]/mesh[i] + eps_z)  
        sg_theta[i] = K1*ur[i]/mesh[i]+ lamb4*(du_rp + eps_z)
        sg_z[i] = K1*eps_z + lamb4*(du_rp + ur[i]/mesh[i])     
    
    sg_r[N-1] = (lamb4+2*mu4)*(ur[N-1]-ur[N-2])/(mesh[N-1]-mesh[N-2]) + lamb4*(ur[N-1]/mesh[N-1] + eps_z) 
    sg_theta[N-1] = (lamb4+2*mu4)*ur[N-1]/mesh[N-1] + lamb4*((ur[N-1]-ur[N-2])/(mesh[N-1]-mesh[N-2]) + eps_z) 
    sg_z[N-1] = (lamb4+2*mu4)*eps_z + lamb4*((ur[N-1]-ur[N-2])/(mesh[N-1]-mesh[N-2]) + ur[N-1]/mesh[N-1] ) 

    sg_trace = sg_r + sg_theta + sg_z

    return mesh, ur, [sg_r, sg_theta, sg_z, sg_trace]

def D12_update(trace, k_coup):
    Q12= 53.92e3 - k_coup*trace
    ans = 1.84e-9*exp(-(Q12)/(R_gas*(273+Temperature)))
    return ans
def D23_update(trace, k_coup):
    Q23= 61.86e3 - k_coup*trace
    ans = 5.48e-9*exp(-(Q23)/(R_gas*(273+Temperature)))
    return ans


def Solver(dt_, T_, mesh_number, p_, k_coup, coordinates, concentration,  D_ ,lamb, mu, eig, ez0, iter):

    sx0,sx1,sx2,sx3,sx4 = coordinates
    c0,c1m,c1p,c2m,c2p,c3m,c3p,c4 = concentration

    p0 = p_
    dt = dt_
    t_num = int(T_/dt)

    x1_new = np.zeros(t_num+1)
    x2_new = np.zeros(t_num+1)
    x3_new = np.zeros(t_num+1)
    ur_total = np.zeros([t_num+1, mesh_number*4])
    mesh_total = np.zeros([t_num+1, mesh_number*4])

    x1_new[0] = sx1
    x2_new[0] = sx2
    x3_new[0] = sx3

    x0 = sx0
    x1 = x1_new[0]
    x2 = x2_new[0]
    x3 = x3_new[0]
    x4 = sx4
    
    D_01s, D_12s, D_23s, D_34s = D_
    D_01 = D_01s
    D_12 = D_12s
    D_23 = D_23s
    D_34 = D_34s

    # number of point in each region, exluding endpoint point
    N1_mech = mesh_number
    N2_mech = mesh_number
    N3_mech = mesh_number
    N4_mech = mesh_number
    N_mech = N1_mech +N2_mech + N3_mech + N4_mech

    mesh_1= np.linspace(0,sx1, N1_mech, endpoint=False)
    mesh_2= np.linspace(sx1,sx2, N2_mech, endpoint=False)
    mesh_3= np.linspace(sx2,sx3, N3_mech, endpoint=False)
    mesh_4= np.linspace(sx3,sx4, N4_mech)
    mesh  = np.concatenate((mesh_1, mesh_2, mesh_3, mesh_4))
    mesh0 = mesh

    #### INITIAL PRE-CYCLE STEP
    #### SOLVE MECHANICAL PART
    #### OUTPUT: MESH< RADIAL DISP, [SG_R, SG_THETA, SG_Z, TRACE]
    mesh_total[0,:], ur_total[0,:], sg = Solver_Mec(coordinates, mesh_number, lamb, mu, eig, 0.0, p_, iter)

    #### UPDATE DIFFUSION COEFFICIENTS
    D_12 = D12_update(sg[3][N1_mech: N1_mech+N2_mech].mean(), k_coup)
    D_23 = D23_update(sg[3][N1_mech+N2_mech: N1_mech+N2_mech+N3_mech].mean(), k_coup)

    #print(sg[3][N1_mech: N1_mech+N2_mech].mean())
    #print(sg[3][N1_mech+N2_mech: N1_mech+N2_mech+N3_mech].mean())

    for t_i in range(t_num):
        x1_new[t_i+1] =  x1_new[t_i] + dt/(c1p-c1m)*(-D_12*(c2m-c1p))/(x2-x1)
        x2_new[t_i+1] =  x2_new[t_i] + dt/(c2p-c2m)*(D_12*(c2m-c1p)/(x2-x1) - D_23*(c3m-c2p)/(x3-x2) )
        x3_new[t_i+1] =  x3_new[t_i] + dt/(c3p-c3m)*( D_23*(c3m-c2p)/(x3-x2) )

        #check
        if (x1_new[t_i+1] - x0)<0:
            print("ERROR   x1 < x0", t_i+1)
            break
        if (x2_new[t_i+1] - x1_new[t_i+1])<0:
            print("ERROR   x2 < x1", t_i+1)
            break
        if (x3_new[t_i+1] - x2_new[t_i+1])<0:
            print("ERROR   x3 < x2", t_i+1)
            break
        if (x4 - x3_new[t_i+1])<0:
            print("ERROR   x4 < x3", t_i+1)
            break
            
        x0 = sx0
        x1 = x1_new[t_i+1]
        x2 = x2_new[t_i+1]
        x3 = x3_new[t_i+1]
        x4 = sx4
        #mesh_total[t_i+1,:], ur_total[t_i+1,:], sg = Solver_Mec([x0,x1,x2,x3,x4], mesh_number, lamb, mu, eig, 0.0, p_, iter)    
        
        #update diffusion coffs
        D_12 = D12_update(sg[3][N1_mech: N1_mech+N2_mech].mean(), k_coup)
        D_23 = D23_update(sg[3][N1_mech+N2_mech: N1_mech+N2_mech+N3_mech].mean(), k_coup)

    return mesh_total, ur_total, x1_new, x2_new, x3_new, sg
