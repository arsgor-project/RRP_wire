import sympy as sp
import numpy as np
from sympy.solvers.solveset import linsolve

### returns analytical expression for only pressure problem
def ExactPressure(p_, lamb, mu, sx):
    
    lamb1, lamb2, lamb3, lamb4 = lamb
    mu1,mu2,mu3,mu4 = mu
    sx0,sx1,sx2,sx3,sx4 = sx

    x = sp.symbols('x')
    
    #b1=0
    ##only compression, epsz =0
    a1, a2, a3, a4, b2, b3, b4 = sp.symbols('a_1, a_2, a_3, a_4, b_2, b_3, b_4')
    p=p_ 

    cof= linsolve([(a1-a2)*sx1 -b2/sx1, (a2-a3)*sx2 -b3/sx2 + b2/sx2, (a3-a4)*sx3 -b4/sx3 + b3/sx3,\
        2*(lamb1+mu1)*(a1) -  2*(lamb2+mu2)*(a2) +  2*mu2/sx1**2*b2,\
            2*(lamb2+mu2)*(a2) -  2*(lamb3+mu3)*(a3) -2*mu2/sx2**2*b2 +  2*mu3/sx2**2*b3, \
                2*(lamb3+mu3)*(a3) -  2*(lamb4+mu4)*(a4) -2*mu3/sx3**2*b3 +  2*mu4/sx3**2*b4,\
                    2*(lamb4+mu4)*(a4) -2*mu4/sx4**2*b4  -p ], (a1,a2,a3,a4,b2,b3,b4))
    a1_,a2_,a3_,a4_,b2_,b3_,b4_ = tuple(*cof)
    ans = sp.Piecewise((a1_*x, x<=sx1), (a2_*x + b2_/x, (x<=sx2)&(x>sx1)), (a3_*x+b3_/x, (x<=sx3)&(x>sx2)), (a4_*x+b4_/x, (x<=sx4)&(x>sx3)    ), (0, True))
    msh = np.linspace(sx0,sx4,400)
    
    f_ans = sp.lambdify(x, ans)

    #print(tuple(*cof))
    
    return msh, f_ans(msh)

 ### returns analytical expression for only eigenstrains problem
def ExactEigenstrain(eig, lamb, mu, sx):

    lamb1, lamb2, lamb3, lamb4 = lamb
    mu1,mu2,mu3,mu4 = mu
    sx0,sx1,sx2,sx3,sx4 = sx
    eig_12,eig_23 = eig
    
    x = sp.symbols('x')
    
    #b1=0
    ##, epsz =0
    a1, a2, a3, a4, b2, b3, b4 = sp.symbols('a_1, a_2, a_3, a_4, b_2, b_3, b_4')

    cof= linsolve([(a1-a2)*sx1 -b2/sx1, (a2-a3)*sx2 -b3/sx2 + b2/sx2, (a3-a4)*sx3 -b4/sx3 + b3/sx3,\
        2*(lamb1+mu1)*(a1) -  2*(lamb2+mu2)*(a2) +  2*mu2/sx1**2*b2 + (3*lamb2+2*mu2)*eig_12,\
            2*(lamb2+mu2)*(a2) -  2*(lamb3+mu3)*(a3) -2*mu2/sx2**2*b2 +  2*mu3/sx2**2*b3 -(3*lamb2+2*mu2)*eig_12 + (3*lamb3+2*mu3)*eig_23, \
                2*(lamb3+mu3)*(a3) -  2*(lamb4+mu4)*(a4) -2*mu3/sx3**2*b3 +  2*mu4/sx3**2*b4 - (3*lamb3+2*mu3)*eig_23,\
                    2*(lamb4+mu4)*(a4) -2*mu4/sx4**2*b4], (a1,a2,a3,a4,b2,b3,b4))
    a1_,a2_,a3_,a4_,b2_,b3_,b4_ = tuple(*cof)
    ans = sp.Piecewise((a1_*x, x<=sx1), (a2_*x + b2_/x, (x<=sx2)&(x>sx1)), (a3_*x+b3_/x, (x<=sx3)&(x>sx2)), (a4_*x+b4_/x, (x<=sx4)&(x>sx3)    ), (0, True))
    msh = np.linspace(sx0,sx4,400)
    
    f_ans = sp.lambdify(x, ans)
    
    return msh, f_ans(msh)
