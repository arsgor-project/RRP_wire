import PyGnuplot as gp
import numpy as np
X = np.arange(10)
Y = np.sin(X/(2*np.pi))
Z = Y**2.0
gp.s([X,Y,Z])
gp.c('plot "tmp.dat" u 1:2 w lp')
gp.c('replot "tmp.dat" u 1:3 w lp')
gp.p('myfigure.ps')