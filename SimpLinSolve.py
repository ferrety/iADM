# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2016  Vesa Ojalehto
#
# This work was supported by the Academy of Finland (grant number 287496)

"""
Module description
"""
import numpy as np
from pyProblem import Problem,Variable

class SimpLinSolve(Problem):
    def __init__(self, nf=3,nx=3,lb=[1,1],ub=[3,3]):
        '''
        '''
        self.nf=nf
        self.nx=nx
        self.variables=[]
        self.ideal=[ 132.7119,540.6497,20.0]
        self.nadir=[4175.257,7299.239,780.3944]        
        for xi in range(nx):
            self.variables.append(Variable(lb[xi],ub[xi]))
    
    def evaluate(self,solution):
        z=self,solution.objectives
        k = len(z) # nr. of objectives
        a = [0.2*0.9^i for i in range(k)] # coefficients
        t = min( # point in the reference direction
            (1-z[i]-sum(a[j]*z[j] if not(i==j) else 0 for j in range(k)))/
            (1+sum(a[j] if not(i==j) else 0 for j in range(k)))
                for i in range(k)
            )
        solution.objectives=np.add(z,t)
