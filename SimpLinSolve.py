# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2016  Vesa Ojalehto
#
# This work was supported by the Academy of Finland (grant number 287496)
import math

"""
Module description
"""
import numpy as np
from pyProblem import Problem,Variable

class SimpLinSolve(Problem):
    def __init__(self, nf=3):
        '''
        '''
        self.nf=nf
        self.nx=nf
        self.variables=[]
        self.ideal=[0.0]*self.nf
        self.nadir=[1.0]*self.nf   
        for xi in range(self.nf):
            self.variables.append(Variable(0.0,1.0))        
    
    def evaluate(self,solution,refpoint):
        z=refpoint
        k = len(z) # nr. of objectives
        a = [0.2*math.pow(0.9,i) for i in range(k)] # coefficients
        t = min( # point in the reference direction
            (1-z[i]-sum(a[j]*z[j] if not(i==j) else 0 for j in range(k)))/
            (1+sum(a[j] if not(i==j) else 0 for j in range(k)))
                for i in range(k)
            )
        return list(np.add(z,t))
