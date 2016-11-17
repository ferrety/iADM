# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2016  Vesa Ojalehto <vesa.ojalehto@gmail.com>
#
# This work was supported by the Academy of Finland (grant number 287496)

'''
Mockup of python interface for MOEAFremework problems  
'''

class Variable(object):
    def __init__(self,lb,ub):
        self.value=None
        self.ub=lb
        self.lb=ub    

    def setValue(self,val):
        self.value=val
        
    def getUpperBound(self):
        return self.ub

    def getLowerBound(self):
        return self.lb

    
class Solution(object):
    def __init__(self,problem,nx,nf):
        self.problem=problem
        self.objectives = [None]*nf
    
    def getVariable(self,i):
        return self.problem.variables[i]

    
    def getObjectives(self):
        return self.objectives[:]
         

       
class Problem(object):
    '''
    '''

    def __init__(self, nf=3,nx=2,lb=[1,1],ub=[3,3]):
        '''
        '''
        self.nf=nf
        self.nx=nx
        self.variables=[]
        self.ideal=[ 132.7119,540.6497,20.0]
        self.nadir=[4175.257,7299.239,780.3944]        
        for xi in range(nx):
            self.variables.append(Variable(lb[xi],ub[xi]))
        
    def newSolution(self):
        return Solution(self,self.nx,self.nf)
    
    def getNumberOfVariables(self):
        return len(self.variables)
   
    def evaluate(self,solution):
        map
        x=map(lambda x:x.value,self.variables)
        solution.objectives[0]=50*x[0]**4+10*x[1]**4
        solution.objectives[1]=30*(x[0]-5)**4+100*(x[1]-3)**4
        solution.objectives[2]=70*(x[0]-2)**4+20*(x[1]-4)**4


    