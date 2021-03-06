# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2016  Vesa Ojalehto
#
# This work was supported by the Academy of Finland (grant number 287496)

import os
import sys
import itertools
import logging

import tempfile
import shutil

import pyProblem
import SimpLinSolve

from sklearn import tree
from scipy.spatial import Rectangle
import copy
import math
import multiprocessing, operator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random
from joblib import Parallel, delayed
from scipy import spatial
from scipy.optimize import differential_evolution

logger = logging.getLogger('pyMOEA')
logger.addHandler(logging.NullHandler())

try:
    JAVA_HOME = os.environ['JAVA_HOME']
except KeyError:

    JAVA_HOME = r'C:\Program Files (x86)\Java\jre6' # r'C:\Program Files\Java\jdk1.8.0_31'
    os.environ['JAVA_HOME'] = JAVA_HOME
os.environ['PATH'] += r';%s\bin;%s\jre\bin\server' % tuple([JAVA_HOME] * 2)

try:
    import jnius_config
except ValueError as e:
    logging.error("Failed to import jinus (Has it been previously initialized?)")
    logging.debug(jnius_config.get_classpath())

# Classpath must be configured
moea_path = r'D:\JYU\MOEAFramework-2.12'

moea_clspath = os.listdir(os.path.join(moea_path, 'lib'))
#  moea_clspath = [
#               'bin',
#               'lib',
#               'build\MOEAFramework-2.3\lib',
#               'lib\commons-cli-1.2.jar',
#               'lib\commons-codec-1.8.jar',
#               'lib\commons-lang3-3.1.jar',
#               'lib\commons-math3-3.1.1.jar',
#               'lib\jcommon-1.0.20.jar',
#               'lib\jfreechart-1.0.15.jar',
#               'lib\JMetal-4.3.jar',
#               'lib\rsyntaxtextarea.jar']

for d in moea_clspath:
    jnius_config.add_classpath(os.path.join(moea_path, 'lib', d))

from jnius import autoclass, cast, JavaException

        # jnius_config.add_classpath(*moea_clspath)



    
def utility_function(uf_n,k_obj):
    """  Return utility function uf_n for k_obj objectives

    :param int uf_n: Utility function to be created
    :param int k_obj: Number of objectives
    :rtype np.vectorize: Utility function
    """
    ########################################################
    # DP
    ########################################################
    
    ### two types of weighting: equal weights and ROC weighting scheme
    w_for_uf=(
        # equal weights       
        [1. for i in range(k_obj)],
        # ROC weights
        [sum(1/(j+1) for j in range(i,k_obj)) for i in range(k_obj)]
        )
    
    # utility function classes - parametric, for defining 
    # different function instances with different parameters
    # in the case of maximization; nadir=0, ideal=1; x = objective vector
    r=0.5 # constant parameter
    
    uf_param=(
    ### CES utility function
    lambda w:(
        lambda x:
        np.sum(np.multiply(np.power(x,r),w))
        ),
    
    ### Cobb-Douglas utility function    
    lambda w:(
        lambda x:
        np.prod(np.power(x,w))
        ),
    
    ### TOPSIS utility function
    lambda w:(
        lambda x:
            np.sqrt(np.sum(np.multiply(np.power(x,2),w)))/
            (np.sqrt(np.sum(np.multiply(np.power(x,2),w))) +
             np.sqrt(np.sum(np.multiply(np.power(
                      np.add(1,np.negative(x))                                 
                                              ,2),w))))
    
        )
            )
        
    ####################################################
    # dummy nadir and ideal representing not local, but global ones,
    # related to the whole problem - redefine to actual nadir and ideal
    ####################################################
    
    ## the list of all used utilities appropriate to the problem setting 
    
    uf_list=[
             uf(w)
                 for uf in uf_param
                     for w in w_for_uf
             ]

    # TODO get the actual ideal and nadir values
    nadir_global=[1.]*k_obj
    ideal_global=[0.]*k_obj
    return np.vectorize(lambda x:uf_list[uf_n]( # transforming the objective vector
        np.add(nadir_global,np.negative(x.mins))/
    np.add(nadir_global,np.negative(ideal_global))
                                 )
                         )
    
def ADM2_reference(rectangles,uf_n=0):
    """ Generate new reference point for ADM

    :param list[Rectangles] rectangles: List of existing rectangels
    :param int uf_n: Utility function to be used, see ``uf_n`` in  :func:`utility_function`

    Selects the point with the maximum value of the utility function as the reference point
    uf_n = index of the uility function selected from uf_list 
    """
    k_obj=len(rectangles[0].mins)
    utility=utility_function(uf_n,k_obj)

    return rectangles[np.argmax(utility(rectangles))].mins

def ADM2_solve(method,problem, rectangles,evals=2000,verbose=1,max_iter=10,**kwargs):
    
    logger.info("AMD2 Solving: %s:%s"%(str((method.__name__,problem)),str(len(rectangles))))
    iter=0
    objs=[]
    refs=[ADM2_reference(rectangles[-1],**kwargs)]
    nf = len(rectangles[-1][-1].mins)
    while iter<max_iter:
        new_rectangles=copy.deepcopy(rectangles[-1])
        
        logger.debug("Running method %s"%method.__name__)     
        objs.append(method(problem,refs[-1],evals=evals))
        try:
            if np.linalg.norm(refs[-1]-refs[-2]) < 0.000001:
                break
        except IndexError:
            pass
        
        srnd=lambda v: list([round(x,6) for x in v])
        ref_str="%s"%srnd(refs[-1])
        
        logger.debug(format("ref point %s -> %s"%(ref_str.ljust(10*nf),srnd(objs[-1]))))
        replace_rec=None
        distances=[]
        outside=True
        for rec in new_rectangles:
            distances.append(rec.min_distance_point(objs[-1]))
            if distances[-1]<=0.0001:
                replace_rec=rec
                outside=False
        
        if outside:
            replace_rec=new_rectangles[np.argmin(distances)]
            
        new_rectangles.remove(replace_rec)
        rectangles.append(new_rectangles)
            
        for i,obj in enumerate(objs[-1][:]):
            low=list(replace_rec.mins)
            if outside:
                if obj>replace_rec.mins[i]:
                    low[i]=replace_rec.mins[i]    
            else:
                low[i]=obj
            up=objs[-1][:]
            up[i]=list(replace_rec.maxes)[i]
            rectangles[-1].append(Rectangle(low,up))      
      
        ref=ADM2_reference(rectangles[-1])
        if np.array_equal(ref,refs[-1]):
            logger.warning("Could not generate new refpoint from PO point %s"%objs[-1])
            break
        refs.append(ref)
        iter +=1
    logger.info("AMD2  %s:%s Done"%(str((method.__name__,problem)),str(len(rectangles))))
    return objs[-1],rectangles,(method.__name__,problem),(objs,refs)


def solve(problem_def,method="RNSGAII",refpoint=None,epsilon=0.00001,evals=10000):
    tmp_dir=tempfile.mkdtemp()
    olddir=os.getcwd()
    os.chdir(tmp_dir)
    
    if refpoint is not None:
        with open(os.path.join("refpoints.txt"),"w") as fd:
            fd.write(str(epsilon)+'\n')
            fd.write(" ".join(map(str,refpoint)))
            fd.close()
    
    pyMOEARunnner = autoclass('pyMOEARunnner')
    rn=pyMOEARunnner()
    res=rn.solve(method,evals,*problem_def)
    os.chdir(olddir)
    shutil.rmtree(tmp_dir)
    points=[]
    siter=res.iterator()
    while siter.hasNext():
        sol=next(siter)
        points.append(sol.getObjectives())
    return points

def evaluate(problem,point):
    sol=problem.newSolution()
    for i,val in enumerate(point):
        try:
            cast('org.moeaframework.core.variable.RealVariable',sol.getVariable(i)).setValue(val)
        except TypeError:
            sol.getVariable(i).setValue(val)
    problem.evaluate(sol)
    return sol.getObjectives()

def plot(points):
    p=np.array(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:,0],p[:,1],p[:,2])
    plt.show(block=False)


def problem(problem_name,nf,nx=None):
    if "External" in problem_name:
        return pyProblem.Problem(nf,nx=nx)
    elif "SimpLinSolve" in problem_name:
        return SimpLinSolve.SimpLinSolve(nf)
    
    try:
        prob_def=problem_def(problem_name,nf,nx=nx)
    except IOError as e:
        print problem_name
        import traceback
        traceback.print_exc()

        raise Exception("Could not generate problem %s\n%s" % (problem_name, e))
    problemClass=autoclass(prob_def[0])
    # Problem families where number of objectives can be changed
    NF=['DTLZ']
    for family in NF:
        if problem_name.startswith(family):
            return problemClass(prob_def[1],prob_def[2]) 
    return problemClass(nx) 

def problem_def(problem_name,nf,nx=None):

    NX={'DTLZ':[4]+[9]*3+[None]*2+[19],
        'ZDT' :[30,30,30,30],
        }

    cdir = None
    for key in list(NX.keys()):
        if problem_name.startswith(key):
            if nx is None:
                idx=int(problem_name[len(key)])-1
                nx=nf+NX[key][idx]
            cdir=key
            break
            
    if cdir:
        ProblemClassName = 'org.moeaframework.problem.%s.%s'%(cdir,problem_name)
    else:
        return None

    return (ProblemClassName,nx,nf)

def bounds(problem):
    EPS=0.000001
    bounds=[]
    sol=problem.newSolution()
    for i in range(problem.getNumberOfVariables()):
#p2.__class__ == pyProblem.Problem
        try:
            var = cast('org.moeaframework.core.variable.RealVariable', sol.getVariable(i))
        except TypeError:
            var = sol.getVariable(i)
        bounds.append((var.getLowerBound()+EPS,var.getUpperBound()-EPS))
    return bounds

def iterate(preference,ref,objs,tol=0.01,p=.9):
    aspir=preference[0]
    w=preference[1]
    nf=len(aspir)
    nadir=[1.0]*nf
    new_ref=[0.0]*nf
    #create the decision tree
    A=np.array(objs)
    nf = len(ref)
    trees=[]
    for i in range(nf):
        trees.append(tree.DecisionTreeClassifier())
        trees[-1].fit(A[:,list(range(0,i))+list(range(i+1,nf))], A[:,i])
    
    obj=objs[-1]
    S=[None]*len(obj)
    pk=p
    for k in reversed(np.argsort(w)):
        fk=obj[k]
        if abs(aspir[k]-fk) < tol and random()<p:
            S[k]=fk
        elif random()<pk:
            S[k]=fk
        if len(S):
            pk-=pk/len(obj)*len(S)
    if len(S)==k:
        return ref
            
    for k,s in enumerate(S):
        if s is not None:
            new_ref[k]=min(aspir[k]-(aspir[k]-obj[k])/2.,nadir[k])
        else:
            new_ref[k]=None

    idx = [idx for idx,v in enumerate(new_ref) if v==None]
    for i in idx:
        n=[0.0 if v is None else v for v in new_ref]
        del n[i]
        new_ref[i]=min(trees[i].predict(n)[0],nadir[i])
    if np.array_equal(nadir,new_ref):
        return ref
    return new_ref

def ACH(x,problem,ref,rho=0.001):
    f=evaluate(problem,x)
    nadir=[1.0]*len(ref)
    utopian=[0.0]*len(ref)
    A=np.array([f,ref,nadir,utopian])    
    return np.max(np.apply_along_axis(lambda x:(x[0]-x[1])/(x[2]-x[3]),0,A)) \
                 +rho*np.sum(np.apply_along_axis(lambda x:x[0]/(x[2]-x[3]),0,A))


def ACH_solution(problem_name, refpoint, evals = 50000, nx = None):
    nf=len(refpoint)
    prob=problem(problem_name,nf,nx=nx)
    bds=bounds(prob)
    res = differential_evolution(ACH, bds, args = (prob, refpoint), maxiter = evals)
    return evaluate(prob,res.x)

def rNSGAII_solution(problem,refpoint,evals=10000):
    nf=len(refpoint)
    points=solve(problem_def(problem,nf),refpoint=refpoint,evals=evals)
    A=np.array(points)
    return list(A[spatial.KDTree(A).query(refpoint)[1]])

def Simple_solution(problem_name,refpoint,evals=None):
    """ Solve problem without using optimization method.
    """
    nf=len(refpoint)
    prob=problem(problem_name,nf,nx=nf)
    return prob.evaluate(None,refpoint)


def proj_ref(nf, problem, ref, evals = 20000, nx = None):
    print("Projecting %s_%i for %s" % (problem, nf, str(ref)))
    res = ACH_solution(problem, ref, evals = evals, nx = nx)
    with open(os.path.join("temp", "%s-%s_%s" % (problem, nf, ref)) + ".res", "w") as fd:
        fd.write(str(res))

    return (nf, problem, ref, res)


def proj_refs(results, PO = None, evals = 50000, nx = None, jobs = None):
    """ Update PO dictonary with reference points projected to Pareto frontier

    Formats

    results (dict)
    key tuple(nf,problem,problem)
    values [
     [Final solution]          # Iteration n
     [[aspir],[weights]]
     [[solution],[ref point]]  # Iteration 1
     ...
     [[solution],[ref point]]  # Iteration n
    ]


    PO (dict)



    """
    proj_refs = []
    if jobs == None:
        jobs = max(1, multiprocessing.cpu_count() - 1)
    if PO is None:
        PO = {}
    for nf, method, problem in sorted(results.keys(), key = operator.itemgetter(0, 1, 2)):
        # if nf==4 or problem=="DTLZ4":
        #    continue
        dist = []
        for res in results[(nf, method, problem)]:
            obj = res[0]
            ref = tuple(res[1][0])
            if not PO.has_key((nf, problem, ref)):
                proj_refs.append((nf, problem, ref))
    print ("Total number of jobs %s" % len(proj_refs))
    res = Parallel(n_jobs = jobs)(
           delayed(proj_ref)(job[0], job[1], job[2], evals = evals, nx = nx)
                for job in proj_refs
        )
    for r in res:
        PO[tuple(r[:3])] = r[3]
    return PO


def agent_paraller_orig(target,preference,adm=1,evals=20000):
    method=target[0]
    problem=target[1]

    return agent_solve(method,problem,preference,evals)


def agent_paraller(target,preference,adm=1,evals=20000):
    method=target[0]
    problem=target[1]

    return ADM2_solve(method,problem,preference,evals)

def agent_solve(method,problem,preference,adm=1,evals=20000):
    print "Solving: %s:%s" % (str((method.__name__, problem)), str(len(preference[0])))
    w=tuple(preference[1])

    i = 0
    objs=[]
    refs=[preference[0][:]]
    nf = len(refs[0])
    while refs[-1] is not None and i<10:
        objs.append(method(problem,refs[-1],evals=evals))
        new_ref=iterate(preference,refs[-1],objs)
        if np.array_equal(new_ref,refs[-1]):
            break
        refs.append(new_ref[:])
        i+=1
    print "Solved: %s:\n%s" % (str((method.__name__, problem)), str(objs[-1]))
    return objs[-1],preference,(method,problem),(objs,refs)

if __name__=='__main__':

    import argparse

    parser= argparse.ArgumentParser(description='Solve problem with RNSGAII')
    parser.add_argument('-p','--problem', type=str, nargs='+',
                        help='Problem to be solved',required=True)

    parser.add_argument('-m','--method', type=str, nargs='+',
                   help = 'Method to be used', default = ["RNSGAII"])

    parser.add_argument('-r', '--refpoint', type=float, nargs='+',
                        help='reference point',default=None)

    parser.add_argument('-e', '--epsilon', type=float,
                        help='epsilon value',default=0.0001)

    parser.add_argument('-f', '--evals', type=int,
                        help='Function evaluations',default=10000)

    parser.add_argument('--plot', help='Plot the approximate set',action='store_true')

    args=parser.parse_args()
    logger = logging.getLogger('pyMOEA')
    logger.addHandler(logging.StreamHandler)
    logger.setLevel(logging.DEBUG)
    logging.info("AAA")

    for method in args.method:
        for problem in args.problem:
            points = solve(problem,method,args.refpoint,args.epsilon,args.evals)
            if args.plot:
                plot(points)
    if args.plot:
        answer = input('Press enter to exit')

