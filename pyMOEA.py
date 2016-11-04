import os
import sys
import itertools
import tempfile
import shutil
from sklearn import tree
from scipy.spatial import Rectangle
import copy
try:
    JAVA_HOME=os.environ['JAVA_HOME']
except KeyError:
    JAVA_HOME=r'C:\Program Files\Java\jdk1.8.0_31'
    os.environ['JAVA_HOME']=JAVA_HOME

os.environ['PATH']+=r';%s\bin;%s\jre\bin\server'%tuple([JAVA_HOME]*2)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random

from scipy import spatial
from scipy.optimize import differential_evolution
import numpy as np

try:
    import jnius_config
    moea_path=r"C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3"

    for d in ['bin',r'lib\*',r'build\MOEAFramework-2.3\lib']:
        jnius_config.add_classpath(os.path.join(moea_path,d))

    moea_clspath=[
                 r"C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\bin;",
                  "C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\commons-cli-1.2.jar;",
                  "C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\commons-codec-1.8.jar;",
                  "C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\commons-lang3-3.1.jar;",
                  "C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\commons-math3-3.1.1.jar;",
                  "C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\jcommon-1.0.20.jar;",
                  "C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\jfreechart-1.0.15.jar;",
                  "C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\JMetal-4.3.jar;",
                  "C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\rsyntaxtextarea.jar"]
    jnius_config.add_classpath(*moea_clspath)

except ValueError,e:
    pass
    #print(e)
    #print("Could not initialize jinus as it has allready initialized")
    #print(jnius_config.get_classpath())

from jnius import autoclass,cast, JavaException


def ADM2_reference(rectangles):
    """ Generate new reference point for ADM"

    Selects the point with the maximum value of the utility function as the reference point

    """
    r=1
    a=.5
    cec_utility=np.vectorize(lambda x:np.power(np.sum(np.power(x.mins,r)),1./r))
    return rectangles[np.argmax(cec_utility(rectangles))].mins


def ADM2_solve(method,problem, rectangles,evals=2000):
    print "AMD2 Solving: %s:%s"%(str((method.__name__,problem)),str(len(rectangles)))
    max_iter=0
    objs=[]
    refs=[ADM2_reference(rectangles[-1])]
    nf = len(rectangles[-1][-1].mins)
    while max_iter<10:
        new_rectangles=copy.deepcopy(rectangles[-1])
        objs.append(method(problem,refs[-1],evals=evals))
        
        #print "ref point {} -> {}".format(refs[-1],objs[-1])
        srnd=lambda v: list(map(lambda x:round(x,6),v))
        ref_str="%s"%srnd(refs[-1])
        print format("ref point %s -> %s"%(ref_str.ljust(10*nf),srnd(objs[-1])))
        #print format("ref point %20s -> %10s"%tuple(map(srnd,[refs[-1],list(objs[-1])])))
        replace_rec=None
        for rec in new_rectangles:
            if rec.min_distance_point(objs[-1])==0.0:
                replace_rec=rec
                break

        if replace_rec is None:
            print("New PO point %s out of bounds\n     For ref %s"%(objs[-1],refs[-1]))
            break
        new_rectangles.remove(replace_rec)
        rectangles.append(new_rectangles)
        
        for i,obj in enumerate(objs[-1][:]):
            low=list(replace_rec.mins)
            low[i]=obj
            up=objs[-1][:]
            up[i]=list(replace_rec.maxes)[i]
            rectangles[-1].append(Rectangle(low,up))
            #rectangles[-1].append(Rectangle((replace_rec.maxes[0],replace_rec.mins[1]),objs[-1]))

            
        ref=ADM2_reference(rectangles[-1])
        if np.array_equal(ref,refs[-1]):
            print "Could not generate new refpoint %s"%objs[-1]
            break
        refs.append(ref)
        max_iter +=1
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
    
    pyMOEARunnner=autoclass('pyMOEARunnner')
    rn=pyMOEARunnner()
    res=rn.solve(method,evals,*problem_def)
    os.chdir(olddir)
    shutil.rmtree(tmp_dir)
    points=[]
    siter=res.iterator()
    while siter.hasNext():
        sol=siter.next()
        points.append(sol.getObjectives())
    return points

def evaluate(problem,point):
    sol=problem.newSolution()
    for i,val in enumerate(point):
        cast('org.moeaframework.core.variable.RealVariable',sol.getVariable(i)).setValue(val)
    problem.evaluate(sol)
    return sol.getObjectives()

def plot(points):
    p=np.array(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:,0],p[:,1],p[:,2])
    plt.show(block=False)


def problem(problem_name,nf,nx=None):
    try:
        prob_def=problem_def(problem_name,nf,nx=nx)
    except:
        raise Exception("Could not generate problem %s"%problem_name)
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
    for key in NX.keys():
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
        var = cast('org.moeaframework.core.variable.RealVariable', sol.getVariable(i))
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
        trees[-1].fit(A[:,range(0,i)+range(i+1,nf)], A[:,i])
    
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
               


def ACH_solution(problem_name,refpoint,evals=10000,nx=None):
    nf=len(refpoint)
    prob=problem(problem_name,nf,nx=nx)
    bds=bounds(prob)
    res=differential_evolution(ACH,bds,args=(prob,refpoint),maxiter=evals)
    return evaluate(prob,res.x)

def rNSGAII_solution(problem,refpoint,evals=10000):
    nf=len(refpoint)
    points=solve(problem_def(problem,nf),refpoint=refpoint,evals=evals)
    A=np.array(points)
    return list(A[spatial.KDTree(A).query(refpoint)[1]])

def proj_ref(nf,problem,ref,evals=20000,nx=None):
    print "Projecting %s_%i for %s"%(problem,nf,str(ref))
    res=ACH_solution(problem,ref,evals=evals,nx=nx)
    return (nf,problem,ref,res)


def agent_paraller_orig(target,preference,adm=1,evals=20000):
    method=target[0]
    problem=target[1]

    return agent_solve(method,problem,preference,evals)


def agent_paraller(target,preference,adm=1,evals=20000):
    method=target[0]
    problem=target[1]

    return ADM2_solve(method,problem,preference,evals)

def agent_solve(method,problem,preference,adm=1,evals=20000):
    print "Solving: %s:%s"%(str((method.__name__,problem)),str(len(preference[0])))
    w=tuple(preference[1])

    i=0
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
    print "Solved: %s:\n%s"%(str((method.__name__,problem)),str(objs[-1]))
    return objs[-1],preference,(method,problem),(objs,refs)

if __name__=='__main__':
    import os
    import sys

    import argparse

    parser= argparse.ArgumentParser(description='Solve problem with RNSGAII')
    parser.add_argument('-p','--problem', type=str, nargs='+',
                        help='Problem to be solved',required=True)

    parser.add_argument('-m','--method', type=str, nargs='+',
                   help='Problem to be solved', default=["RNSGAII"])

    parser.add_argument('-r', '--refpoint', type=float, nargs='+',
                        help='reference point',default=None)

    parser.add_argument('-e', '--epsilon', type=float,
                        help='epsilon value',default=0.0001)

    parser.add_argument('-f', '--evals', type=int,
                        help='Function evaluations',default=10000)

    parser.add_argument('--plot', help='Plot the approximate set',action='store_true')

    args=parser.parse_args()

    for method in args.method:
        for problem in args.problem:
            points = solve(problem,method,args.refpoint,args.epsilon,args.evals)
            if args.plot:
                plot(points)
    if args.plot:
        answer = raw_input('Press enter to exit')

