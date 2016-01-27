import os
import sys

JAVA_HOME=r'C:\Program Files\Java\jdk1.8.0_25'

os.environ['JAVA_HOME']=JAVA_HOME
os.environ['PATH']+=r';%s\bin;%s\jre\bin\server'%tuple([JAVA_HOME]*2)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


try:
    import jnius_config
    moea_path=r"C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3"

    for d in ['bin',r'lib\*',r'build\MOEAFramework-2.3\lib']:
        jnius_config.add_classpath(os.path.join(moea_path,d))

    moea_clspath=r"C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\bin;C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\commons-cli-1.2.jar;C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\commons-codec-1.8.jar;C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\commons-lang3-3.1.jar;C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\commons-math3-3.1.1.jar;C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\jcommon-1.0.20.jar;C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\jfreechart-1.0.15.jar;C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\JMetal-4.3.jar;C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3\lib\rsyntaxtextarea.jar".split(";")
    jnius_config.add_classpath(*moea_clspath)

except ValueError,e:
    pass
    #print(e)
    #print("Could not initialize jinus as it has allready initialized")
    #print(jnius_config.get_classpath())

from jnius import autoclass,cast, JavaException

def solve(problem_def,method="RNSGAII",refpoint=None,epsilon=0.00001,evals=10000):
    MOEA_dir=r"C:\MyTemp\nimbus\hnsga2-moea\MOEAFramework-2.3"

    if refpoint is not None:
        with open("refpoints.txt","w") as fd:
            fd.write(str(epsilon)+'\n')
            fd.write(" ".join(map(str,refpoint)))
            fd.close()
    
    pyMOEARunnner=autoclass('pyMOEARunnner')
    rn=pyMOEARunnner()
    res=rn.solve(method,evals,*problem_def)
    
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


def problem(problem_name,nf):
    try:
        prob_def=problem_def(problem_name,nf)
    except:
        raise Exception("Could not generate probelm %s"%problem_name)
    problemClass=autoclass(prob_def[0])
    return problemClass(prob_def[1],prob_def[2]) 

def problem_def(problem_name,nf,nx=None):

    NX={'DTLZ':[4]+[9]*3+[None]*2+[19]}

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

