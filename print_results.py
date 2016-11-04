# coding: utf-8
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution
import pandas
from scipy import spatial,stats
import numpy as np
import pyMOEA

from collections import defaultdict
import itertools                               
import operator

def prj_ref(results,PO=None,evals=20000):
    proj_refs=[]
    if PO is None:
        PO={}
    for nf,method,problem in sorted(results.keys(),key=operator.itemgetter(0,1,2)):
        #if nf==4 or problem=="DTLZ4":
        #    continue
        dist=[]
        for res in results[(nf,method,problem)]:
            obj=res[0]
            ref=tuple(res[1][0])
            if not PO.has_key((nf,problem,ref)):
                print (nf,problem,ref)
                proj_refs.append((nf,problem,ref))
    res=Parallel(n_jobs=8)(
           delayed(pyMOEA.proj_ref)(job[0],job[1],job[2],evals=evals) 
                for job in proj_refs        
        )
    for r in res:
        PO[tuple(r[:3])]=r[3]
    return PO

def print_results(results,POs=None,evals=20000):
    missing=[]
    error=[]
    if POs is None:
        POs={}
    full_dist=[]
    pp=defaultdict(list)
    for nf,method,problem in sorted(results.keys(),key=operator.itemgetter(1,0,2)):
        pp[method].append([nf,problem])
        nf = int(nf)
        dist=[]
        for res in results[(nf,method,problem)]:
            obj=res[0]
            ref=tuple(res[1][0])
            if POs.has_key((nf,problem,ref)):
                PO=POs[nf,problem,ref]
            else:
                missing.append((nf,method,problem,ref))
                #print "Projection not available"
                PO = pyMOEA.ACH_solution(problem,ref,evals=evals)
                POs[nf,problem,ref]=PO
                continue
            dist.append(spatial.distance.euclidean(obj,PO))
        mean, sigma = np.mean(dist), np.std(dist)
        try:
            pp[method][-1].extend([np.mean(dist),np.std(dist),np.min(dist),stats.norm.interval(0.95, loc=mean, scale=sigma / np.sqrt(len(dist)))])
            pp[method][-1].append(len(dist))
        except ValueError:
            error.append((nf,method,problem))
        
    for method in pp.keys():
        print method
        out=[]
        for values in pp[method]:
            try: 
                values[4]
            except IndexError:
                continue
            out.append([values[1],values[0],values[2],values[3],values[4],values[6],np.abs((values[5][1]-values[5][0])/2)])
        try:
            print pandas.DataFrame(out,columns=["problem","k","mean","deviation","min","n","2-sigma"]).to_latex(float_format=lambda x:"%11.4f"%x)
        except Exception,e:
            print e
            print out
        
    print "Projections not available: %i"%len(missing)
    print "Misconfigured values: %i"%len(error)
       
    return missing,error

def print_ADM2(results):
    """
    """
    for res in results:
        for ref in res[3][0]:
            print ref
    #objs[-1],rectangles,(method,problem),(objs,refs)
    #(nf,r[2][0].__name__,r[2][1])].append((r[0],r[1],r[3])


