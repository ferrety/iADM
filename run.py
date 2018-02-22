# coding: utf-8
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2016  Vesa Ojalehto
#
# This work was supported by the Academy of Finland (grant number 287496)

import pyMOEA
from random import random
from joblib import Parallel, delayed
from scipy import spatial
from collections import defaultdict
import itertools
from scipy.spatial import Rectangle


def make_run(problem,ref,nf=3,runs=10,evals=10000):
    nsgaii=[]
    ach=[]
    dist=[]
    best=[0,0]
    for f in range(runs):
        nsgaii.append(pyMOEA.rNSGAII_solution(problem,ref,nf=nf,evals=evals))
        nd=spatial.distance.euclidean(ref,nsgaii[-1])

        ach.append(pyMOEA.ACH_solution(problem,ref,nf=nf))
        ad=spatial.distance.euclidean(ref,ach[-1])
        dist.append((ad,nd))
        if ad<nd:
            best[0]+=1

        else:
            best[1]+=1
    return (best,ach,nsgaii,dist)


def random_pref(nf=3):
    aspir = [0.0] * nf
    w = [None] * nf
    for k in range(nf):
        w[k] = random()
        if random() < .5:
            aspir[k] = random()e
    return aspir, w

def ADM2_run(nf=3,jobs=8,evals=20000,runs=4,single_pref=True,
              methods=[pyMOEA.ACH_solution,pyMOEA.rNSGAII_solution],
              problems=['DTLZ1','DTLZ2','DTLZ3','DTLZ4']):
    """
    Returns dict with keys (nf,method_name,problem_name)

    and values of final_solution,(aspir,weigth),(objectives,references)
    """

    ideal=[0.0]*nf
    nadir=[1.0]*nf
    bounds=(ideal,nadir)

    init_pref=[[Rectangle(ideal,nadir)]]
    results=defaultdict(list)
    for r in range(runs):
        res=Parallel(n_jobs=jobs)(
           delayed(pyMOEA.agent_paraller)(target,init_pref,evals=evals)
                for target in itertools.product(methods,problems)
        )
        for r in res:
            results[(nf,r[2][0].__name__,r[2][1])].append((r[0],r[1],r[3]))

    return results



def agent_run(nf=3,jobs=8,evals=20000,runs=10,single_pref=True,
              methods=[pyMOEA.ACH_solution,pyMOEA.rNSGAII_solution],
              problems=['DTLZ1','DTLZ2','DTLZ3','DTLZ4']):
    """
    Returns dict with keys (nf,method_name,problem_name)

    and values of final_solution,(aspir,weigth),(objectives,references)
    """

    ideal=[0.0]*nf
    nadir=[1.0]*nf
    bounds=(ideal,nadir)
    pref_list=[]

    if single_pref:
        pref_list=[random_pref(nf)]*runs
    else:
        pref_list=[random_pref(nf) for i in range(runs)]
    results=defaultdict(list)
    for r in range(runs):
        res=Parallel(n_jobs=jobs)(
           delayed(pyMOEA.agent_paraller_orig)(target, pref_list[r], evals = evals)
                for target in itertools.product(methods,problems)
        )
        for r in res:
            results[(nf,r[2][0].__name__,r[2][1])].append((r[0],r[1],r[3]))
            #objs[-1],preference,(objs,refs)

    return results


if __name__=='__main__':
    import os
    import sys

    import argparse

    parser= argparse.ArgumentParser(description='Solve problem with RNSGAII')
    parser.add_argument('-p','--problem', type=str, nargs='+',
                        help='Problem to be solved',required=False)

    parser.add_argument('-r', '--refpoint', type=float, nargs='+',
                        help='reference point',required=False)

    parser.add_argument('-e', '--epsilon', type=float,
                        help='epsilon value',default=0.0001)

    args=parser.parse_args()
    results=defaultdict(list)

    # import datetime,pickle
    # nf=2
    # res=ADM2_run(nf=nf,runs=1,jobs=1,single_pref=True, methods=[pyMOEA.ACH_solution],
    #          problems=['DTLZ2','DTLZ1'])
    
    # ADM2
    # import adm2_run
    # adm2_run.plot_res(res[0])
    
    # hADM
    res_all = agent_run(nf = 2, runs = 1, single_pref = True, problems = ['DTLZ1'])
    # pickle.dump(results, open("results-nf%i-%s.dmp" % (nf, datetime.datetime.now().date().isoformat()), "w"))
