""" Python codes for 'An interactive surrogate-assisted reference vector guided evolutionary multiobjective optimization algorithm'

.. seealso::

    Overleaf <https://www.overleaf.com/12692710tnqqbwcvgkpb>


"""

import os, pickle
import multiprocessing
from joblib import Parallel, delayed
import numpy as np

import ADM
import pyDECOMO
from scipy.optimize import differential_evolution

import argparse
from pandas.lib import has_infs_f4
import platform


class iKRVEA:

    def __init__(self, resdir="ik-rvea", ikrvea_path='..',
                 NFs=[4, 6, 8],
                 problems=['DTLZ1' , 'DTLZ2', 'DTLZ4', 'DTLZ5', 'DTLZ7'],
                 max_jobs = None,
                 runs=15,
                 ):
        self.resdir = resdir

        if not os.path.exists(self.resdir):
            os.makedirs(self.resdir)
        self.NFs = NFs
        self.problems = problems
        
        self.nx = 10

        # Just to make sure
        plat_jobs= platform_max_jobs()
        if max_jobs is None:
            self.max_jobs = plat_jobs
        else:
            self.max_jobs = min(max_jobs, plat_jobs)

        self.runs = runs
        self.ikrvea_path = ikrvea_path
    def _fn(self, filename):
        return os.path.join(self.resdir, filename)

    def create_preferences(self):
        try:
            self.preferences = pickle.load(open(self._fn("preferences.dmp"), "r"))
        except IOError:
            self.preferences = {}
        for nf in self.NFs:
            if self.preferences.has_key(nf):
                continue
            pref = []
            for i in range(self.runs):
                # TODO: Must be problem specific, between icv and nadir
                ta = ADM.hADM('DTLZ1', nf, nx=10)
                pref.append(ta.preference)
            self.preferences[nf] = tuple(pref)

        pickle.dump(self.preferences, open(self._fn("preferences.dmp"), "w"))

    def project_prefs(self, fkt_project):
        try:
            PO = pickle.load(open(self._fn("PO.dmb")), "r")
        except IOError:
            PO = {}
        self.create_preferences()
        jobs = []
        evals = 20000
        for nf in self.NFs:
            for pref in self.preferences[nf]:
                for p in self.problems:
                    if (nf, p, pref) not in PO:
                        jobs.append({
                                'self_arg':self,
                                'problem_name':p,
                                'nf':nf,
                                'ref':pref,
                                'nx':self.nx,
                                'evals':evals
                                })

        res = Parallel(n_jobs=self.max_jobs)(
               delayed(fkt_project)(**job)
                        for job in jobs
            )
        for r in res:
            PO[tuple(r[:3])] = r[3]

        pickle.dump(PO, open(self._fn("PO.dmb")), "w")

        return PO

    def ACH(self, ox, problem, ref, rho=0.001):
        f = problem.evaluate(ox)
        utopian, nadir = problem.bounds()
        A = np.array(map(np.array, [f, ref[1], utopian, nadir]))
        return np.max(np.apply_along_axis(lambda x:(x[0] - x[1]) / (x[2] - x[3]), 0, A)) \
                     + rho * np.sum(np.apply_along_axis(lambda x:x[0] / (x[2] - x[3]), 0, A))

    def ACH_solution(self, problem_name, nf, refpoint, evals=50000, nx=None):
        problem = self.problem(problem_name, nf)
        
        bds = np.rot90(np.array(([[0.0] * self.nx, [1.0] * self.nx])))

        res = differential_evolution(self.ACH, bds, args=(problem, refpoint), maxiter=evals)
        return problem.evaluate(res.x)

    def problem(self, problem_name, nf):
        """ Construct a new problem instance"""
        return pyDECOMO.pyDECOMO(problem_name, nf, self.ikrvea_path)


# Unwrag self from the delayed class
def proj_ref(self_arg, nf, problem_name, ref, evals=20000, nx=None):
    print("Projecting %s_%i for %s" % (problem_name, nf, str(ref)))
    res = iKRVEA.ACH_solution(self_arg, problem_name=problem_name, nf=nf, refpoint=ref, evals=evals, nx=nx)
    return (nf, problem_name, ref, res)


def platform_max_jobs():
    """ Hard set job limit for paasikivi.it.jyu.fi

    """
    if platform.node() == "paasikivi":
        max_jobs = 15 
    else:
        max_jobs = multiprocessing.cpu_count() - 1
    return max_jobs
    

if __name__ == '__main__':
    max_jobs = platform_max_jobs()
    
    parser = argparse.ArgumentParser(description = 'Create results for ikrvea journal article', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--no-preferences', '-np', action='store_false', help="Do not generate preference information", default=True)
    parser.add_argument('--ikrvea', type=str, help="Path to ik-rvea matlab files", default=r'../')
    parser.add_argument('--jobs', '-j', type = int, metavar = "[1-%i]" % max_jobs,
                        help = "Maximum number concurrent runs",
                        default = multiprocessing.cpu_count() - 1, choices=range(1,max_jobs))
    parser.add_argument('--runs', '-r', type = int, help = "Runs", default = 15)
    parser.add_argument('-p', '--problems', type = str, nargs = '+',
                        help = 'Problems to be solved', default = ['DTLZ1' , 'DTLZ2', 'DTLZ4', 'DTLZ5', 'DTLZ7'])
    parser.add_argument('-d', '--resdir', type = str,
                        help = 'Result directory', default = './ik-rvea')

    parser.add_argument('--clear', action = 'store_true', help = "Clear previous results")

    args = parser.parse_args()

    if args.clear:
        for f in ["preferences.dmp", "PO.dmp"]:
            try:
                os.unlink(os.path.join(args.resdir, f))
            except OSError:
                pass

    ikrvea = iKRVEA(resdir = args.resdir, ikrvea_path = args.ikrvea, NFs = [4], problems = args.problems, runs = args.runs, max_jobs = args.jobs)
    PO = ikrvea.project_prefs(proj_ref)
