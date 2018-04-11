# coding: utf-8

import logging
import random
import pickle

import os

import numpy as np
import pandas as pd
from sklearn import tree
from scipy import spatial

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d]  %(message)s', filename='ADM.log', datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def iterate():
    return [True, [.5, .5, .5, .5, .5, .5, .1]]


def ffile(name, d="results"):

    return os.path.join(os.path.dirname(os.path.realpath(__file__)), d, name)


def preferences(problem, nf, it, resdir="results"):
    preferences = pickle.load(open(ffile('preferences.dmp',resdir), 'r'))
    pref = preferences[(int(nf), problem)][int(it)]
    pareto_points = pickle.load(open(ffile('PO.dmp',resdir), 'r'))
    aspPO = pareto_points[(int(nf), problem, pref)]
    logger.debug("Preference %i\n" % it, (pref, aspPO))
    return (pref, aspPO)


class hADM:
    """ Automatic Decision Maker (ADM)
    introduced in PPSN2016 utilizing decision tree for the current context

    :var preference [double, double]: [Initial reference point, weights for each objective]
    :var aspPO list[double]: Project intial reference points

    """

    def __init__(self,
                 problem='DLTZ2',
                 nf=3,
                 method='InteractiveMethod',
                 preference=None,  # Initial preferences and corresponding PO
                 nx=None,
                 resdir='./results',
                 tol=0.01,
                 p=.9,
                 run=0,
                 extra_parameters=None):
        self.max_iter = 10
        self.problem = problem
        self.nf = int(nf)
        self.stop = False
        self.method = method
        self.ideal = [0.0] * self.nf
        self.nadir = [1.0] * self.nf
        if problem == "DTLZ7":
            self.ideal[-1] = 3.0
            self.nadir[-1] = 16.0
        self.nx = nx
        if preference is None:
            self.preference = self._random_pref()
            self.aspPO = None
        else:
            self.preference = preference[0]
            self.aspPO = preference[1]
        self.resdir = resdir
        self.prev_refp = self.preference[0]
        self.refs = [self.preference[0][:]]

        self.extra_parameters = extra_parameters

        self.run = run

        self.iter = 0  # Current iteration
        self.sPO = []  # Set of Pareto optimal solutions selected by the ADM
        self.tol = tol
        self.p = p

        logger.info("Initial preference:\n%s\n%s" % preference)

    def stop(self):
        return self.stop

    def _fn(self, fn):
        if not os.path.exists(self.resdir):
            os.makedirs(self.resdir)

        return os.path.join(self.resdir, fn)

    def _proj_ref(self, refp):
        try:
            PO = pickle.load(open(self._fn("PO.dmp"), "r"))
        except IOError:
            PO = {}
        if (self.nf, self.problem, tuple(refp)) not in PO:
            # We do not wish to start JVM if not needed
            import pyMOEA
            logger.info("Updating PO.dmp")
            PO[(self.nf, self.problem, tuple(refp))] = pyMOEA.proj_ref(self.nf, self.problem, refp, evals=50000, nx=self.nx)[-1]
            pickle.dump(PO, open(self._fn("PO.dmp"), "w"))
        return PO[(self.nf, self.problem, tuple(refp))]

    def _random_pref(self):
        aspir = self.nadir[:]
        w = [None] * self.nf
        for k in range(self.nf):
            w[k] = random.random()
            if not random.random() < .5 * w[k]:
                aspir[k] = random.uniform(self.ideal[k], self.nadir[k])
        return tuple(aspir), tuple(w)

    def next_refp(self, ref, objs):
        """ Return next reference point, or None if the process is finished
        """
        logger.debug("Previous reference point:\n", ref)
        aspir = self.preference[0]
        w = self.preference[1]

        nadir = self.nadir
        new_ref = [0.0] * self.nf
        # create the decision tree
        A = np.array(objs)

        trees = []
        for i in range(self.nf):
            trees.append(tree.DecisionTreeRegressor())
            trees[-1].fit(
                A[:, list(range(0, i)) + list(range(i + 1, self.nf))], A[:, i])

        if len(objs) == 1:
            obj = objs[-1]
        else:
            A = np.array(objs)
            self.sPO.append(list(A[spatial.KDTree(A).query(self.aspPO)[1]]))
            obj = list(A[spatial.KDTree(A).query(self.aspPO)[1]])

        self.sPO.append(obj)
        S = [None] * len(obj)
        pk = self.p

        k = 0
        for k in reversed(np.argsort(w)):
            fk = obj[k]
            if abs(aspir[k] - fk) < self.tol and random.random() < self.p:
                S[k] = fk
            elif random.random() < pk:
                S[k] = fk
            if len(S):
                pk -= pk / len(obj) * len(S)
        if len(S) == k:
            return ref

        for k, s in enumerate(S):
            if s is not None:
                new_ref[k] = min(aspir[k] - (aspir[k] - obj[k]) / 2., nadir[k])
            else:
                new_ref[k] = None

        idx = [idx for idx, v in enumerate(new_ref) if v is None]
        for i in idx:
            n = [0.0 if v is None else v for v in new_ref]
            del n[i]
            new_ref[i] = min(trees[i].predict(n)[0], nadir[i])
        if np.array_equal(nadir, new_ref):
            return ref
        logger.debug("New reference point:\n%s" % new_ref)
        return new_ref

    def save(self, run):
        fn = self._fn("%s-%s_%i.csv" % (self.method, self.problem, self.nf))

        columns = ['problem', 'nf', 'pref_id', 'refs', 'PO'] + self.extra_parameters.keys()
        df = pd.DataFrame(columns=columns)

        df['PO'] = [self.sPO]
        df['problem'] = self.problem
        df['nf'] = self.nf
        for extra in self.extra_parameters.keys():
            df[extra] = self.extra_parameters[extra]
        df['refs'] = [self.refs]

        df['pref_id'] = self.run
        if os.path.exists(fn):
            with open(fn, "a") as f:
                df.to_csv(f, header=False, index=False)
        else:
            with open(fn, "a") as f:
                df.to_csv(f, index=False)

    def next_iteration(self, PO):
        try:
            if isinstance(PO, (np.ndarray, np.generic)):
                PO = PO.reshape(len(PO) / self.nf, self.nf).tolist()

            logger.info("Solving: %s:%i\n%s", self.problem, self.nf, PO)

            if self.refs[-1] is not None and self.iter < self.max_iter:
                new_ref = self.next_refp(self.refs[-1], PO)
                if np.array_equal(new_ref, self.prev_refp):
                    self.stop = True
                    return None
                self.refs.append(new_ref[:])
                self.iter += 1
            else:
                self.stop = True
                return None
            self.prev_refp = new_ref
            logger.info("Iteration %i: %s %i:\n%s" , self.iter, self.problem, self.nf, new_ref)
            return new_ref
        except ValueError:
            self.stop = True
            logger.exception("Failed to create new refpoint")
            return None
