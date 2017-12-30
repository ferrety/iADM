# coding: utf-8

import logging
import random
import pickle

import os

import numpy as np
from sklearn import tree
from scipy import spatial


def iterate():
    return [True, [.5, .5, .5, .5, .5, .5, .1]]


RUNPATH = r'D:\JYU\iADM'


def ffile(name, d=""):

    return os.path.join(RUNPATH, d, name)


def preferences(nf, it):
    preferences = pickle.load(open(ffile('preferences.dmp'), 'r'))
    print(nf, it, preferences[int(nf)][int(it)])
    return preferences[int(nf)][int(it)]


class hADM:
    """ Automatic Decision Maker (ADM)
    introduced in PPSN2016 utilizing decision tree for the current context"""

    def __init__(self,
                 problem='DLTZ2',
                 nf=3,
                 preference=None,
                 nx=None,
                 tol=0.01,
                 p=.9):
        self.problem = problem
        self.nf = int(nf)
        self.stop = False
        self.ideal = [0.0] * self.nf
        self.nadir = [1.0] * self.nf
        self.ideal[-1] = 4.0
        self.nadir[-1] = 6.0
        self.nx = int(nx)
        if preference is None:
            self.preference = self._random_pref()
        else:
            self.preference = preference
        self.prev_refp = self.preference[0]
        self.refs = [self.preference[0][:]]

        self.aspPO = self._proj_ref(self.preference[0])
        self.iter = 0  # Current iteration
        self.sPO = []  # Set of Pareto optimal solutions selected by ADM
        self.tol = tol
        self.p = p

    def _proj_ref(self, refp):
        PO = pickle.load(open(ffile("PO.dmp"), "r"))
        if (self.nf, self.problem, tuple(refp)) in PO:
            # We do not wish to start JVM if not needed
            import pyMOEA
            logging.info("Updating PO.dmp")
            PO[(self.nf, self.problem, tuple(refp))] = pyMOEA.proj_ref(
                self.nf, self.problem, refp, evals=50000, nx=self.nx)[-1]
            pickle.dump(PO, open(ffile("PO.dmp"), "w"))
        return PO[(self.nf, self.problem, tuple(refp))]

    def _random_pref(self):
        aspir = self.ideal[:]
        w = [None] * self.nf
        for k in range(self.nf):
            w[k] = random.random()
            if random.random() < .5:
                aspir[k] = random.uniform(self.ideal[k], self.nadir[k])
        return aspir, w

    def next_refp(self, ref, objs):
        """ Return next reference point, or None if the process is finished
        """

        aspir = self.preference[0]
        w = self.preference[1]
        nf = len(aspir)
        nadir = self.nadir
        new_ref = [0.0] * nf
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
        return new_ref

    def save(self, method, run):
        fn = ffile("%s-%s_%i.dmp" % (method, self.problem, self.nf))
        try:
            res = pickle.load(open(fn, "r"))
        except IOError:
            res = {}
        res[(self.problem, self.nf, run)] = (self.refs, self.sPO)
        pickle.dump(res, open(fn, "w"))

    def next_iteration(self, PO):
        logging.info("Solving: %s:%i", self.problem, self.nf)
        try:
            if isinstance(PO, (np.ndarray, np.generic)):
                PO = PO.reshape(len(PO) / self.nf, self.nf).tolist()

            if self.refs[-1] is not None and self.iter < 10:
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
            logging.info("Solved: %s:\n%s", self.problem, new_ref)
            return new_ref
        except ValueError:
            self.stop = True
            return None
