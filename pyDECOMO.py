""" Python API for accessing DECOMO project related Matlab BSR_ROUTINES

Requires Matlab Engine API for Python to be installed
See: https://se.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

NOTE:
For python 2.7 as Python 3.x support is not available for older MATLAB


TODO

Instead of launching new matlab, join shared matlab engine,
see https://se.mathworks.com/help/matlab/matlab_external/connect-python-to-running-matlab-session.html

"""

import os

import matlab.engine

import numpy as np


class Problem(object):
    pass


class pyDECOMO(Problem):

    def __init__(self, problem_name, nf, ikrvea):

        self.ikrvea = ikrvea
        self.problem_name = problem_name
        self.nf = nf

        self.eng = self._start_matlab_eng()

        # Use actual Pareto Frontier
        # self.ideal, self.nadir = self.__bounds()

    def _start_matlab_eng(self, extra_paths=[]):

        eng = matlab.engine.start_matlab()
        eng.addpath(self.ikrvea)
        eng.addpath(os.path.join(self.ikrvea, 'Public'))
        for path in extra_paths:
            eng.addpath(path)

        return eng

    def evaluate(self, values):
        """ Evaluate DECOMO problem

        """
        # TODO: Constructing matlab arary cannot be like this!
        try:
            mvalues = values.tolist()
        except AttributeError:
            mvalues = values[:]
        try:
            return list(self.eng.P_objective('value', self.problem_name, matlab.double([self.nf]), matlab.double(mvalues))[0])
        except Exception, e:
            raise e

    def nx(self, problem_name, nf):

        NX = {'DTLZ':[4] + [9] * 3 + [None] * 2 + [19],
            'ZDT' :[30, 30, 30, 30],
            }

        cdir = None
        for key in list(NX.keys()):
            if problem_name.startswith(key):
                idx = int(problem_name[len(key)]) - 1
                nx = nf + NX[key][idx]
                cdir = key
                break
        self.nx = nx
        return self.nx

    def bounds(self):
        """ Find boundaries for the DTLZ problems"""
        self.ideal = [0.0] * self.nf
        self.nadir = [1.0] * self.nf
        if self.problem_name == "DTLZ7":
            self.ideal[-1] = 3.0
            self.nadir[-1] = 16.0
        return self.ideal, self.nadir

    def __bounds(self):
        """ Find boundaries for the DECOMO problem. also initializes the problem"""
        pop = np.array(self.eng.P_objective('true', self.problem_name, matlab.double([self.nf]), 10000))

        ideal = np.min(pop, 0)
        nadir = np.max(pop, 0)
        return ideal, nadir

