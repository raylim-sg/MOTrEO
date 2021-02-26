# This code is for building non-linear source-to-target mapping Ms in continuous search spaces

import scipy as sp
import numpy as np
import scipy.special
import copy
from domain_adaptation.probabilisticModel import ProbabilisticModel


def learn_map(source_solutions, target_solutions, reg):
    source_sols = copy.deepcopy(source_solutions)
    if source_sols.shape[0] != target_solutions.shape[0]:
        raise Exception('Number of source data instances must be equal to the number of target instances.')

    source_sols = source_sols.transpose()
    source_sols = np.concatenate((source_sols, np.array([np.ones(source_sols.shape[1])])), axis = 0)
    M = NonlinearTrans()
    M.learn(source_sols, target_solutions.transpose(), reg)
    return M


def solution_transform(solution, M):
    # To return transformed solutions such that variables bounded in [0, 1]
    sol = copy.deepcopy(solution)
    sol = np.append(sol, 1)
    new_solution = M.transform(sol)
    new_solution[new_solution < 0] = 0
    new_solution[new_solution > 1] = 1
    return new_solution


def model_transform(source_model, M, nSol = None):
    # Monte Carlo sampling based approximate source model transformation
    if nSol is None:
        nSol = 10*M.outputs

    solutions = source_model.sample(nSol)
    transformed_solutions1 = [solution_transform(solutions[i], M) for i in range(nSol)]
    transformed_solutions = np.array(transformed_solutions1)

    new_model = ProbabilisticModel('mvarnorm')
    new_model.buildModel(transformed_solutions)

    return new_model


class NonlinearTrans(object):
    def __init__(self):
        self.layer1 = None
        self.layer2 = None
        self.inputs = None
        self.hidden_nodes = None
        self.outputs = None
        self.activation = 'expit'

    def learn(self, source_solutions, target_solutions, reg):
        self.inputs = source_solutions.shape[0]
        self.outputs = target_solutions.shape[0]
        self.hidden_nodes = 2*self.outputs
        self.layer1 = sp.randn(self.hidden_nodes, self.inputs)
        self.layer2 = np.zeros([self.outputs, self.hidden_nodes])

        if target_solutions.shape[1] < self.hidden_nodes + 1:
            raise Exception('Population size insufficient for learning nonlinear map.')

        layer1_transformation = np.matmul(self.layer1, source_solutions)
        layer1_transformation = sp.special.expit(layer1_transformation)

        for i in range(self.outputs):
            temp1 = sp.linalg.inv(np.matmul(layer1_transformation, layer1_transformation.transpose()) +
                                  (1 + sp.rand(1)*(reg)) * np.identity(self.hidden_nodes))
            temp2 = np.matmul(layer1_transformation, target_solutions[i])
            self.layer2[i,:] = np.matmul(temp1, temp2).transpose()

    def transform(self, solution):
        new_solution = np.matmul(self.layer2, sp.special.expit(np.matmul(self.layer1, solution)))
        return new_solution
