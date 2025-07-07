import example
from src.constrained_min import ConstrainedMinimizer
import src.utils as utils

import unittest

import numpy as np


class TestExamples(unittest.TestCase):

    def __run_example(self, problem, x0):
        minimizer = ConstrainedMinimizer()

        x_opt = minimizer.interior_pt(
            problem.FUNC,
            problem.INEQ_CONSTRAINTS,
            problem.EQ_CONSTRAINT_MAT,
            problem.EQ_CONSTRAINT_RHS,
            x0
        )

        utils.plot_feasible_region_and_path(problem, minimizer.central_path)
        utils.plot_objective_history({'': minimizer.objective_values}, problem.__class__.__name__)

        print(f'Optimal solution: {np.round(x_opt, 3)}\n'
              f'Objective value at optimal: {np.round(problem.FUNC(x_opt, False)[0], 3)}\n'
              f'Inequality constraints at optimal:\n    * '
              f'{"\n    * ".join([str(np.round(g(x_opt, False)[0], 3)) for g in problem.INEQ_CONSTRAINTS])}\n')

    
    def test_qp(self):
        problem = example.ConstrainedQPExample()
        x0 = np.array([0.1, 0.2, 0.7])

        self.__run_example(problem, x0)

    def test_lp(self):
        problem = example.ConstrainedLPExample()
        x0 = np.array([0.5, 0.75])

        self.__run_example(problem, x0)