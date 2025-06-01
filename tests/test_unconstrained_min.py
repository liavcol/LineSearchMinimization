import example
from src.unconstrained_min import UnconstrainedMinimizer, OptimizationMethod
import src.utils as utils

import unittest

import numpy as np


class TestExamples(unittest.TestCase):

    def run_example(self, f: utils.ObjectiveFunction, x0: np.ndarray, max_iter: int = 100):
        gd_optimizer = UnconstrainedMinimizer(OptimizationMethod.GRADIENT_DESCENT)
        x_opt_gd, f_opt_gd, success_gd = gd_optimizer.minimize(f, x0, max_iter=max_iter)

        nt_optimizer = UnconstrainedMinimizer(OptimizationMethod.NEWTON_METHOD)
        x_opt_nt, f_opt_nt, success_nt = nt_optimizer.minimize(f, x0, max_iter=max_iter)

        print('Gradient Descent Results:\n'
              f'Iterations: {len(gd_optimizer.x_path)}\n'
              f'Final point: {x_opt_gd}\n'
              f'Final objective value: {f_opt_gd}\n'
              f'Success: {success_gd}\n'
              '---\n'
              'Newton Method Results:\n'
              f'Iterations: {len(nt_optimizer.x_path)}\n'
              f'Final point: {x_opt_nt}\n'
              f'Final objective value: {f_opt_nt}\n'
              f'Success: {success_nt}')

        utils.plot_contour_with_path(f, {
            'Gradient Descent': gd_optimizer.x_path,
            'Newton Method': nt_optimizer.x_path
        })

        utils.plot_objective_history({
            'Gradient Descent': gd_optimizer.f_path,
            'Newton Method': nt_optimizer.f_path
        }, f.__class__.__name__)

    def test_quadratic_circle(self):
        f = example.QuadraticCircle()
        x0 = np.array([1, 1])

        self.run_example(f, x0)
    
    def test_quadratic_ellipse(self):
        f = example.QuadraticEllipse()
        x0 = np.array([1, 1])

        self.run_example(f, x0)
    
    def test_quadratic_rotated_ellipse(self):
        f = example.QuadraticRotatedEllipse()
        x0 = np.array([1, 1])

        self.run_example(f, x0)
    
    def test_rosenbrock(self):
        f = example.Rosenbrock()
        x0 = np.array([-1, 2])

        self.run_example(f, x0, max_iter=10000)
    
    def test_linear_example(self):
        f = example.LinearExample()
        x0 = np.array([1, 1])

        self.run_example(f, x0)
    
    def test_smoothed_corner_triangle(self):
        f = example.SmoothedCornerTriangle()
        x0 = np.array([1, 1])

        self.run_example(f, x0)
