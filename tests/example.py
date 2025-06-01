from src.utils import ObjectiveFunction

from typing import Optional

import numpy as np


class QuadraticCircle(ObjectiveFunction):
    """
    Description
    -----------
    f(x) = x^T*Q*x, with Q = I.
    """
    
    Q = np.eye(2)

    @classmethod
    def __call__(cls, x: np.ndarray, hessian: bool = False) -> tuple[float, np.ndarray, Optional[np.ndarray]]:            
        f = x.T @ cls.Q @ x
        g = 2 * cls.Q @ x
        H = 2 * cls.Q if hessian else None
        return f, g, H


class QuadraticEllipse(ObjectiveFunction):
    """
    Description
    -----------
    f(x) = x^T*Q*x, with Q = diag(1, 100).
    """
    
    Q = np.diag([1, 100])

    @classmethod
    def __call__(cls, x: np.ndarray, hessian: bool = False):
        f = x.T @ cls.Q @ x
        g = 2 * cls.Q @ x
        H = 2 * cls.Q if hessian else None
        return f, g, H


class QuadraticRotatedEllipse(ObjectiveFunction):
    """
    Description
    -----------
    f(x) = x^T*Q*x, with Q = R^T*diag(100, 1)*R, where R is a rotation matrix.
    """
    
    sqrt3_2 = np.sqrt(3) / 2
    R = np.array([[sqrt3_2, -0.5],
                  [0.5,      sqrt3_2]])
    Q = R.T @ np.diag([100, 1]) @ R

    @classmethod
    def __call__(cls, x: np.ndarray, hessian: bool = False):
        f = x.T @ cls.Q @ x
        g = 2 * cls.Q @ x
        H = 2 * cls.Q if hessian else None
        return f, g, H


class Rosenbrock(ObjectiveFunction):
    """
    Description
    -----------
    f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2.
    """
    
    @classmethod
    def __call__(cls, x: np.ndarray, hessian: bool = False):
        x1, x2 = x
        f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
        df_dx1 = 400 * (-x2*x1 + x1**3) + 2 * (-1 + x1)
        df_dx2 = 200 * (x2 - x1**2)
        g = np.array([df_dx1, df_dx2])
        H = None
        if hessian:
            d2f_dx1dx1 = 400 * (-x2 + 3*x1**2) + 2
            d2f_dx1dx2 = -400 * x1
            d2f_dx2dx2 = 200
            H = np.array([[d2f_dx1dx1, d2f_dx1dx2],
                          [d2f_dx1dx2, d2f_dx2dx2]])
        return f, g, H


class LinearExample(ObjectiveFunction):
    """
    Description
    -----------
    f(x) = a^T * x, with a = [1, -1].
    """
    
    a = np.array([1, -1])

    @classmethod
    def __call__(cls, x: np.ndarray, hessian: bool = False):
        f = cls.a @ x
        g = cls.a.copy()
        H = np.zeros((2, 2)) if hessian else None
        return f, g, H


class SmoothedCornerTriangle(ObjectiveFunction):
    """
    Description
    -----------
    f(x) = exp(x1+3x2-0.1) + exp(x1-3x2-0.1) + exp(-x1-0.1).
    """
    
    @classmethod
    def __call__(cls, x: np.ndarray, hessian: bool = False):
        x1, x2 = x
        e1 = np.exp(x1 + 3 * x2 - 0.1)
        e2 = np.exp(x1 - 3 * x2 - 0.1)
        e3 = np.exp(-x1 - 0.1)
        f = e1 + e2 + e3
        df_dx1 = e1 + e2 - e3
        df_dx2 = 3 * e1 - 3 * e2
        g = np.array([df_dx1, df_dx2])
        H = None
        if hessian:
            d2f_dx1dx1 = e1 + e2 + e3
            d2f_dx1dx2 = 3 * e1 - 3 * e2
            d2f_dx2dx2 = 9 * e1 + 9 * e2
            H = np.array([[d2f_dx1dx1, d2f_dx1dx2],
                          [d2f_dx1dx2, d2f_dx2dx2]])
        return f, g, H
