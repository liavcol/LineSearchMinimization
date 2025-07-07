from src.utils import Function, ConstrainedProblem, NonNegConstraint, LEConstraint

from typing import Optional

import numpy as np


class QuadraticCircle(Function):
    """
    Description
    -----------
    f(x) = x^T*Q*x, with Q = I.
    """
    
    Q = np.eye(2)

    def __call__(self, x: np.ndarray, hessian: bool = False) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
        f = x.T @ self.Q @ x
        g = 2 * self.Q @ x
        H = 2 * self.Q if hessian else None
        return f, g, H  # type: ignore


class QuadraticEllipse(Function):
    """
    Description
    -----------
    f(x) = x^T*Q*x, with Q = diag(1, 100).
    """
    
    Q = np.diag([1, 100])

    def __call__(self, x: np.ndarray, hessian: bool = False) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
        f = x.T @ self.Q @ x
        g = 2 * self.Q @ x
        H = 2 * self.Q if hessian else None
        return f, g, H  # type: ignore


class QuadraticRotatedEllipse(Function):
    """
    Description
    -----------
    f(x) = x^T*Q*x, with Q = R^T*diag(100, 1)*R, where R is a rotation matrix.
    """
    
    sqrt3_2 = np.sqrt(3) / 2
    R = np.array([[sqrt3_2, -0.5],
                  [0.5,      sqrt3_2]])
    Q = R.T @ np.diag([100, 1]) @ R

    def __call__(self, x: np.ndarray, hessian: bool = False) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
        f = x.T @ self.Q @ x
        g = 2 * self.Q @ x
        H = 2 * self.Q if hessian else None
        return f, g, H  # type: ignore


class Rosenbrock(Function):
    """
    Description
    -----------
    f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2.
    """
    
    def __call__(self, x: np.ndarray, hessian: bool = False) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
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
        return f, g, H  # type: ignore


class LinearExample(Function):
    """
    Description
    -----------
    f(x) = a^T * x, with a = [1, -1].
    """
    
    a = np.array([1, -1])

    def __call__(self, x: np.ndarray, hessian: bool = False) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
        f = self.a @ x
        g = self.a.copy()
        H = np.zeros((2, 2)) if hessian else None
        return f, g, H  # type: ignore


class SmoothedCornerTriangle(Function):
    """
    Description
    -----------
    f(x) = exp(x1+3x2-0.1) + exp(x1-3x2-0.1) + exp(-x1-0.1).
    """
    
    def __call__(self, x: np.ndarray, hessian: bool = False) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
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
        return f, g, H  # type: ignore


class ConstrainedQPExample(ConstrainedProblem):
    """
    Description
    -----------
    A constrained QP exmaple:
    min f = x^2 + y^2 + (z + 1)^2
    s.t.: x + y + z = 1
          x >= 0, y >= 0, z >= 0
    """
    class Func(Function):
        def __call__(self, x: np.ndarray, hessian: bool = False
                    ) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
            x_, y_, z_ = x
            f = x_**2 + y_**2 + (z_ + 1)**2
            g = np.array([2 * x_, 2 * y_, 2 * (z_ + 1)])
            H = np.diag([2, 2, 2]) if hessian else None
            return f, g, H
    
    FUNC = Func()
    INEQ_CONSTRAINTS = [NonNegConstraint(i) for i in range(3)]
    EQ_CONSTRAINT_MAT = np.array([[1, 1, 1]])
    EQ_CONSTRAINT_RHS = np.array([1])


class ConstrainedLPExample(ConstrainedProblem):
    """
    Description
    -----------
    A constrained LP example:
    max f = x + y
    s.t.: y >= -x + 1
          y <= 1, x <= 2, y >= 0 
    """
    class Func(Function):
        """
        Description
        -----------
        max f = x + y -> min f = -x - y
        """
        def __call__(self, x: np.ndarray, hessian: bool = False
                    ) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
            f = -x[0] - x[1]
            g = np.array([-1, -1])
            H = np.zeros((2, 2)) if hessian else None
            return f, g, H
    
    class Constraint(Function):
        """
        Description
        -----------
        y >= -x + 1 -> -x - y + 1 <= 0
        """
        def __call__(self, x: np.ndarray, hessian: bool = False
                     ) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
            val = -x[0] - x[1] + 1
            grad = np.array([-1, -1])
            H = np.zeros((2, 2)) if hessian else None
            return val, grad, H
    
    FUNC = Func()
    INEQ_CONSTRAINTS = [
        Constraint(), LEConstraint(i=1, le_than=1), LEConstraint(i=0, le_than=2), NonNegConstraint(i=1)
    ]
    EQ_CONSTRAINT_MAT = np.array([])
    EQ_CONSTRAINT_RHS = np.array([])
    