from .utils import backtracking_wolfe

import numpy as np
from enum import StrEnum, auto
from typing import Callable


class OptimizationMethod(StrEnum):
    GRADIENT_DESCENT = auto()
    NEWTON_METHOD = auto()


class UnconstrainedMinimizer:
    def __init__(self, method: OptimizationMethod):
        """
        Parameters
        ----------
        method : OptimizationMethod
            The optimiztion method to use.
        """
        self.method = method

        # Will record the entire optimization path:
        self.x_path: list[np.ndarray] = []
        self.f_path: list[float] = []

    @staticmethod
    def __backtracking_wolfe(
        f: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]],
        x_k: np.ndarray,
        p_k: np.ndarray,
        alpha0: float = 1.0,
        rho: float = 0.5,
        c: float = 1e-4,
    ) -> float:
        alpha = alpha0
        f_x, g_x, _  = f(x_k)
        c_gx_pk = c * g_x.dot(p_k)

        while f(x_k + alpha * p_k)[0] > f_x + alpha * c_gx_pk:
            alpha *= rho
        
        return alpha

    def __gradient_descent(self, f: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]],
                           x0: np.ndarray,
                            obj_tol: float = 1e-6, param_tol: float = 1e-6, max_iter: int = 100):
        """
        Gradient descent method.

        Parameters
        ----------
        f : callable, f(x) -> float
            The objective function to minimize.

        Returns
        -------
        x_star : np.ndarray
            The final point.
        f_star : float
            The final objective value.
        success : bool
            True if any termination criterion met.
        """
        x_k = x0.copy()
        f_k, g_k, _ = f(x_k)
        self.x_path = [x_k.copy()]
        self.f_path = [f_k]
        
        for i in range(1, max_iter + 1):
            # Find search direction
            p_k = -g_k

            # Find step length using backtracking line search
            alpha_k = UnconstrainedMinimizer.__backtracking_wolfe(f, x_k, p_k)

            # Take step
            x_next = x_k + alpha_k * p_k
            f_next, g_next, _ = f(x_next)

            print(f'[Iter {i:3d}] x = {x_next}, f = {f_next:.6e}, α = {alpha_k:.3e}')

            # Check termination criteria: objective change is sufficiently small or step length is small.
            if abs(f_next - f_k) < obj_tol or np.linalg.norm(x_next - x_k) < param_tol:
                return x_next, f_next, True

            x_k, f_k, g_k = x_next, f_next, g_next
            self.x_path.append(x_k.copy())
            self.f_path.append(f_k)

        return x_k, f_k, False


    def __newtons_method(self, f: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]],
                           x0: np.ndarray,
                            obj_tol: float = 1e-6, param_tol: float = 1e-6, max_iter: int = 100):
        x_k = x0.copy()
        f_k, g_k, H_k = f(x_k)
        self.x_path = [x_k.copy()]
        self.f_path = [f_k]

        for i in range(1, max_iter + 1):
            # Find search direction
            # `np.linalg.solve` is used to solve the linear system H_k * x = g_k.
            # The solution x is given by H_k^{-1} * g_k, which is what we learned in class.
            p_k = -np.linalg.solve(H_k, g_k)

            # Find step length using backtracking line search
            alpha_k = UnconstrainedMinimizer.__backtracking_wolfe(f, x_k, p_k)

            # Take step
            x_next = x_k + alpha_k * p_k
            f_next, g_next, H_next = f(x_next)

            print(f'[Iter {i:3d}] x = {x_next}, f = {f_next:.6e}, α = {alpha_k:.3e}')

            # Check termination criteria: Newton's decreament or objective change is sufficiently small or step length is small.
            decrement = g_next.dot(np.linalg.solve(H_next, g_next))
            if decrement / 2 < obj_tol or abs(f_next - f_k) < obj_tol or np.linalg.norm(x_next - x_k) < param_tol:
                return x_next, f_next, True

            x_k, f_k, g_k = x_next, f_next, g_next
            self.x_path.append(x_k.copy())
            self.f_path.append(f_k)

        return x_k, f_k, False


    def minimize(self,
                 f: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]],
                 x0: np.ndarray,
                 obj_tol: float = 1e-6, param_tol: float = 1e-6, max_iter: int = 100):
        """
        Minimize a smooth function with either gradient descent or Newton’s method.

        Parameters
        ----------
        f        : callable, f(x) -> float
        grad     : callable, ∇f(x) -> np.ndarray
        hess     : callable, ∇²f(x) -> np.ndarray (only used if method="newton")
        x0       : np.ndarray, initial guess
        obj_tol  : tolerance on |f_{k+1} - f_k| or Newton decrement/2
        param_tol: tolerance on ‖x_{k+1} − x_k‖
        max_iter : maximum number of iterations

        Returns
        -------
        x_star   : final point
        f_star   : final objective value
        success  : bool (True if any termination criterion met)
        """
        if self.method == OptimizationMethod.GRADIENT_DESCENT:
            return self.__gradient_descent(f, x0, obj_tol, param_tol, max_iter)
        elif self.method == OptimizationMethod.NEWTON_METHOD:
            return self.__newtons_method(f, x0, obj_tol, param_tol, max_iter)
        else:
            raise ValueError(f'Unknown optimization method: {self.method}')