from .utils import Function

from enum import StrEnum, auto

import numpy as np


class OptimizationMethod(StrEnum):
    """
    Description
    -----------
    Enum for optimization methods.
    """
    GRADIENT_DESCENT = auto()
    NEWTON_METHOD = auto()


class UnconstrainedMinimizer:
    def __init__(self, method: OptimizationMethod):
        """
        Summary
        -------
        Initialize the UnconstrainedMinimizer with a specific optimization method.

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
    def __backtracking_wolfe(f: Function, x_k: np.ndarray, p_k: np.ndarray,
                             rho: float = 0.5, c: float = 0.01) -> float:
        """
        Summary
        -------
        Backtracking line search to find step length that satisfies the Wolfe conditions.

        Parameters
        ----------
        f : ObjectiveFunction
            The objective function to minimize.
        x_k : np.ndarray
            Current point.
        p_k : np.ndarray
            Search direction.
        rho : float, default = 0.5
            Step length reduction factor.
        c : float, default = 0.01
            Wolfe condition parameter.
        
        Returns
        -------
        float
            Step length that satisfies the Wolfe conditions.
        """
        alpha = 1
        fx, gx, _  = f(x_k)
        c_gx_pk = c * gx.dot(p_k)

        while f(x_k + alpha * p_k)[0] > fx + alpha * c_gx_pk:
            alpha *= rho
        
        return alpha

    def __gradient_descent(self, f: Function, x0: np.ndarray,
                           obj_tol: float = 10e-12, param_tol: float = 10e-8,
                           max_iter: int = 100) -> tuple[np.ndarray, float, bool]:
        x_k = x0.copy()
        f_k, g_k, _ = f(x_k)
        self.x_path = [x_k.copy()]
        self.f_path = [f_k]
        
        print('Running Gradient Descent...')
        print(f'[Iter {0:3d}] x = {x_k}, f(x) = {f_k:.6e}')

        for i in range(1, max_iter + 1):
            # Find search direction
            p_k = -g_k

            # Find step length using backtracking line search
            alpha_k = UnconstrainedMinimizer.__backtracking_wolfe(f, x_k, p_k)

            # Take step
            x_next = x_k + alpha_k * p_k
            f_next, g_next, _ = f(x_next)

            print(f'[Iter {i:3d}] x = {x_next}, f(x) = {f_next:.6e}')
            self.x_path.append(x_next.copy())
            self.f_path.append(f_next)

            # Check termination criteria: objective change is sufficiently small or step length is small.
            if abs(f_next - f_k) < obj_tol or np.linalg.norm(x_next - x_k) < param_tol:
                return x_next, f_next, True

            x_k, f_k, g_k = x_next, f_next, g_next

        return x_k, f_k, False


    def __newtons_method(self, f: Function, x0: np.ndarray,
                         obj_tol: float = 10e-12, param_tol: float = 10e-8,
                         max_iter: int = 100) -> tuple[np.ndarray, float, bool]:
        x_k = x0.copy()
        f_k, g_k, H_k = f(x_k, hessian=True)
        self.x_path = [x_k.copy()]
        self.f_path = [f_k]

        print('Running Newton Method...')
        print(f'[Iter {0:3d}] x = {x_k}, f(x) = {f_k:.6e}')

        # Find search direction
        # `np.linalg.solve` is used to solve the linear system H_k * x = g_k.
        # The solution x is given by H_k^{-1} * g_k, which is what we learned in class.
        try:
            p_k = -np.linalg.solve(H_k, g_k)
        except np.linalg.LinAlgError:
            p_k = -g_k  # Fallback to gradient descent if H_k is singular

        for i in range(1, max_iter + 1):
            # Find step length using backtracking line search
            alpha_k = UnconstrainedMinimizer.__backtracking_wolfe(f, x_k, p_k)

            # Take step
            x_next = x_k + alpha_k * p_k
            f_next, g_next, H_next = f(x_next, hessian=True)

            print(f'[Iter {i:3d}] x = {x_next}, f(x) = {f_next:.6e}')
            self.x_path.append(x_next.copy())
            self.f_path.append(f_next)

            # Update search direction for next iteration as we'll need it for checking termination criteria.
            try:
                p_k = -np.linalg.solve(H_next, g_next)
            except np.linalg.LinAlgError:
                p_k = -g_next  # Fallback to gradient descent if H_k is singular

            # Check termination criteria: Newton's decreament or objective change is sufficiently small or step length is small.

            decrement = abs(g_next.dot(p_k))
            if decrement / 2 < obj_tol or abs(f_next - f_k) < obj_tol or np.linalg.norm(x_next - x_k) < param_tol:
                return x_next, f_next, True

            x_k, f_k, g_k = x_next, f_next, g_next
            
        return x_k, f_k, False


    def minimize(self, f: Function, x0: np.ndarray,
                 obj_tol: float = 10e-12, param_tol: float = 10e-8,
                 max_iter: int = 100) -> tuple[np.ndarray, float, bool]:
        """
        Minimize a function.

        Parameters
        ----------
        f : ObjectiveFunction
            The objective function to minimize.
        x0 : np.ndarray
            Initial point, shape (n,).
        obj_tol : float, default = 10e-12
            Objective function change tolenrance for termination criterion.
        param_tol : float, default = 10e-8 
        max_iter : int, default = 100
            Maximum number of iterations to run.

        Returns
        -------
        np.ndarray
            Final point reached.
        float
            Final objective value reached.
        bool
            True if any termination criterion met.
        """
        if self.method == OptimizationMethod.GRADIENT_DESCENT:
            return self.__gradient_descent(f, x0, obj_tol, param_tol, max_iter)
        elif self.method == OptimizationMethod.NEWTON_METHOD:
            return self.__newtons_method(f, x0, obj_tol, param_tol, max_iter)
        else:
            raise ValueError(f'Unknown optimization method: {self.method}')