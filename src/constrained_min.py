from .utils import Function
import numpy as np


class ConstrainedMinimizer:
    def __init__(self):
        self.central_path = []
        self.objective_values = []

    def interior_pt(self, func: Function, ineq_constraints: list[Function], eq_constraints_mat: np.ndarray, eq_constraints_rhs: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """
        Summary
        -------
        Solve a constrained minimization problem using the interior point method.

        Parameters
        ----------
        func : Function
            The objective function to minimize.
        ineq_constraints : list[Function]
            List of inequality constraint functions.
        eq_constraints_mat : np.ndarray
            Coefficients of equality constraints.
        eq_constraints_rhs : np.ndarray
            Right-hand side values of equality constraints.
        x0 : np.ndarray
            Initial guess for the optimization variables.
    
        Returns
        -------
        np.ndarray
            Optimal solution that satisfies the constraints.
        """
        m = len(ineq_constraints)
        x = x0.copy()
        t = 1

        self.central_path = [x.copy()]
        self.objective_values = []

        def log_barrier_obj(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
            f_val, f_grad, f_hess = func(x, hessian=True)
            barrier = 0
            barrier_grad = np.zeros_like(x)
            barrier_hess = np.zeros((x.shape[0], x.shape[0]))

            for g in ineq_constraints:
                g_val, g_grad, g_hess = g(x, hessian=True)
                if g_val >= 0:
                    raise ValueError('All inequality constraints must be strictly negative for the log barrier method.')

                barrier -= np.log(-g_val)
                barrier_grad += g_grad / -g_val

                barrier_hess += (np.outer(g_grad, g_grad) / (g_val**2)) + (-g_hess / g_val)  # type: ignore

            total_val = t * f_val + barrier
            total_grad = t * f_grad + barrier_grad
            total_hess = t * f_hess + barrier_hess  # type: ignore
            return total_val, total_grad, total_hess

        A = eq_constraints_mat.copy()
        b = eq_constraints_rhs.copy()
        has_eq = A.size and b.shape[0] > 0

        while True:
            # Solve centering problem with equality constraints using Newton's method
            while True:
                _, grad, hess = log_barrier_obj(x)
                if has_eq:
                    # KKT system
                    KKT_top = np.hstack((hess, A.T))
                    KKT_bottom = np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))
                    KKT = np.vstack((KKT_top, KKT_bottom))
                    rhs = -np.concatenate([grad, A @ x - b])

                    try:
                        sol = np.linalg.solve(KKT, rhs)
                    except np.linalg.LinAlgError:
                        raise ValueError('KKT system is singular.')

                    dx = sol[:x.shape[0]]
                else:
                    # Unconstrained case
                    try:
                        dx = -np.linalg.solve(hess, grad)
                    except np.linalg.LinAlgError:
                        raise ValueError('Hessian is singular.')

                if np.linalg.norm(dx) < 10e-8:
                    break

                # Backtracking line search to maintain feasibility
                alpha = 1
                while any(g(x + alpha * dx, hessian=False)[0] >= 0 for g in ineq_constraints):
                    alpha *= 0.5

                x += alpha * dx

            self.central_path.append(x.copy())
            f_val, _, _ = func(x, hessian=False)
            self.objective_values.append(f_val)

            # Stop if duality gap small
            if m / t < 10e-8:
                break

            t *= 10

        return x
