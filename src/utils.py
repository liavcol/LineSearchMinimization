from typing import Optional

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



class Function(ABC):
    """
    Description
    -----------
    Base class for objective functions.
    """

    @abstractmethod
    def __call__(cls, x: np.ndarray, hessian: bool = False
                 ) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
        """
        Summary
        -------
        Evaluate the function at point x.
        
        Parameters
        ----------
        x : np.ndarray
            Input point, shape (n,).
        hessian : bool, default False
            Whether to compute the Hessian matrix.
        Returns
        -------
        float
            Function value at x.
        np.ndarray
            Gradient at x, shape (n,).
        Optional[np.ndarray]
            Hessian matrix at x, shape (n, n) if hessian is True, otherwise None. 
        """
        pass


class ConstrainedProblem(ABC):
    """
    Description
    -----------
    Base class for constrained optimization problems.
    This class defines the structure of a constrained optimization problem.
    """
    FUNC: Function
    INEQ_CONSTRAINTS: list[Function]
    EQ_CONSTRAINT_MAT: np.ndarray
    EQ_CONSTRAINT_RHS: np.ndarray


class NonNegConstraint(Function):
    """
    Description
    -----------
    Create a non-negativity constraint function for the i-th variable.
    """

    def __init__(self, i: int) -> None:
        super().__init__()
        self.i = i

    def __call__(self, x: np.ndarray, hessian: bool = False) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
        val = -x[self.i]
        dim = x.shape[0]
        grad = np.zeros(dim)
        grad[self.i] = -1.0
        hess = np.zeros((dim, dim)) if hessian else None
        return val, grad, hess


class LEConstraint(Function):
    """
    Description
    -----------
    Create a less-than-or-equal-to constraint function for the i-th variable.
    This constraint checks if the i-th variable is less than or equal to a specified value.
    """
    def __init__(self, i: int, le_than: float) -> None:
        super().__init__()
        self.i = i
        self.le_than = le_than
    
    def __call__(self, x: np.ndarray, hessian: bool = False) -> tuple[np.float64, np.ndarray, Optional[np.ndarray]]:
        val = x[self.i] - self.le_than 
        dim = x.shape[0]
        grad = np.zeros(dim)
        grad[self.i] = 1.0
        H = np.zeros((dim, dim)) if hessian else None
        return val, grad, H


def plot_contour_with_path(f: Function, paths: dict[str, list[np.ndarray]]):
    """
    Summary
    -------
    Plot 2D contours of f and overlay multiple optimization paths.

    Parameters
    ----------
    f : ObjectiveFunction
        The objective function to plot.
    paths : dict[str, list[np.ndarray]]
        Dictionary of optimization paths, where keys are labels for naming the optimization paths
        and values are lists of points.
    """
    all_x = np.vstack([np.array(x_path) for x_path in paths.values()])

    # Determine plot limits automatically based on where the paths are located
    xmin, xmax = all_x[:, 0].min() - 1, all_x[:, 0].max() + 1
    ymin, ymax = all_x[:, 1].min() - 1, all_x[:, 1].max() + 1

    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, 400),
        np.linspace(ymin, ymax, 400),
    )
    Z = np.vectorize(lambda x, y: f(np.array([x, y]))[0])(X, Y)

    _, ax = plt.subplots()
    contour = ax.contour(X, Y, Z, levels=50)

    for label, x_path in paths.items():
        xp = np.array(x_path)
        ax.plot(xp[:, 0], xp[:, 1], linestyle='--', marker='o', linewidth=1.5, label=label)

    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_title(f'Contour Plot: {f.__class__.__name__}')
    ax.legend()

    plt.colorbar(contour, ax=ax, label='f(x) contour')

    # plt.savefig(f'contour_with_path_{f.__class__.__name__}.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_objective_history(histories: dict[str, list[float]], f_name: str = ''):
    """
    Summary
    -------
    Plot objective value history over iterations for multiple optimization runs.

    Parameters
    ----------
    histories : dict[str, list[float]]
        Dictionary of objective value histories, where keys are labels for naming the optimization runs
        and values are lists of objective values at each iteration.
    f_name : str, default = ''
        Name of the objective function, used in the plot title.
    """
    _, ax = plt.subplots()

    for label, f_path in histories.items():
        ax.plot(range(len(f_path)), f_path, marker='o', label=label)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Objective value vs. iteration: {f_name}')
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.savefig(f'objective_history_{f_name}.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_feasible_region_and_path(problem: ConstrainedProblem, central_path: list[np.ndarray]):
    """
    Summary
    -------
    Plot the feasible region of a constrained optimization problem and overlay the central path.
    Can handle both 2D and 3D problems.

    Parameters
    ----------
    problem : ConstrainedProblem
        The constrained optimization problem containing the objective function and constraints.
    central_path : list[np.ndarray]
        List of points representing the central path of the optimization algorithm.
    """
    pts = np.vstack(central_path)
    dim = pts.shape[1]
    # Handle 2D plotting
    if dim == 2:
        pts_path = pts
        # Determine plot limits automatically based on where the paths are located
        xmin, xmax = pts_path[:, 0].min() - 1, pts_path[:, 0].max() + 1
        ymin, ymax = pts_path[:, 1].min() - 1, pts_path[:, 1].max() + 1
        
        X, Y = np.meshgrid(
            np.linspace(xmin, xmax, 400),
            np.linspace(ymin, ymax, 400),
        )

        # Compute objective values on grid for contour
        Z = np.vectorize(lambda x, y: problem.FUNC(np.array([x, y]))[0])(X, Y)

        # Identify feasible subset
        grid = np.stack([X.ravel(), Y.ravel()], axis=1)
        mask = np.ones(grid.shape[0], dtype=bool)
        for g in problem.INEQ_CONSTRAINTS:
            vals = np.array([g(pt, False)[0] for pt in grid])
            mask &= (vals <= 0)
        feasible = grid[mask]

        fig, ax = plt.subplots()
        contour = ax.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='f(x) contour')
        ax.scatter(feasible[:, 0], feasible[:, 1], s=1, alpha=0.3, label='feasible region')
        ax.plot(pts_path[:, 0], pts_path[:, 1], 'r--o', label='central path')
        ax.plot(pts_path[-1, 0], pts_path[-1, 1], 'bs', label='solution')
    # Handle 3D plotting
    elif dim == 3:
        # 3D feasible region on plane A x = b
        A, b = problem.EQ_CONSTRAINT_MAT, problem.EQ_CONSTRAINT_RHS
        pts_path = pts
        
        # Sample (x,y) grid and solve for z on the plane A x = b
        xs = np.linspace(pts_path[:, 0].min() - 1, pts_path[:, 0].max() + 1, 100)
        ys = np.linspace(pts_path[:, 1].min() - 1, pts_path[:, 1].max() + 1, 100)
        
        grid = []
        for xi in xs:
            for yi in ys:
                zi = (b[0] - A[0,0] * xi - A[0,1] * yi) / A[0, 2]
                pt = np.array([xi, yi, zi])
                if all(g(pt, False)[0] <= 0 for g in problem.INEQ_CONSTRAINTS):
                    grid.append(pt)
        grid = np.array(grid)

        # Compute objective values for coloring
        fvals = np.array([problem.FUNC(pt, False)[0] for pt in grid])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], c=fvals, alpha=0.3, cmap='viridis')
        plt.colorbar(sc, ax=ax, label='f(x)')

        ax.plot(pts_path[:, 0], pts_path[:, 1], pts_path[:, 2], 'r--o', label='central path')
        ax.scatter(pts_path[-1, 0], pts_path[-1, 1], pts_path[-1, 2], c='b', marker='s', label='solution')
    else:
        raise ValueError('plot_feasible_region_and_path only supports 2D or 3D.')
    
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_title(f'Feasible Region: {problem.__class__.__name__}')
    ax.legend()
    # plt.savefig(f'objective_history_{f.__class__.__class__}.png', bbox_inches='tight', dpi=300)
    plt.show()

