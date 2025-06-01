from typing import Optional

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt



class ObjectiveFunction(ABC):
    """
    Description
    -----------
    Base class for objective functions.
    """

    @classmethod
    @abstractmethod
    def __call__(cls, x: np.ndarray, hessian: bool = False) -> tuple[float, np.ndarray, Optional[np.ndarray]]:
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


def plot_contour_with_path(f: ObjectiveFunction, paths: dict[str, list[np.ndarray]]):
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

    plt.savefig(f'contour_with_path_{f.__class__.__name__}.png', bbox_inches='tight', dpi=300)
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

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.savefig(f'objective_history_{f_name}.png', bbox_inches='tight', dpi=300)
    plt.show()
    
