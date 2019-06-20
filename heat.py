from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr
import os

def show_function_plot(x_domain, y_domain, function_to_print, population):
    # # Make data.
    x = np.arange(x_domain[0], x_domain[1], 0.05)
    y = np.arange(y_domain[0], y_domain[1], 0.05)
    x, y = np.meshgrid(x, y)
    expr = parse_expr(function_to_print)
    f = lambdify(symbols("x y"), expr, 'numpy')
    z = f(x,y)

    # Plot the heatmap.
    plt.contourf(x, y, z.T, levels=200)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(population[:, 0], population[:, 1])
    plt.show()



if __name__ == "__main__":
    show_function_plot(x_domain=[-2, 2],y_domain=[-2,2],function_to_print="x**2+y**2", population=np.random.uniform(-2, 2 ** 0.05, size=(20, 2)))
