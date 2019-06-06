from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def show_function_plot(x_domain, y_domain, function_to_print):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    x = np.arange(x_domain[0], x_domain[1], 0.1)
    y = np.arange(y_domain[0], y_domain[1], 0.1)
    x, y = np.meshgrid(x, y)
    z = eval(function_to_print)

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.0, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F(X,Y)')
    plt.show()
