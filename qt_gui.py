import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLineEdit, QFormLayout, QWidget, \
    QTabWidget, QHBoxLayout, QSpinBox, QLabel, QDoubleSpinBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr
from mpl_toolkits.mplot3d import Axes3D


class Window(QTabWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.x_start = QDoubleSpinBox()
        self.x_end = QDoubleSpinBox()
        self.y_start = QDoubleSpinBox()
        self.y_end = QDoubleSpinBox()
        self.function = QLineEdit()
        self.create_tab = QWidget()
        self.solve_tab = QWidget()
        self.plot_tab = QWidget()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.addTab(self.create_tab, "Create")
        self.addTab(self.solve_tab, "Solve")
        self.addTab(self.plot_tab, "Generations plots")

        self.create_tab_ui()
        self.solve_tab_ui()
        self.plot_tab_ui()
        self.setWindowTitle("Genetic Algorithms")

    def create_tab_ui(self):

        button = QPushButton('Show function')
        toolbar = NavigationToolbar(self.canvas, self)
        button.clicked.connect(self.plot)
        data_form = QFormLayout()
        function_data = QHBoxLayout()
        function_data.addWidget(self.function)
        function_data.addWidget(QLabel("X=["))
        function_data.addWidget(self.x_start)
        function_data.addWidget(QLabel(","))
        function_data.addWidget(self.x_end)
        function_data.addWidget(QLabel("]"))
        function_data.addWidget(QLabel("  Y=["))
        function_data.addWidget(self.y_start)
        function_data.addWidget(QLabel(","))
        function_data.addWidget(self.y_end)
        function_data.addWidget(QLabel("]"))
        data_form.addRow("f(x,y) = ", function_data)
        data_form.addRow(toolbar)
        data_form.addRow(self.canvas)
        data_form.addRow(button)

        self.create_tab.setLayout(data_form)

    def solve_tab_ui(self):

        pass

    def plot_tab_ui(self):
        pass

    def plot(self):

        # instead of ax.hold(False)
        self.figure.clear()

        ax = self.figure.gca(projection='3d')

        # Make data.
        x = np.arange(self.x_start.value(), self.x_end.value(), 0.1)
        y = np.arange(self.y_start.value(), self.y_end.value(), 0.1)
        x, y = np.meshgrid(x, y)
        expr = parse_expr(self.function.text())
        f = lambdify(symbols("x y"), expr, 'numpy')
        z = f(x, y)

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.0, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        self.figure.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('F(X,Y)')

        # refresh canvas
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())