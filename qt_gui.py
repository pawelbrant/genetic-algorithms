import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLineEdit, QFormLayout, QWidget, \
    QTabWidget, QHBoxLayout, QSpinBox, QLabel, QDoubleSpinBox, QTextEdit, QRadioButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr
from mpl_toolkits.mplot3d import Axes3D

# GA
import genetic_algorithm
import function_parser as fp


class Window(QTabWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # Windows
        self.create_tab = QWidget()
        self.solve_tab = QWidget()
        self.plot_tab = QWidget()
        self.summary_tab = QWidget()

        self.addTab(self.create_tab, "Create")
        self.addTab(self.solve_tab, "Solve")
        self.addTab(self.plot_tab, "Generations plots")
        self.addTab(self.summary_tab, "Summary")

        # Create Window
        self.x_start = QDoubleSpinBox()
        self.x_start.setRange(-1000000, 1000000)
        self.x_start.setValue(-2)
        self.x_end = QDoubleSpinBox()
        self.x_end.setRange(-1000000, 1000000)
        self.x_end.setValue(2)
        self.y_start = QDoubleSpinBox()
        self.y_start.setRange(-1000000, 1000000)
        self.y_start.setValue(-2)
        self.y_end = QDoubleSpinBox()
        self.y_end.setRange(-1000000, 1000000)
        self.y_end.setValue(2)
        self.function = QLineEdit()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Solve Window
        self.crossover_prob = QDoubleSpinBox()
        self.crossover_prob.setMinimum(0)
        self.crossover_prob.setMaximum(1)
        self.crossover_prob.setSingleStep(0.1)
        self.crossover_prob.setValue(0.6)
        self.mutation_prob = QDoubleSpinBox()
        self.mutation_prob.setMinimum(0)
        self.mutation_prob.setMaximum(1)
        self.mutation_prob.setSingleStep(0.1)
        self.mutation_prob.setValue(0.1)
        self.num_ind = QSpinBox()
        self.num_ind.setRange(2, 100)
        self.num_ind.setSingleStep(2)
        self.num_ind.setValue(10)
        self.num_gen = QSpinBox()
        self.num_gen.setRange(1, 100)
        self.num_gen.setValue(5)
        self.command_line = QTextEdit()
        self.command_line.setMinimumHeight(475)
        self.command_line.isReadOnly()
        self.generation_figure = plt.figure()
        self.generation_canvas = FigureCanvas(self.generation_figure)
        self.population_list = []
        self.current_gen = None
        self.min = QRadioButton("Min")
        self.min.toggled.connect(lambda: self.select_option(self.min))
        self.max = QRadioButton("Max")
        self.max.setChecked(True)
        self.max.toggled.connect(lambda: self.select_option(self.max))
        self.find_max = 1

        # Summary Window
        self.best_x = ""
        self.best_y = ""
        self.best_solution = float("nan")
        self.summary_figure = plt.figure()
        self.summary_canvas = FigureCanvas(self.summary_figure)
        self.mean = []
        self.median = []
        self.best = []
        self.best_x_label = QLabel("Best solution x: " + str(self.best_x))
        self.best_y_label = QLabel("y: " + str(self.best_y))
        self.best_f_label = QLabel("f(x,y): " + str(self.best_solution))

        # Initialize app windows
        self.create_tab_ui()
        self.solve_tab_ui()
        self.plot_tab_ui()
        self.summary_tab_ui()
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
        function_plot = QHBoxLayout()
        function_plot.addWidget(QLabel(""))
        function_plot.addWidget(self.canvas)
        function_plot.addWidget(QLabel(""))
        data_form.addRow(function_plot)
        data_form.addRow(button)

        self.create_tab.setLayout(data_form)

    def select_option(self, b):

        if b.text() == "Max":
            if b.isChecked() == True:
                self.find_max = 1
            else:
                self.find_max = 0

        if b.text() == "Min":
            if b.isChecked() == True:
                self.find_max = 0
            else:
                self.find_max = 1

    def solve_tab_ui(self):

        ga_button = QPushButton('Find solution')
        ga_button.clicked.connect(self.solve)
        GA_form = QFormLayout()
        GA_data = QHBoxLayout()
        GA_data.addWidget(QLabel("Crossover probability: "))
        GA_data.addWidget(self.crossover_prob)
        GA_data.addWidget(QLabel("Mutation probability: "))
        GA_data.addWidget(self.mutation_prob)
        GA_data.addWidget(QLabel("Number of individuals: "))
        GA_data.addWidget(self.num_ind)
        GA_data.addWidget(QLabel("Number of generations: "))
        GA_data.addWidget(self.num_gen)
        GA_form.addRow(GA_data)
        GA_form.addRow(self.command_line)
        GA_options = QHBoxLayout()
        GA_options.addWidget(ga_button)
        GA_options.addWidget(self.max)
        GA_options.addWidget(self.min)
        GA_form.addRow(GA_options)

        self.solve_tab.setLayout(GA_form)

    def plot_tab_ui(self):
        previous_button = QPushButton('Previous generation')
        previous_button.clicked.connect(self.previous_plot)
        next_button = QPushButton('Next generation')
        next_button.clicked.connect(self.next_plot)
        GA_generations_plots = QVBoxLayout()
        ga_toolbar = NavigationToolbar(self.generation_canvas, self)
        GA_generations_plots.addWidget(ga_toolbar)
        GA_generations_plots.addWidget(previous_button)
        GA_generations_plots.addWidget(next_button)
        GA_generations_plots.addWidget(self.generation_canvas)

        self.plot_tab.setLayout(GA_generations_plots)

    def summary_tab_ui(self):
        summary_button = QPushButton('Show summary')
        summary_button.clicked.connect(self.summary)
        GA_summary = QFormLayout()
        GA_summary_data = QHBoxLayout()

        GA_summary_data.addWidget(self.best_x_label)
        GA_summary_data.addWidget(self.best_y_label)
        GA_summary_data.addWidget(self.best_f_label)
        ga_toolbar = NavigationToolbar(self.summary_canvas, self)
        GA_summary.addRow(GA_summary_data)
        GA_summary.addRow(ga_toolbar)
        GA_summary.addRow(self.summary_canvas)
        GA_summary.addRow(summary_button)
        self.summary_tab.setLayout(GA_summary)

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

    def solve(self, option):
        g = genetic_algorithm.GA(
            gui_x_domain=[self.x_start.value(), self.x_end.value()],
            gui_y_domain=[self.y_start.value(), self.y_end.value()],
            gui_crossover_prob=self.crossover_prob.value(),
            gui_mutation_prob=self.mutation_prob.value(),
            gui_num_ind=self.num_ind.value(),
            gui_num_gen=self.num_gen.value()
        )
        self.population_list = []
        self.command_line.clear()
        x = np.linspace(self.x_start.value(), self.x_end.value(), 200)
        y = np.linspace(self.y_start.value(), self.y_end.value(), 200)
        x, y = np.meshgrid(x, y)
        for generation_index in range(g.num_generations):
            new = "**********************************\n"
            new = new + 'Generation ' + str(generation_index + 1) + ';\n' + ' Population:\n '
            new = new + str(g.pop_float) + ';\n' + ' Function value: \n'
            fitness = fp.fitness_function(self.function.text(), g.pop_float)
            new = new + str(fitness) + '\n'
            self.command_line.append(new)
            for fit_index, fit in enumerate(fitness):
                if fit == max(fitness):
                    self.best_x = g.pop_float[fit_index, 0]
                    self.best_y = g.pop_float[fit_index, 1]
                    self.best_solution = fit
            if self.find_max == 1:
                g.best_solution_in_each_generation.append(np.max(fitness))
            else:
                g.best_solution_in_each_generation.append(np.min(fitness))
            g.mean_solution_in_each_generation.append(np.mean(fitness))
            g.median_solution_in_each_generation.append(np.median(fitness))
            self.population_list.append(g.pop_float)
            if generation_index != g.num_generations:
                if self.find_max == 0:
                    negative_fitness = [i * -1 for i in fitness]
                    parents = g.select_parents(negative_fitness)
                else:
                    parents = g.select_parents(fitness)
                g.pop = g.crossover(parents)
                g.pop = g.mutation()
                g.pop = g.bin2int()
                g.pop_float = g.relaxation_function()
        self.current_gen = generation_index
        self.generation_figure.clear()
        ax = self.generation_figure.add_subplot(111)
        # plot data
        ax.plot(g.pop_float[:, 0], g.pop_float[:, 1], 'bo', alpha=0.7)
        expr = parse_expr(self.function.text())
        f = lambdify(symbols("x y"), expr, 'numpy')
        z = f(x, y)

        # Plot the heatmap.
        cmap = cm.viridis
        ax.contourf(x, y, z, levels=200, cmap=cmap)
        mappable = ax.collections[0]
        norm = colors.Normalize(vmin=np.min(z), vmax=np.max(z))
        self.generation_figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

        # ax.scatter(g.pop_float[:, 0], g.pop_float[:, 1])
        ax.set_title("Wykres f(x,y). Pokolenie: " + str(self.current_gen + 1))
        ax.set_xlim(left=self.x_start.value(), right=self.x_end.value())
        ax.set_ylim(bottom=self.y_start.value(), top=self.y_end.value())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # ax.legend()
        ax.grid()
        # refresh canvas
        self.generation_canvas.draw()
        self.command_line.update()
        self.mean = g.mean_solution_in_each_generation
        self.median = g.median_solution_in_each_generation
        self.best = g.best_solution_in_each_generation

    def previous_plot(self):
        if self.current_gen is not None and self.current_gen > 0:
            self.current_gen = self.current_gen - 1
            self.generation_figure.clear()
            ax = self.generation_figure.add_subplot(111)
            # plot data
            ax.plot(self.population_list[self.current_gen][:, 0], self.population_list[self.current_gen][:, 1], 'bo', alpha=0.7)
            x = np.linspace(self.x_start.value(), self.x_end.value(), 200)
            y = np.linspace(self.y_start.value(), self.y_end.value(), 200)
            x, y = np.meshgrid(x, y)
            expr = parse_expr(self.function.text())
            f = lambdify(symbols("x y"), expr, 'numpy')
            z = f(x, y)

            # Plot the heatmap.
            cmap = cm.viridis
            ax.contourf(x, y, z, levels=200, cmap=cmap)
            mappable = ax.collections[0]
            norm = colors.Normalize(vmin=np.min(z), vmax=np.max(z))
            self.generation_figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

            # ax.scatter(g.pop_float[:, 0], g.pop_float[:, 1])
            ax.set_title("Wykres f(x,y). Pokolenie: " + str(self.current_gen + 1))
            ax.set_xlim(left=self.x_start.value(), right=self.x_end.value())
            ax.set_ylim(bottom=self.y_start.value(), top=self.y_end.value())
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            # ax.legend()
            ax.grid()
            # refresh canvas
            self.generation_canvas.draw()

    def next_plot(self):
        if self.current_gen is not None and self.current_gen < self.num_gen.value() - 1:
            self.current_gen = self.current_gen + 1
            self.generation_figure.clear()
            ax = self.generation_figure.add_subplot(111)
            # plot data
            ax.plot(self.population_list[self.current_gen][:, 0], self.population_list[self.current_gen][:, 1], 'bo', alpha=0.7)
            x = np.linspace(self.x_start.value(), self.x_end.value(), 200)
            y = np.linspace(self.y_start.value(), self.y_end.value(), 200)
            x, y = np.meshgrid(x, y)
            expr = parse_expr(self.function.text())
            f = lambdify(symbols("x y"), expr, 'numpy')
            z = f(x, y)

            # Plot the heatmap.
            cmap = cm.viridis
            ax.contourf(x, y, z, levels=200, cmap=cmap)
            mappable = ax.collections[0]
            norm = colors.Normalize(vmin=np.min(z), vmax=np.max(z))
            self.generation_figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

            # ax.scatter(g.pop_float[:, 0], g.pop_float[:, 1])
            ax.set_title("Wykres f(x,y). Pokolenie: " + str(self.current_gen + 1))
            ax.set_xlim(left=self.x_start.value(), right=self.x_end.value())
            ax.set_ylim(bottom=self.y_start.value(), top=self.y_end.value())
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            # ax.legend()
            ax.grid()
            # refresh canvas
            self.generation_canvas.draw()

    def summary(self):
        self.summary_figure.clear()
        ax = self.summary_figure.add_subplot(111)
        # plot data
        p1 = ax.plot(range(1, self.num_gen.value() + 1), self.best, color="cyan", linestyle='-',
                     marker="x", alpha=0.5, label="Best")
        p2 = ax.plot(range(1, self.num_gen.value() + 1), self.mean, color="aquamarine",
                     linestyle='-', marker="s", alpha=0.5, label="Mean")
        p3 = ax.plot(range(1, self.num_gen.value() + 1), self.median, color="teal",
                     linestyle='-', marker="p", alpha=0.5, label="Median")
        ax.set_title("Wykres wartości osiągniętych w każdej iteracji")
        ax.set_xlabel("Generacja")
        ax.set_ylabel("Wartość funkcji dopasowania")
        ax.legend()
        ax.grid()
        # refresh canvas
        self.summary_canvas.draw()
        self.best_x_label.setText("Best solution x: " + str(self.best_x))
        self.best_y_label.setText("y: " + str(self.best_y))
        self.best_f_label.setText("f(x,y): " + str(self.best_solution))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
