#!/usr/bin/python
# -*- coding: utf-8 -*-
import pyforms
import main
import function_parser as fp
import function_plot as fplot
from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlText
from pyforms.controls import ControlButton
from pyforms.controls import ControlTextArea
from pyforms.controls import ControlCombo
from pyforms.controls import ControlMatplotlib
from pyforms.controls import ControlNumber
import os

__author__ = "Ricardo Ribeiro"
__credits__ = ["Ricardo Ribeiro"]
__license__ = "MIT"
__version__ = "0.0"
__maintainer__ = "Ricardo Ribeiro"
__email__ = "ricardojvr@gmail.com"
__status__ = "Development"


class GeneticAlgorithms(BaseWidget):

    def __init__(self):
        super(GeneticAlgorithms, self).__init__('GeneticAlgorithms')

        # Definition of the forms fields
        self._crossover_prob = ControlNumber('Choose crossover probability', minimum=0, maximum=1, decimals=3, step=0.10)
        self._mutation_prob = ControlNumber("Choose mutation probability", minimum=0, maximum=1, decimals=3, step=0.10)
        self._num_individuals = ControlNumber('Choose number of individuals', minimum=2, maximum=100, step=1)
        self._num_generations = ControlNumber('Choose number of generations', minimum=1, maximum=100, step=1)
        self._function = ControlText('Function f(x,y)=')
        self._x_start = ControlNumber('X start', decimals=3, minimum=-1000, maximum=1000, step=0.1)
        self._x_end = ControlNumber('X end', decimals=3, minimum=-1000, maximum=1000, step=0.1)
        self._y_start = ControlNumber('Y start', decimals=3, minimum=-1000, maximum=1000, step=0.1)
        self._y_end = ControlNumber('Y end', decimals=3, step=0.1)
        self._loaddatabutton = ControlButton('Load data')
        self._showfunctionbutton = ControlButton('Show function')
        self._showfunctionimage = ControlMatplotlib()
        self._terminal = ControlTextArea()
        # Define the button action
        self._loaddatabutton.value = self.__buttonAction
        self._showfunctionbutton.value = self.__function_plotAction
        # Define the organization of the forms
        self._formset = [{
            'Create': [('_function', '_x_start', '_x_end', '_y_start', '_y_end'),
                       ('_crossover_prob', '_mutation_prob', '_num_individuals', '_num_generations'),
                       ('_loaddatabutton'),
                       ('_terminal')
                       ],
            'Show function': [('_showfunctionbutton'),
                              ('_showfunctionimage')],
            'Solve': []
        }]

    def __function_plotAction(self):
        try:
            fig = fplot.show_function_plot(function_to_print=self._function.value,
                                     x_domain=[self._x_start.value, self._x_end.value],
                                     y_domain=[self._y_start.value, self._y_end.value])

            self._showfunctionimage.fig = fig
            self._showfunctionimage.value = fig
            self._showfunctionimage.draw()
        except Exception as e:
            print(e)

    def __buttonAction(self):
        """Button action event"""
        try:
            g = main.GA(
                gui_x_domain=[self._x_start.value, self._x_end.value],
                gui_y_domain=[self._y_start.value, self._y_end.value]
            )

            self._terminal.value = fp.fitness_function(self._function.value, g.pop_float)
        except Exception as e:
            self._terminal.value = e


##################################################################################################################
##################################################################################################################
##################################################################################################################

# Execute the application
if __name__ == "__main__":     pyforms.start_app(GeneticAlgorithms)

# Use dictionaries for tabs
        # Use the sign '=' for a vertical splitter
        # Use the signs '||' for a horizontal splitter

        # 'Create:'['_file','_activationfunction','_learningfactor'],
        # 'Solve:'['_fullname']
        # self.mainmenu = [
        #     {'File': [
        #         {'Open': self.__openEvent},
        #         '-',
        #         {'Save': self.__saveEvent},
        #         {'Save as': self.__saveAsEvent}
        #     ]
        #     }
        # ]