#!/usr/bin/python
# -*- coding: utf-8 -*-
import pyforms
from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlText
from pyforms.controls import ControlButton
from pyforms.controls import ControlCombo
from pyforms.controls import ControlFile
from pyforms.controls import ControlNumber

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
        # changing variables
        crossover_prob = 0.6
        mutation_prob = 0.2
        num_individuals = 16
        num_generations = 30

        # Definition of the forms fields
        self._crossover_prob = ControlNumber('Choose learning factor value', minimum=0, maximum=1, decimals=3,
                                             step=0.10)
        self._fullname = ControlText('Full name')
        self._activationfunction.add_item('unipolar sigmoid function', 0)
        self._activationfunction.add_item('bipolar sigmoid function', 1)
        # Define the organization of the forms
        self._formset = [{
            'Create': [('_file', '_activationfunction', '_learningfactor')],
            'Solve': []
        }]
        # Use dictionaries for tabs
        # Use the sign '=' for a vertical splitter
        # Use the signs '||' for a horizontal splitter

        # 'Create:'['_file','_activationfunction','_learningfactor'],
        # 'Solve:'['_fullname']
        self.mainmenu = [
            {'File': [
                {'Open': self.__openEvent},
                '-',
                {'Save': self.__saveEvent},
                {'Save as': self.__saveAsEvent}
            ]
            }
        ]

        # Define the button action

    def __openEvent(self):
        pass

    def __saveEvent(self):
        pass

    def __saveAsEvent(self):
        pass


##################################################################################################################
##################################################################################################################
##################################################################################################################

# Execute the application
if __name__ == "__main__":     pyforms.start_app(NeuralNetwork)
