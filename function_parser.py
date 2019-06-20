from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import parse_expr
import numpy as np


def fitness_function(function: str, population: np.ndarray) -> np.ndarray:
    x, y = symbols("x y")
    expr = parse_expr(function)
    f = lambdify((x, y), expr, 'numpy')
    return f(population[:, 0], population[:, 1])
