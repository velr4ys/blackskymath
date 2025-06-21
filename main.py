import math
import re
import sys
import time
from typing import Union, List, Tuple, Dict, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
from sympy import (symbols, Eq, solve, simplify, diff, integrate, limit, 
                  sympify, latex, series, Matrix, lcm, gcd, factorint,
                  isprime, primerange, nextprime, prevprime, sqrt, N, sin, cos)
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, 
                                       implicit_multiplication_application)
from sympy.plotting import plot as sympy_plot
import statistics
from scipy.optimize import fsolve, minimize
from collections import defaultdict
import random
from fractions import Fraction
from datetime import datetime, timedelta
import calendar
import pytz
from math import comb, perm
from scipy import stats
import mpmath
import json
from dataclasses import dataclass
from enum import Enum, auto

transformations = standard_transformations + (implicit_multiplication_application,)

class CalculatorMode(Enum):
    SCIENTIFIC = auto()
    GRAPHING = auto()
    MATRIX = auto()
    STATISTICAL = auto()
    FINANCIAL = auto()
    PROGRAMMING = auto()
    UNIT_CONVERTER = auto()
    DATE_TIME = auto()
    GAME_THEORY = auto()
    CHEMISTRY = auto()
    PHYSICS = auto()
    ENGINEERING = auto()

@dataclass
class FinancialResult:
    present_value: float
    future_value: float
    payment: float
    periods: int
    rate: float
    type: str  # 'loan' or 'investment'

class BlackskyMath:
    def __init__(self):
        self.variables = {}
        self.history = []
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': math.inf,
            'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
            'c': 299792458,  # Speed of light (m/s)
            'G': 6.67430e-11,  # Gravitational constant
            'h': 6.62607015e-34,  # Planck constant
            'k': 1.380649e-23,  # Boltzmann constant
            'R': 8.314462618,  # Gas constant
            'Na': 6.02214076e23,  # Avogadro's number
            'g': 9.80665,  # Standard gravity
            'eps0': 8.8541878128e-12,  # Vacuum permittivity
            'mu0': 1.25663706212e-6,  # Vacuum permeability
        }
        self.functions = {
            # Basic math
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'asinh': math.asinh,
            'acosh': math.acosh,
            'atanh': math.atanh,
            'log': math.log10,
            'ln': math.log,
            'log2': math.log2,
            'sqrt': math.sqrt,
            'cbrt': lambda x: x**(1/3),
            'abs': abs,
            'round': round,
            'ceil': math.ceil,
            'floor': math.floor,
            'fact': math.factorial,
            'gamma': math.gamma,
            'gcd': math.gcd,
            'lcm': lambda a, b: abs(a*b) // math.gcd(a, b) if a and b else 0,
            'rad': math.radians,
            'deg': math.degrees,
            'sum': sum,
            'avg': lambda *args: sum(args)/len(args),
            'max': max,
            'min': min,
            'mod': math.fmod,
            'hypot': math.hypot,
            'erf': math.erf,
            'erfc': math.erfc,
            
            # Statistical
            'mean': statistics.mean,
            'median': statistics.median,
            'mode': statistics.mode,
            'stdev': statistics.stdev,
            'variance': statistics.variance,
            'quantile': self._quantile,
            'corr': lambda x, y: np.corrcoef(x, y)[0, 1],
            
            # Combinatorics
            'comb': comb,
            'perm': perm,
            'binom': lambda n, k: comb(n, k),
            
            # Random
            'rand': random.random,
            'randint': random.randint,
            'choice': random.choice,
            
            # Financial
            'fv': self._future_value,
            'pv': self._present_value,
            'pmt': self._payment,
            'nper': self._num_periods,
            'rate': self._interest_rate,
            'irr': self._irr,
            'npv': self._npv,
            
            # Engineering
            'db': lambda x: 10 * math.log10(x),
            'idb': lambda x: 10 ** (x / 10),
            'rms': lambda x: math.sqrt(sum(i**2 for i in x)/len(x)),
            'snr': lambda s, n: 20 * math.log10(math.sqrt(sum(si**2 for si in s)/sum(ni**2 for ni in n))) if n else 0,
            
            # Special
            'fib': self._fibonacci,
            'prime': isprime,
            'nextprime': nextprime,
            'prevprime': prevprime,
            # 'blackscholes': self._black_scholes,  # Commented out or implement this function if needed
        }
        self.commands = {
            # Basic commands
            'help': self.show_help,
            'exit': self.exit_cli,
            'quit': self.exit_cli,
            'clear': self.clear_screen,
            'history': self.show_history,
            'vars': self.show_variables,
            'del': self.delete_variable,
            'save': self.save_session,
            'load': self.load_session,
            'menu': self.show_calculator_menu,
            'mode': self.switch_mode,
            
            # Mathematical operations
            'solve': self.solve_equation,
            'plot': self.plot_function,
            'derive': self.derive_function,
            'integrate': self.integrate_function,
            'limit': self.calculate_limit,
            'series': self.calculate_series,
            'factor': self.factor_expression,
            'expand': self.expand_expression,
            
            # Numerical methods
            'root': self.find_root,
            'minimize': self.find_minimum,
            'maximize': self.find_maximum,
            
            # Statistics
            'stats': self.calculate_stats,
            'regress': self.linear_regression,
            'dist': self.probability_distribution,
            'hist': self.create_histogram,
            
            # Matrix operations
            'matrix': self.matrix_operations,
            'det': self.matrix_determinant,
            'inv': self.matrix_inverse,
            'eigen': self.matrix_eigen,
            
            # Unit conversion
            'convert': self.unit_conversion,
            
            # Number theory
            'prime': self.prime_operations,
            'gcd': self.gcd_operation,
            'lcm': self.lcm_operation,
            'factorint': self.factor_integer,
            
            # Calculus
            'taylor': self.taylor_series,
            'diffeq': self.solve_ode,
            
            # Financial
            'finance': self.financial_calculations,
            'amort': self.amortization_schedule,
            'roi': self.calculate_roi,
            
            # Date/time
            'date': self.date_calculations,
            'time': self.time_conversion,
            'countdown': self.countdown_timer,
            
            # System
            'system': self.system_info,
            'bench': self.benchmark,
            
            # Programming
            'eval': self.evaluate_code,
            'lambda': self.create_lambda,
            
            # Graph theory
            'graph': self.graph_operations,
            
            # New features
            'game': self.game_theory,
            'chem': self.chemistry_calculator,
            'physics': self.physics_calculator,
            'eng': self.engineering_calculator,
            'puzzle': self.math_puzzles,
            'quiz': self.math_quiz,
        }
        self.aliases = {
            '?': 'help',
            'q': 'quit',
            'h': 'history',
            'v': 'vars',
            'd': 'del',
            's': 'solve',
            'p': 'plot',
            'dx': 'derive',
            'int': 'integrate',
            'lim': 'limit',
            'stat': 'stats',
            'mat': 'matrix',
            'conv': 'convert',
            'pr': 'prime',
            'fin': 'finance',
            'am': 'amort',
        }
        self.units = {
            'length': {
                'm': 1,
                'cm': 0.01,
                'mm': 0.001,
                'km': 1000,
                'in': 0.0254,
                'ft': 0.3048,
                'yd': 0.9144,
                'mi': 1609.34,
                'nmi': 1852,
                'au': 1.496e11,
                'ly': 9.461e15,
                'pc': 3.086e16
            },
            'mass': {
                'kg': 1,
                'g': 0.001,
                'mg': 0.000001,
                'lb': 0.453592,
                'oz': 0.0283495,
                'ton': 907.185,
                'tonne': 1000,
                'st': 6.35029,
                'carat': 0.0002
            },
            'time': {
                's': 1,
                'ms': 0.001,
                'us': 1e-6,
                'ns': 1e-9,
                'min': 60,
                'h': 3600,
                'day': 86400,
                'week': 604800,
                'year': 31536000
            },
            'temperature': ['c', 'f', 'k'],
            'volume': {
                'l': 1,
                'ml': 0.001,
                'm3': 1000,
                'cm3': 0.001,
                'gal': 3.78541,
                'qt': 0.946353,
                'pt': 0.473176,
                'cup': 0.236588,
                'floz': 0.0295735,
                'tbsp': 0.0147868,
                'tsp': 0.00492892
            },
            'pressure': {
                'pa': 1,
                'kpa': 1000,
                'mpa': 1e6,
                'bar': 1e5,
                'atm': 101325,
                'mmhg': 133.322,
                'psi': 6894.76,
                'torr': 133.322
            },
            'energy': {
                'j': 1,
                'kj': 1000,
                'cal': 4.184,
                'kcal': 4184,
                'wh': 3600,
                'kwh': 3.6e6,
                'ev': 1.60218e-19,
                'btu': 1055.06
            },
            'power': {
                'w': 1,
                'kw': 1000,
                'mw': 1e6,
                'hp': 745.7,
                'btu/h': 0.293071
            },
            'speed': {
                'm/s': 1,
                'km/h': 0.277778,
                'mph': 0.44704,
                'knot': 0.514444,
                'fps': 0.3048,
                'c': 299792458
            },
            'angle': {
                'rad': 1,
                'deg': math.pi/180,
                'grad': math.pi/200,
                'arcmin': math.pi/10800,
                'arcsec': math.pi/648000
            }
        }
        self.distributions = {
            'normal': stats.norm,
            'uniform': stats.uniform,
            'binom': stats.binom,
            'poisson': stats.poisson,
            'expon': stats.expon,
            'gamma': stats.gamma,
            'beta': stats.beta,
            'chi2': stats.chi2,
            't': stats.t,
            'f': stats.f,
            'lognorm': stats.lognorm,
            'weibull': stats.weibull_min
        }
        self.calculator_modes = {
            '1': 'Scientific Calculator',
            '2': 'Graphing Calculator',
            '3': 'Matrix Calculator',
            '4': 'Statistical Calculator',
            '5': 'Financial Calculator',
            '6': 'Programming Calculator',
            '7': 'Unit Converter',
            '8': 'Date/Time Calculator',
            '9': 'Game Theory Calculator',
            '10': 'Chemistry Calculator',
            '11': 'Physics Calculator',
            '12': 'Engineering Calculator',
            '13': 'Back to Main'
        }
        self.graph_types = {
            'directed': defaultdict(list),
            'undirected': defaultdict(list),
            'weighted': defaultdict(dict)
        }
        self.current_mode = CalculatorMode.SCIENTIFIC
        self.financial_cache = {}
        self.chemical_elements = self._load_chemical_elements()
        self.physical_constants = self._load_physical_constants()
        self.engineering_tools = self._load_engineering_tools()
        self.game_theory_games = self._load_game_theory_games()
        self.math_puzzles_list = self._load_math_puzzles()
        self.quiz_active = False

    def _load_chemical_elements(self) -> Dict:
        """Load chemical elements data."""
        return {
            'H': {'name': 'Hydrogen', 'atomic_number': 1, 'mass': 1.008},
            'He': {'name': 'Helium', 'atomic_number': 2, 'mass': 4.0026},
            'Li': {'name': 'Lithium', 'atomic_number': 3, 'mass': 6.94},
            'Be': {'name': 'Beryllium', 'atomic_number': 4, 'mass': 9.0122},
            'B': {'name': 'Boron', 'atomic_number': 5, 'mass': 10.81},
            'C': {'name': 'Carbon', 'atomic_number': 6, 'mass': 12.011},
            'N': {'name': 'Nitrogen', 'atomic_number': 7, 'mass': 14.007},
            'O': {'name': 'Oxygen', 'atomic_number': 8, 'mass': 15.999},
            'F': {'name': 'Fluorine', 'atomic_number': 9, 'mass': 18.998},
            'Ne': {'name': 'Neon', 'atomic_number': 10, 'mass': 20.180},
            'Na': {'name': 'Sodium', 'atomic_number': 11, 'mass': 22.990},
            'Mg': {'name': 'Magnesium', 'atomic_number': 12, 'mass': 24.305},
            'Al': {'name': 'Aluminum', 'atomic_number': 13, 'mass': 26.982},
            'Si': {'name': 'Silicon', 'atomic_number': 14, 'mass': 28.085},
            'P': {'name': 'Phosphorus', 'atomic_number': 15, 'mass': 30.974},
            'S': {'name': 'Sulfur', 'atomic_number': 16, 'mass': 32.06},
            'Cl': {'name': 'Chlorine', 'atomic_number': 17, 'mass': 35.45},
            'Ar': {'name': 'Argon', 'atomic_number': 18, 'mass': 39.948},
            'K': {'name': 'Potassium', 'atomic_number': 19, 'mass': 39.098},
            'Ca': {'name': 'Calcium', 'atomic_number': 20, 'mass': 40.078}
        }

    def _load_physical_constants(self) -> Dict:
        """Load physical constants data."""
        return {
            'gravitational_acceleration': 9.80665,  # m/s²
            'speed_of_light': 299792458,  # m/s
            'planck_constant': 6.62607015e-34,  # J·s
            'electron_charge': 1.602176634e-19,  # C
            'avogadro_number': 6.02214076e23,  # mol⁻¹
            'boltzmann_constant': 1.380649e-23,  # J/K
            'gas_constant': 8.314462618,  # J/(mol·K)
            'vacuum_permittivity': 8.8541878128e-12,  # F/m
            'vacuum_permeability': 1.25663706212e-6,  # N/A²
            'gravitational_constant': 6.67430e-11  # m³/(kg·s²)
        }

    def _load_engineering_tools(self) -> Dict:
        """Load engineering tools and formulas."""
        return {
            'beam_deflection': {
                'formula': "δ = (P*L³)/(48*E*I)",
                'description': "Maximum deflection of a simply supported beam with central load"
            },
            'stress_strain': {
                'formula': "σ = E*ε",
                'description': "Hooke's Law relating stress and strain"
            },
            'fluid_flow': {
                'formula': "Q = A*v",
                'description': "Volumetric flow rate through a pipe"
            }
        }

    def _load_game_theory_games(self) -> Dict:
        """Load game theory example games."""
        return {
            'prisoners_dilemma': {
                'description': "Classic non-zero-sum game",
                'payoffs': {
                    ('cooperate', 'cooperate'): (-1, -1),
                    ('cooperate', 'defect'): (-3, 0),
                    ('defect', 'cooperate'): (0, -3),
                    ('defect', 'defect'): (-2, -2)
                }
            },
            'battle_of_sexes': {
                'description': "Coordination game with different preferences",
                'payoffs': {
                    ('opera', 'opera'): (3, 2),
                    ('opera', 'football'): (0, 0),
                    ('football', 'opera'): (0, 0),
                    ('football', 'football'): (2, 3)
                }
            }
        }

    def _load_math_puzzles(self) -> List[Dict]:
        """Load math puzzles and brain teasers."""
        return [
            {
                'question': "What is the smallest number that is evenly divisible by all numbers from 1 to 20?",
                'answer': "232792560",
                'hint': "Think about least common multiple (LCM)"
            },
            {
                'question': "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                'answer': "5",
                'hint': "Consider the rate of production per machine"
            },
            {
                'question': "What is the next number in this sequence: 1, 11, 21, 1211, 111221, ...?",
                'answer': "312211",
                'hint': "Look at the sequence and describe each number verbally"
            }
        ]

    def _quantile(self, data: List[float], q: float) -> float:
        """Calculate quantile of a dataset."""
        if not data:
            raise ValueError("Data cannot be empty")
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1")
            
        sorted_data = sorted(data)
        n = len(sorted_data)
        index = q * (n - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = int(index)
            upper = lower + 1
            weight = index - lower
            return (1 - weight) * sorted_data[lower] + weight * sorted_data[upper]

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n < 0:
            raise ValueError("Fibonacci sequence is only defined for non-negative integers")
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    def evaluate_expression(self, expr: str):
        """Evaluate a mathematical expression."""
        try:
            # Replace constants
            for const, value in self.constants.items():
                expr = expr.replace(const, str(value))
            
            # Replace variables
            for var, value in self.variables.items():
                expr = expr.replace(var, str(value))
            
            # Check for function calls
            for func in self.functions:
                if func + '(' in expr:
                    # Handle function calls
                    pattern = re.compile(rf"{func}\(([^)]*)\)")
                    matches = pattern.findall(expr)
                    for args_str in matches:
                        args = [self.evaluate_expression(arg.strip()) for arg in args_str.split(',')]
                        result = self.functions[func](*args)
                        expr = expr.replace(f"{func}({args_str})", str(result))
            
            # Evaluate remaining expression
            if any(op in expr for op in ['+', '-', '*', '/', '^', '**']):
                # Use sympy for more complex expressions
                expr = parse_expr(expr, transformations=transformations)
                result = N(expr)
            else:
                # Simple evaluation
                result = float(expr) if '.' in expr else int(expr)
            
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")

    def show_help(self, args: str = ""):
        """Show help information."""
        if not args:
            print("\nBlacksky Math Help:")
            print("="*60)
            print("Basic Usage:")
            print("- Enter mathematical expressions to evaluate them")
            print("- Assign variables with = (e.g., x = 5)")
            print("- Use commands for advanced functionality")
            print("\nAvailable Commands:")
            for cmd in sorted(self.commands.keys()):
                print(f"- {cmd}")
            print("\nType 'help <command>' for more information about a command")
            print("="*60)
        else:
            cmd = args.strip().lower()
            if cmd in self.commands:
                print(f"\nHelp for '{cmd}':")
                print("="*60)
                if cmd == 'solve':
                    print("Solve equations symbolically")
                    print("Usage: solve <equation>")
                    print("Example: solve x^2 - 4 = 0")
                elif cmd == 'plot':
                    print("Plot functions")
                    print("Usage: plot <function> [xmin:xmax]")
                    print("Example: plot sin(x) -2*pi:2*pi")
                # Add more command-specific help here
                else:
                    print("No detailed help available for this command yet")
                print("="*60)
            else:
                print(f"Unknown command: {cmd}")

    def clear_screen(self, _=None):
        """Clear the screen."""
        print("\033c", end="")

    def show_history(self, _=None):
        """Show command history."""
        print("\nCommand History:")
        print("="*60)
        for i, cmd in enumerate(self.history[-20:], 1):
            print(f"{i:3}. {cmd}")
        print("="*60)

    def show_variables(self, _=None):
        """Show defined variables."""
        print("\nVariables:")
        print("="*60)
        for var, value in self.variables.items():
            print(f"{var} = {value}")
        print("="*60)

    def delete_variable(self, args: str):
        """Delete a variable."""
        var = args.strip()
        if var in self.variables:
            del self.variables[var]
            print(f"Deleted variable: {var}")
        else:
            print(f"Variable not found: {var}")

    def save_session(self, filename: str = "blacksky_session.json"):
        """Save current session to file."""
        try:
            data = {
                'variables': self.variables,
                'history': self.history
            }
            with open(filename.strip(), 'w') as f:
                json.dump(data, f)
            print(f"Session saved to {filename}")
        except Exception as e:
            print(f"Error saving session: {e}")

    def load_session(self, filename: str = "blacksky_session.json"):
        """Load session from file."""
        try:
            with open(filename.strip(), 'r') as f:
                data = json.load(f)
            self.variables = data.get('variables', {})
            self.history = data.get('history', [])
            print(f"Session loaded from {filename}")
        except Exception as e:
            print(f"Error loading session: {e}")

    def show_calculator_menu(self, _=None):
        """Show calculator mode menu."""
        print("\nCalculator Modes:")
        print("="*60)
        for num, desc in sorted(self.calculator_modes.items()):
            print(f"{num:>2}. {desc}")
        print("="*60)
        print("Use 'mode <name>' to switch modes (e.g., 'mode financial')")

    def solve_equation(self, equation: str):
        """Solve an equation symbolically."""
        try:
            x = symbols('x')
            eq = Eq(*[parse_expr(part.strip(), transformations=transformations) 
                     for part in equation.split('=', 1)])
            solutions = solve(eq, x)
            print("\nSolutions:")
            for sol in solutions:
                print(f"x = {sol}")
        except Exception as e:
            print(f"Error solving equation: {e}")

    def plot_function(self, args: str):
        """Plot a function."""
        try:
            parts = args.split(':')
            if len(parts) == 1:
                expr = parse_expr(parts[0], transformations=transformations)
                sympy_plot(expr, title=str(expr))
            elif len(parts) == 2:
                expr = parse_expr(parts[0], transformations=transformations)
                x_range = [float(x) for x in parts[1].split()]
                sympy_plot(expr, (symbols('x'), x_range[0], x_range[1]), title=str(expr))
            else:
                print("Usage: plot <expr> [xmin xmax]")
        except Exception as e:
            print(f"Error plotting function: {e}")

    def derive_function(self, args: str):
        """Calculate derivative of a function."""
        try:
            x = symbols('x')
            expr = parse_expr(args, transformations=transformations)
            derivative = diff(expr, x)
            print(f"\nDerivative of {expr}:")
            print(f"d/dx = {derivative}")
            print(f"Simplified: {simplify(derivative)}")
        except Exception as e:
            print(f"Error calculating derivative: {e}")

    def integrate_function(self, args: str):
        """Calculate integral of a function."""
        try:
            x = symbols('x')
            expr = parse_expr(args, transformations=transformations)
            integral = integrate(expr, x)
            print(f"\nIntegral of {expr}:")
            print(f"∫ dx = {integral} + C")
            print(f"Simplified: {simplify(integral)} + C")
        except Exception as e:
            print(f"Error calculating integral: {e}")

    def calculate_limit(self, args: str):
        """Calculate limit of a function."""
        try:
            parts = args.split()
            if len(parts) < 2:
                print("Usage: limit <expr> <point> [direction]")
                return
            
            x = symbols('x')
            expr = parse_expr(parts[0], transformations=transformations)
            point = parse_expr(parts[1], transformations=transformations)
            direction = parts[2] if len(parts) > 2 else '+'
            
            lim = limit(expr, x, point, direction)
            print(f"\nLimit of {expr} as x → {point}{' from ' + direction if len(parts) > 2 else ''}:")
            print(f"lim = {lim}")
        except Exception as e:
            print(f"Error calculating limit: {e}")

    def calculate_series(self, args: str):
        """Calculate series expansion of a function."""
        try:
            parts = args.split()
            if len(parts) < 2:
                print("Usage: series <expr> <point> [order]")
                return
            
            x = symbols('x')
            expr = parse_expr(parts[0], transformations=transformations)
            point = parse_expr(parts[1], transformations=transformations)
            order = int(parts[2]) if len(parts) > 2 else 6
            
            ser = series(expr, x, point, order).removeO()
            print(f"\nSeries expansion of {expr} at x = {point} (order {order}):")
            print(ser)
        except Exception as e:
            print(f"Error calculating series: {e}")

    def factor_expression(self, args: str):
        """Factor a polynomial expression."""
        try:
            expr = parse_expr(args, transformations=transformations)
            factored = factor(expr)
            print(f"\nFactored form of {expr}:")
            print(factored)
        except Exception as e:
            print(f"Error factoring expression: {e}")

    def expand_expression(self, args: str):
        """Expand a polynomial expression."""
        try:
            expr = parse_expr(args, transformations=transformations)
            expanded = expand(expr)
            print(f"\nExpanded form of {expr}:")
            print(expanded)
        except Exception as e:
            print(f"Error expanding expression: {e}")

    def find_root(self, args: str):
        """Find root of a function numerically."""
        try:
            x = symbols('x')
            expr = parse_expr(args, transformations=transformations)
            f = lambda x_val: float(expr.subs(x, x_val))
            
            # Try to find root near 0
            root = fsolve(f, 0)[0]
            print(f"\nRoot of {expr}:")
            print(f"x ≈ {root:.6f}")
        except Exception as e:
            print(f"Error finding root: {e}")

    def find_minimum(self, args: str):
        """Find minimum of a function numerically."""
        try:
            x = symbols('x')
            expr = parse_expr(args, transformations=transformations)
            f = lambda x_val: float(expr.subs(x, x_val))
            
            res = minimize(f, 0)
            if res.success:
                print(f"\nMinimum of {expr}:")
                print(f"x ≈ {res.x[0]:.6f}, f(x) ≈ {res.fun:.6f}")
            else:
                print("Failed to find minimum")
        except Exception as e:
            print(f"Error finding minimum: {e}")

    def find_maximum(self, args: str):
        """Find maximum of a function numerically."""
        try:
            x = symbols('x')
            expr = parse_expr(args, transformations=transformations)
            f = lambda x_val: -float(expr.subs(x, x_val))  # Negate for minimization
            
            res = minimize(f, 0)
            if res.success:
                print(f"\nMaximum of {expr}:")
                print(f"x ≈ {res.x[0]:.6f}, f(x) ≈ {-res.fun:.6f}")
            else:
                print("Failed to find maximum")
        except Exception as e:
            print(f"Error finding maximum: {e}")

    def calculate_stats(self, args: str):
        """Calculate basic statistics of a dataset."""
        try:
            data = [float(x) for x in args.split()]
            if not data:
                print("Need at least one data point")
                return
            
            print("\nStatistics:")
            print(f"Count: {len(data)}")
            print(f"Mean: {statistics.mean(data):.4f}")
            print(f"Median: {statistics.median(data):.4f}")
            try:
                print(f"Mode: {statistics.mode(data):.4f}")
            except statistics.StatisticsError:
                print("Mode: No unique mode found")
            print(f"Standard Deviation: {statistics.stdev(data):.4f}")
            print(f"Variance: {statistics.variance(data):.4f}")
            print(f"Minimum: {min(data):.4f}")
            print(f"Maximum: {max(data):.4f}")
            print(f"Range: {max(data) - min(data):.4f}")
        except Exception as e:
            print(f"Error calculating statistics: {e}")

    def linear_regression(self, args: str):
        """Perform linear regression on data."""
        try:
            if ';' not in args:
                print("Usage: regress <x1,x2,...>;<y1,y2,...>")
                return
            
            x_str, y_str = args.split(';', 1)
            x_data = [float(x) for x in x_str.split(',')]
            y_data = [float(y) for y in y_str.split(',')]
            
            if len(x_data) != len(y_data):
                print("X and Y data must have same length")
                return
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            print("\nLinear Regression Results:")
            print(f"Slope: {slope:.4f}")
            print(f"Intercept: {intercept:.4f}")
            print(f"R-squared: {r_value**2:.4f}")
            print(f"Equation: y = {slope:.4f}x + {intercept:.4f}")
            
            # Plot the data and regression line
            plt.scatter(x_data, y_data, label='Data')
            plt.plot(x_data, [slope*x + intercept for x in x_data], 'r', label='Regression Line')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Linear Regression')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error performing regression: {e}")

    def probability_distribution(self, args: str):
        """Calculate probability distribution values."""
        try:
            parts = args.split()
            if len(parts) < 2:
                print("Usage: dist <distribution> <params> [x]")
                print("Available distributions:", ", ".join(self.distributions.keys()))
                return
            
            dist_name = parts[0].lower()
            if dist_name not in self.distributions:
                print(f"Unknown distribution: {dist_name}")
                return
            
            dist = self.distributions[dist_name]
            params = [float(p) for p in parts[1:-1]]
            x = float(parts[-1]) if len(parts) > 2 else None
            
            if x is not None:
                pdf = dist.pdf(x, *params)
                cdf = dist.cdf(x, *params)
                print(f"\n{dist_name.title()} Distribution:")
                print(f"PDF at x={x}: {pdf:.6f}")
                print(f"CDF at x={x}: {cdf:.6f}")
            else:
                # Just show distribution info
                print(f"\n{dist_name.title()} Distribution:")
                print(f"Parameters: {params}")
                mean = dist.mean(*params)
                var = dist.var(*params)
                std = dist.std(*params)
                print(f"Mean: {mean:.4f}")
                print(f"Variance: {var:.4f}")
                print(f"Standard Deviation: {std:.4f}")
        except Exception as e:
            print(f"Error calculating distribution: {e}")

    def create_histogram(self, args: str):
        """Create a histogram from data."""
        try:
            data = [float(x) for x in args.split()]
            if not data:
                print("Need at least one data point")
                return
            
            plt.hist(data, bins='auto', edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error creating histogram: {e}")

    def matrix_operations(self, args: str):
        """Perform matrix operations."""
        try:
            if not args:
                print("Usage: matrix <operation> <args>")
                print("Operations: create, add, multiply, transpose, det, inv")
                return
                
            parts = args.split(maxsplit=1)
            operation = parts[0].lower()
            
            if operation == 'create':
                if len(parts) < 2:
                    print("Usage: matrix create <rows> <elements>")
                    print("Example: matrix create 2 [[1, 2], [3, 4]]")
                    return
                
                rows = int(parts[1].split()[0])
                elements = eval(' '.join(parts[1].split()[1:]))
                mat = Matrix(elements)
                var_name = f"matrix_{len(self.variables)}"
                self.variables[var_name] = mat
                print(f"Created matrix {var_name}:")
                print(mat)
                
            elif operation in ('add', 'multiply'):
                if len(parts) < 2:
                    print(f"Usage: matrix {operation} <matrix1> <matrix2>")
                    return
                
                mat1 = self.variables.get(parts[1].split()[0])
                mat2 = self.variables.get(parts[1].split()[1])
                
                if not mat1 or not mat2:
                    print("Both matrices must exist")
                    return
                
                if operation == 'add':
                    result = mat1 + mat2
                else:
                    result = mat1 * mat2
                
                print(f"Result of {operation}:")
                print(result)
                
            elif operation == 'transpose':
                if len(parts) < 2:
                    print("Usage: matrix transpose <matrix>")
                    return
                
                mat = self.variables.get(parts[1])
                if not mat:
                    print("Matrix not found")
                    return
                
                print("Transposed matrix:")
                print(mat.T)
                
            else:
                print(f"Unknown matrix operation: {operation}")
                
        except Exception as e:
            print(f"Error performing matrix operation: {e}")

    def matrix_determinant(self, args: str):
        """Calculate matrix determinant."""
        try:
            mat = self.variables.get(args.strip())
            if not mat:
                print("Matrix not found")
                return
                
            det = mat.det()
            print(f"Determinant: {det}")
        except Exception as e:
            print(f"Error calculating determinant: {e}")

    def matrix_inverse(self, args: str):
        """Calculate matrix inverse."""
        try:
            mat = self.variables.get(args.strip())
            if not mat:
                print("Matrix not found")
                return
                
            if mat.rows != mat.cols:
                print("Matrix must be square")
                return
                
            inv = mat.inv()
            print("Inverse matrix:")
            print(inv)
        except Exception as e:
            print(f"Error calculating inverse: {e}")

    def matrix_eigen(self, args: str):
        """Calculate matrix eigenvalues and eigenvectors."""
        try:
            mat = self.variables.get(args.strip())
            if not mat:
                print("Matrix not found")
                return
                
            if mat.rows != mat.cols:
                print("Matrix must be square")
                return
                
            eigenvals = mat.eigenvals()
            eigenvects = mat.eigenvects()
            
            print("Eigenvalues:")
            for val, mult in eigenvals.items():
                print(f"{val}: multiplicity {mult}")
                
            print("\nEigenvectors:")
            for vect in eigenvects:
                val, mult, basis = vect
                print(f"Eigenvalue {val}:")
                for b in basis:
                    print(b)
        except Exception as e:
            print(f"Error calculating eigenvalues: {e}")

    def unit_conversion(self, args: str):
        """Convert between units."""
        try:
            parts = args.split()
            if len(parts) < 4:
                print("Usage: convert <value> <from_unit> to <to_unit>")
                print("Example: convert 100 cm to m")
                print("Available unit categories:", ", ".join(self.units.keys()))
                return
                
            value = float(parts[0])
            from_unit = parts[1].lower()
            to_unit = parts[3].lower()
            
            # Find the category
            category = None
            for cat, units in self.units.items():
                if isinstance(units, dict):
                    if from_unit in units and to_unit in units:
                        category = cat
                        break
                elif from_unit in units and to_unit in units:  # For temperature
                    category = cat
                    break
                    
            if not category:
                print("Units must be of the same category")
                return
                
            if category == 'temperature':
                # Special handling for temperature
                if from_unit == 'c' and to_unit == 'f':
                    result = value * 9/5 + 32
                elif from_unit == 'f' and to_unit == 'c':
                    result = (value - 32) * 5/9
                elif from_unit == 'c' and to_unit == 'k':
                    result = value + 273.15
                elif from_unit == 'k' and to_unit == 'c':
                    result = value - 273.15
                elif from_unit == 'f' and to_unit == 'k':
                    result = (value - 32) * 5/9 + 273.15
                elif from_unit == 'k' and to_unit == 'f':
                    result = (value - 273.15) * 9/5 + 32
                else:
                    result = value  # Same unit
            else:
                # Standard conversion
                factor_from = self.units[category][from_unit]
                factor_to = self.units[category][to_unit]
                result = value * factor_from / factor_to
                
            print(f"{value} {from_unit} = {result:.6g} {to_unit}")
        except Exception as e:
            print(f"Error converting units: {e}")

    def prime_operations(self, args: str):
        """Perform prime number operations."""
        try:
            if not args:
                print("Usage: prime <operation> <args>")
                print("Operations: check, next, prev, range, factor")
                return
                
            parts = args.split()
            operation = parts[0].lower()
            
            if operation == 'check':
                if len(parts) < 2:
                    print("Usage: prime check <number>")
                    return
                    
                num = int(parts[1])
                if isprime(num):
                    print(f"{num} is prime")
                else:
                    print(f"{num} is not prime")
                    
            elif operation == 'next':
                if len(parts) < 2:
                    print("Usage: prime next <number>")
                    return
                    
                num = int(parts[1])
                print(f"Next prime after {num}: {nextprime(num)}")
                
            elif operation == 'prev':
                if len(parts) < 2:
                    print("Usage: prime prev <number>")
                    return
                    
                num = int(parts[1])
                print(f"Previous prime before {num}: {prevprime(num)}")
                
            elif operation == 'range':
                if len(parts) < 3:
                    print("Usage: prime range <start> <end>")
                    return
                    
                start = int(parts[1])
                end = int(parts[2])
                primes = list(primerange(start, end))
                print(f"Primes between {start} and {end}:")
                print(primes)
                
            elif operation == 'factor':
                if len(parts) < 2:
                    print("Usage: prime factor <number>")
                    return
                    
                num = int(parts[1])
                factors = factorint(num)
                print(f"Prime factors of {num}:")
                for p, exp in factors.items():
                    print(f"{p}^{exp}")
                    
            else:
                print(f"Unknown prime operation: {operation}")
                
        except Exception as e:
            print(f"Error performing prime operation: {e}")

    def gcd_operation(self, args: str):
        """Calculate greatest common divisor."""
        try:
            nums = [int(x) for x in args.split()]
            if len(nums) < 2:
                print("Need at least two numbers")
                return
                
            result = nums[0]
            for num in nums[1:]:
                result = gcd(result, num)
                
            print(f"GCD of {nums}: {result}")
        except Exception as e:
            print(f"Error calculating GCD: {e}")

    def lcm_operation(self, args: str):
        """Calculate least common multiple."""
        try:
            nums = [int(x) for x in args.split()]
            if len(nums) < 2:
                print("Need at least two numbers")
                return
                
            result = nums[0]
            for num in nums[1:]:
                result = lcm(result, num)
                
            print(f"LCM of {nums}: {result}")
        except Exception as e:
            print(f"Error calculating LCM: {e}")

    def factor_integer(self, args: str):
        """Factor an integer into primes."""
        try:
            num = int(args.strip())
            factors = factorint(num)
            print(f"Prime factorization of {num}:")
            for p, exp in factors.items():
                print(f"{p}^{exp}")
        except Exception as e:
            print(f"Error factoring integer: {e}")

    def taylor_series(self, args: str):
        """Calculate Taylor series expansion."""
        try:
            parts = args.split()
            if len(parts) < 3:
                print("Usage: taylor <expr> <variable> <point> [order]")
                return
                
            expr = parse_expr(parts[0], transformations=transformations)
            var = symbols(parts[1])
            point = float(parts[2])
            order = int(parts[3]) if len(parts) > 3 else 6
            
            series_expr = series(expr, var, point, order).removeO()
            print(f"Taylor series of {expr} at {var}={point}:")
            print(series_expr)
        except Exception as e:
            print(f"Error calculating Taylor series: {e}")

    def solve_ode(self, args: str):
        """Solve ordinary differential equation."""
        try:
            # This is a placeholder - actual ODE solving would require more complex implementation
            print("ODE solving is not yet fully implemented")
            print("Example usage would be: diffeq <equation> <initial_conditions>")
        except Exception as e:
            print(f"Error solving ODE: {e}")

    def financial_calculations(self, args: str):
        """Perform financial calculations."""
        try:
            if not args:
                print("Usage: finance <operation> <args>")
                print("Operations: fv, pv, pmt, nper, rate, npv, irr")
                return
                
            parts = args.split()
            operation = parts[0].lower()
            
            if operation == 'fv':
                if len(parts) < 5:
                    print("Usage: finance fv <rate> <nper> <pmt> <pv> [type]")
                    return
                    
                rate = float(parts[1])
                nper = int(parts[2])
                pmt = float(parts[3])
                pv = float(parts[4])
                typ = int(parts[5]) if len(parts) > 5 else 0
                
                fv = self._future_value(rate, nper, pmt, pv, typ)
                print(f"Future Value: {fv:.2f}")
                
            elif operation == 'pv':
                if len(parts) < 5:
                    print("Usage: finance pv <rate> <nper> <pmt> <fv> [type]")
                    return
                    
                rate = float(parts[1])
                nper = int(parts[2])
                pmt = float(parts[3])
                fv = float(parts[4])
                typ = int(parts[5]) if len(parts) > 5 else 0
                
                pv = self._present_value(rate, nper, pmt, fv, typ)
                print(f"Present Value: {pv:.2f}")
                
            elif operation == 'pmt':
                if len(parts) < 5:
                    print("Usage: finance pmt <rate> <nper> <pv> <fv> [type]")
                    return
                    
                rate = float(parts[1])
                nper = int(parts[2])
                pv = float(parts[3])
                fv = float(parts[4])
                typ = int(parts[5]) if len(parts) > 5 else 0
                
                pmt = self._payment(rate, nper, pv, fv, typ)
                print(f"Payment: {pmt:.2f}")
                
            elif operation == 'nper':
                if len(parts) < 5:
                    print("Usage: finance nper <rate> <pmt> <pv> <fv> [type]")
                    return
                    
                rate = float(parts[1])
                pmt = float(parts[2])
                pv = float(parts[3])
                fv = float(parts[4])
                typ = int(parts[5]) if len(parts) > 5 else 0
                
                nper = self._num_periods(rate, pmt, pv, fv, typ)
                print(f"Number of periods: {nper:.2f}")
                
            elif operation == 'rate':
                if len(parts) < 5:
                    print("Usage: finance rate <nper> <pmt> <pv> <fv> [type] [guess]")
                    return
                    
                nper = int(parts[1])
                pmt = float(parts[2])
                pv = float(parts[3])
                fv = float(parts[4])
                typ = int(parts[5]) if len(parts) > 5 else 0
                guess = float(parts[6]) if len(parts) > 6 else 0.1
                
                rate = self._interest_rate(nper, pmt, pv, fv, typ, guess)
                print(f"Interest rate: {rate*100:.4f}%")
                
            elif operation == 'npv':
                if len(parts) < 3:
                    print("Usage: finance npv <rate> <cashflow1> <cashflow2> ...")
                    return
                    
                rate = float(parts[1])
                cashflows = [float(x) for x in parts[2:]]
                npv = self._npv(rate, cashflows)
                print(f"Net Present Value: {npv:.2f}")
                
            elif operation == 'irr':
                if len(parts) < 2:
                    print("Usage: finance irr <cashflow1> <cashflow2> ... [guess]")
                    return
                    
                cashflows = [float(x) for x in parts[1:]]
                guess = 0.1
                if isinstance(cashflows[-1], str) and cashflows[-1].startswith('guess='):
                    guess = float(cashflows[-1].split('=')[1])
                    cashflows = cashflows[:-1]
                    
                irr = self._irr(cashflows, guess)
                print(f"Internal Rate of Return: {irr*100:.4f}%")
                
            else:
                print(f"Unknown financial operation: {operation}")
                
        except Exception as e:
            print(f"Error performing financial calculation: {e}")

    def _future_value(self, rate: float, nper: int, pmt: float, pv: float, typ: int = 0) -> float:
        """Calculate future value."""
        if rate == 0:
            return -pv - pmt * nper
        factor = (1 + rate) ** nper
        return -pv * factor - pmt * (1 + rate * typ) * (factor - 1) / rate

    def _present_value(self, rate: float, nper: int, pmt: float, fv: float, typ: int = 0) -> float:
        """Calculate present value."""
        if rate == 0:
            return -fv - pmt * nper
        factor = (1 + rate) ** nper
        return (-fv - pmt * (1 + rate * typ) * (factor - 1) / rate) / factor

    def _payment(self, rate: float, nper: int, pv: float, fv: float, typ: int = 0) -> float:
        """Calculate payment."""
        if rate == 0:
            return (-pv - fv) / nper
        factor = (1 + rate) ** nper
        return (pv * factor + fv) * rate / ((1 + rate * typ) * (1 - factor))

    def _num_periods(self, rate: float, pmt: float, pv: float, fv: float, typ: int = 0) -> float:
        """Calculate number of periods."""
        if rate == 0:
            return (-pv - fv) / pmt
        if pmt == 0:
            return math.log(fv / -pv) / math.log(1 + rate)
        a = pmt * (1 + rate * typ) / rate
        b = -fv + a
        c = pv + a
        return math.log(b / c) / math.log(1 + rate)

    def _interest_rate(self, nper: int, pmt: float, pv: float, fv: float, typ: int = 0, guess: float = 0.1) -> float:
        """Calculate interest rate."""
        def fn(rate):
            return self._future_value(rate, nper, pmt, pv, typ) + fv
            
        rate = fsolve(fn, guess)[0]
        return rate if rate > 0 else 0

    def _npv(self, rate: float, cashflows: List[float]) -> float:
        """Calculate net present value."""
        return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))

    def _irr(self, cashflows: List[float], guess: float = 0.1) -> float:
        """Calculate internal rate of return."""
        def fn(rate):
            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))
            
        irr = fsolve(fn, guess)[0]
        return irr if irr > -1 else -1

    def amortization_schedule(self, args: str):
        """Generate loan amortization schedule."""
        try:
            parts = args.split()
            if len(parts) < 4:
                print("Usage: amort <principal> <rate> <term> [type]")
                print("Example: amort 100000 0.05 360 (for 30-year loan at 5%)")
                return
                
            principal = float(parts[0])
            rate = float(parts[1]) / 12  # Convert annual to monthly
            term = int(parts[2])
            typ = int(parts[3]) if len(parts) > 3 else 0
            
            pmt = self._payment(rate, term, principal, 0, typ)
            print(f"Monthly Payment: {pmt:.2f}")
            print("\nAmortization Schedule (first 12 months):")
            print("Month\tPayment\tPrincipal\tInterest\tBalance")
            
            balance = principal
            for month in range(1, min(13, term + 1)):
                interest = balance * rate
                principal_pmt = pmt - interest
                balance -= principal_pmt
                
                print(f"{month}\t{pmt:.2f}\t{principal_pmt:.2f}\t{interest:.2f}\t{balance:.2f}")
                
        except Exception as e:
            print(f"Error generating amortization schedule: {e}")

    def calculate_roi(self, args: str):
        """Calculate return on investment."""
        try:
            parts = args.split()
            if len(parts) < 2:
                print("Usage: roi <gain> <cost>")
                return
                
            gain = float(parts[0])
            cost = float(parts[1])
            roi = (gain - cost) / cost * 100
            print(f"ROI: {roi:.2f}%")
        except Exception as e:
            print(f"Error calculating ROI: {e}")

    def date_calculations(self, args: str):
        """Perform date calculations."""
        try:
            if not args:
                print("Usage: date <operation> <args>")
                print("Operations: now, add, diff, weekday, leap")
                return
                
            parts = args.split()
            operation = parts[0].lower()
            
            if operation == 'now':
                now = datetime.now()
                print(f"Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
            elif operation == 'add':
                if len(parts) < 3:
                    print("Usage: date add <date> <days>")
                    print("Date format: YYYY-MM-DD")
                    return
                    
                date_str = parts[1]
                days = int(parts[2])
                date = datetime.strptime(date_str, "%Y-%m-%d")
                new_date = date + timedelta(days=days)
                print(f"New date: {new_date.strftime('%Y-%m-%d')}")
                
            elif operation == 'diff':
                if len(parts) < 3:
                    print("Usage: date diff <date1> <date2>")
                    print("Date format: YYYY-MM-DD")
                    return
                    
                date1 = datetime.strptime(parts[1], "%Y-%m-%d")
                date2 = datetime.strptime(parts[2], "%Y-%m-%d")
                delta = abs(date2 - date1)
                print(f"Difference: {delta.days} days")
                
            elif operation == 'weekday':
                if len(parts) < 2:
                    print("Usage: date weekday <date>")
                    print("Date format: YYYY-MM-DD")
                    return
                    
                date = datetime.strptime(parts[1], "%Y-%m-%d")
                weekday = calendar.day_name[date.weekday()]
                print(f"Weekday: {weekday}")
                
            elif operation == 'leap':
                if len(parts) < 2:
                    print("Usage: date leap <year>")
                    return
                    
                year = int(parts[1])
                if calendar.isleap(year):
                    print(f"{year} is a leap year")
                else:
                    print(f"{year} is not a leap year")
                    
            else:
                print(f"Unknown date operation: {operation}")
                
        except Exception as e:
            print(f"Error performing date calculation: {e}")

    def time_conversion(self, args: str):
        """Convert between timezones."""
        try:
            if not args:
                print("Usage: time <time> <from_tz> <to_tz>")
                print("Example: time 12:00:00 UTC EST")
                print("Available timezones:", ", ".join(pytz.all_timezones[:10]), "...")
                return
                
            parts = args.split()
            if len(parts) < 3:
                print("Need time, from timezone, and to timezone")
                return
                
            time_str = parts[0]
            from_tz = parts[1]
            to_tz = parts[2]
            
            # Parse time (assuming HH:MM:SS format)
            time_parts = time_str.split(':')
            if len(time_parts) < 2:
                print("Time format should be HH:MM or HH:MM:SS")
                return
                
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            second = int(time_parts[2]) if len(time_parts) > 2 else 0
            
            # Get current date for timezone conversion
            now = datetime.now()
            naive_time = datetime(now.year, now.month, now.day, hour, minute, second)
            
            # Convert between timezones
            from_zone = pytz.timezone(from_tz)
            to_zone = pytz.timezone(to_tz)
            
            localized_time = from_zone.localize(naive_time)
            converted_time = localized_time.astimezone(to_zone)
            
            print(f"{time_str} {from_tz} = {converted_time.strftime('%H:%M:%S')} {to_tz}")
        except Exception as e:
            print(f"Error converting time: {e}")

    def countdown_timer(self, args: str):
        """Start a countdown timer."""
        try:
            if not args:
                print("Usage: countdown <seconds>")
                return
                
            seconds = int(args.strip())
            print(f"Countdown: {seconds} seconds")
            
            for remaining in range(seconds, 0, -1):
                mins, secs = divmod(remaining, 60)
                print(f"\r{mins:02d}:{secs:02d}", end='')
                time.sleep(1)
                
            print("\nTime's up!")
        except Exception as e:
            print(f"Error starting countdown: {e}")

    def system_info(self, _=None):
        """Show system information."""
        print("\nSystem Information:")
        print("="*60)
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        print(f"SymPy version: {sympy.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Matplotlib version: {plt.matplotlib.__version__}")
        print("="*60)

    def benchmark(self, _=None):
        """Run performance benchmarks."""
        print("\nRunning benchmarks...")
        
        # Math operations
        start = time.time()
        for _ in range(100000):
            math.sin(math.pi/4)
        math_time = time.time() - start
        
        # SymPy operations
        x = symbols('x')
        start = time.time()
        for _ in range(1000):
            diff(sin(x), x)
        sympy_time = time.time() - start
        
        # NumPy operations
        arr = np.random.rand(1000)
        start = time.time()
        for _ in range(1000):
            np.sin(arr)
        numpy_time = time.time() - start
        
        print("\nBenchmark Results:")
        print("="*60)
        print(f"Math operations: {math_time:.4f} sec (100,000 sin calculations)")
        print(f"SymPy operations: {sympy_time:.4f} sec (1,000 symbolic derivatives)")
        print(f"NumPy operations: {numpy_time:.4f} sec (1,000 array sin calculations)")
        print("="*60)

    def evaluate_code(self, args: str):
        """Evaluate Python code (restricted)."""
        try:
            # Very basic and unsafe - in production you'd want a proper sandbox
            if any(bad in args.lower() for bad in ['import', 'open', 'exec', 'eval', 'os.', 'sys.', 'subprocess']):
                print("Potentially dangerous operation blocked")
                return
                
            result = eval(args, {'__builtins__': None}, {
                'math': math,
                'np': np,
                'plt': plt,
                'sympy': sympy,
                'stats': stats,
                'random': random
            })
            print(result)
        except Exception as e:
            print(f"Error evaluating code: {e}")

    def create_lambda(self, args: str):
        """Create a lambda function."""
        try:
            parts = args.split(':', 1)
            if len(parts) < 2:
                print("Usage: lambda <args>:<expression>")
                return
                
            var_name = f"lambda_{len(self.variables)}"
            self.variables[var_name] = eval(f"lambda {parts[0]}: {parts[1]}", {'__builtins__': None}, {
                'math': math,
                'np': np,
                'sympy': sympy
            })
            print(f"Created lambda function: {var_name}")
        except Exception as e:
            print(f"Error creating lambda: {e}")

    def graph_operations(self, args: str):
        """Perform graph theory operations."""
        try:
            if not args:
                print("Usage: graph <operation> <args>")
                print("Operations: create, add_node, add_edge, shortest_path")
                return
                
            parts = args.split()
            operation = parts[0].lower()
            
            if operation == 'create':
                if len(parts) < 2:
                    print("Usage: graph create <name> <type>")
                    print("Types: directed, undirected, weighted")
                    return
                    
                name = parts[1]
                graph_type = parts[2] if len(parts) > 2 else 'undirected'
                
                if graph_type not in self.graph_types:
                    print(f"Unknown graph type: {graph_type}")
                    return
                    
                self.graph_types[graph_type][name] = {}
                print(f"Created {graph_type} graph: {name}")
                
            elif operation == 'add_node':
                if len(parts) < 3:
                    print("Usage: graph add_node <graph> <node>")
                    return
                    
                graph_name = parts[1]
                node = parts[2]
                
                # Find the graph
                graph = None
                for g_type in self.graph_types.values():
                    if graph_name in g_type:
                        graph = g_type[graph_name]
                        break
                        
                if not graph:
                    print(f"Graph not found: {graph_name}")
                    return
                    
                graph[node] = {}
                print(f"Added node {node} to graph {graph_name}")
                
            elif operation == 'add_edge':
                if len(parts) < 4:
                    print("Usage: graph add_edge <graph> <node1> <node2> [weight]")
                    return
                    
                graph_name = parts[1]
                node1 = parts[2]
                node2 = parts[3]
                weight = float(parts[4]) if len(parts) > 4 else 1
                
                # Find the graph
                graph = None
                graph_type = None
                for g_type_name, g_type in self.graph_types.items():
                    if graph_name in g_type:
                        graph = g_type[graph_name]
                        graph_type = g_type_name
                        break
                        
                if not graph:
                    print(f"Graph not found: {graph_name}")
                    return
                    
                if node1 not in graph:
                    print(f"Node not found: {node1}")
                    return
                    
                if node2 not in graph:
                    print(f"Node not found: {node2}")
                    return
                    
                if graph_type == 'weighted':
                    graph[node1][node2] = weight
                    graph[node2][node1] = weight  # Undirected weighted
                else:
                    graph[node1].append(node2)
                    if graph_type == 'undirected':
                        graph[node2].append(node1)
                        
                print(f"Added edge between {node1} and {node2}")
                
            elif operation == 'shortest_path':
                if len(parts) < 4:
                    print("Usage: graph shortest_path <graph> <start> <end>")
                    return
                    
                graph_name = parts[1]
                start = parts[2]
                end = parts[3]
                
                # Find the graph
                graph = None
                graph_type = None
                for g_type_name, g_type in self.graph_types.items():
                    if graph_name in g_type:
                        graph = g_type[graph_name]
                        graph_type = g_type_name
                        break
                        
                if not graph:
                    print(f"Graph not found: {graph_name}")
                    return
                    
                # Dijkstra's algorithm for weighted graphs
                if graph_type == 'weighted':
                    # Implementation would go here
                    print("Shortest path calculation for weighted graphs not yet implemented")
                else:
                    # BFS for unweighted graphs
                    visited = {start: None}
                    queue = [start]
                    
                    while queue:
                        current = queue.pop(0)
                        if current == end:
                            break
                            
                        for neighbor in graph.get(current, []):
                            if neighbor not in visited:
                                visited[neighbor] = current
                                queue.append(neighbor)
                                
                    # Reconstruct path
                    path = []
                    current = end
                    while current:
                        path.append(current)
                        current = visited.get(current)
                        
                    path.reverse()
                    
                    if path[0] == start:
                        print(f"Shortest path: {' -> '.join(path)}")
                    else:
                        print(f"No path exists between {start} and {end}")
                        
            else:
                print(f"Unknown graph operation: {operation}")
                
        except Exception as e:
            print(f"Error performing graph operation: {e}")

    def game_theory(self, args: str):
        """Perform game theory calculations."""
        try:
            if not args:
                print("Usage: game <operation> <args>")
                print("Operations: list, show, nash")
                print("Available games:", ", ".join(self.game_theory_games.keys()))
                return
                
            parts = args.split()
            operation = parts[0].lower()
            
            if operation == 'list':
                print("\nAvailable Games:")
                for game in self.game_theory_games:
                    print(f"- {game}: {self.game_theory_games[game]['description']}")
                    
            elif operation == 'show':
                if len(parts) < 2:
                    print("Usage: game show <game_name>")
                    return
                    
                game_name = parts[1]
                if game_name not in self.game_theory_games:
                    print(f"Game not found: {game_name}")
                    return
                    
                game = self.game_theory_games[game_name]
                print(f"\nGame: {game_name}")
                print(f"Description: {game['description']}")
                print("\nPayoff Matrix:")
                for (move1, move2), (payoff1, payoff2) in game['payoffs'].items():
                    print(f"{move1}/{move2}: Player1={payoff1}, Player2={payoff2}")
                    
            elif operation == 'nash':
                if len(parts) < 2:
                    print("Usage: game nash <game_name>")
                    return
                    
                game_name = parts[1]
                if game_name not in self.game_theory_games:
                    print(f"Game not found: {game_name}")
                    return
                    
                game = self.game_theory_games[game_name]
                payoffs = game['payoffs']
                
                # Very basic Nash equilibrium check (only pure strategies)
                print("\nChecking for Nash equilibria (pure strategies only)...")
                
                # Get all possible moves
                moves_p1 = set()
                moves_p2 = set()
                for (m1, m2) in payoffs.keys():
                    moves_p1.add(m1)
                    moves_p2.add(m2)
                    
                # Check each strategy profile
                equilibria = []
                for m1 in moves_p1:
                    for m2 in moves_p2:
                        current_payoff = payoffs[(m1, m2)]
                        is_equilibrium = True
                        
                        # Check if player 1 can do better by deviating
                        for alt_m1 in moves_p1:
                            if alt_m1 != m1 and payoffs[(alt_m1, m2)][0] > current_payoff[0]:
                                is_equilibrium = False
                                break
                                
                        if is_equilibrium:
                            # Check if player 2 can do better by deviating
                            for alt_m2 in moves_p2:
                                if alt_m2 != m2 and payoffs[(m1, alt_m2)][1] > current_payoff[1]:
                                    is_equilibrium = False
                                    break
                                    
                        if is_equilibrium:
                            equilibria.append((m1, m2))
                            
                if equilibria:
                    print("Nash equilibria found:")
                    for eq in equilibria:
                        print(f"- {eq[0]} (Player 1), {eq[1]} (Player 2)")
                        print(f"  Payoffs: {payoffs[eq]}")
                else:
                    print("No pure strategy Nash equilibria found")
                    
            else:
                print(f"Unknown game operation: {operation}")
                
        except Exception as e:
            print(f"Error performing game theory operation: {e}")

    def chemistry_calculator(self, args: str):
        """Perform chemistry calculations."""
        try:
            if not args:
                print("Usage: chem <operation> <args>")
                print("Operations: element, molar_mass, balance")
                return
                
            parts = args.split()
            operation = parts[0].lower()
            
            if operation == 'element':
                if len(parts) < 2:
                    print("Usage: chem element <symbol>")
                    return
                    
                symbol = parts[1].capitalize()
                if symbol not in self.chemical_elements:
                    print(f"Element not found: {symbol}")
                    return
                    
                element = self.chemical_elements[symbol]
                print(f"\nElement: {element['name']} ({symbol})")
                print(f"Atomic Number: {element['atomic_number']}")
                print(f"Atomic Mass: {element['mass']} g/mol")
                
            elif operation == 'molar_mass':
                if len(parts) < 2:
                    print("Usage: chem molar_mass <formula>")
                    print("Example: chem molar_mass H2O")
                    return
                    
                formula = parts[1]
                total_mass = 0
                
                # Very basic formula parsing (only works for simple formulas)
                matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
                if not matches:
                    print("Invalid chemical formula")
                    return
                    
                for element, count in matches:
                    if element not in self.chemical_elements:
                        print(f"Unknown element: {element}")
                        return
                        
                    atomic_mass = self.chemical_elements[element]['mass']
                    num_atoms = int(count) if count else 1
                    total_mass += atomic_mass * num_atoms
                    
                print(f"Molar mass of {formula}: {total_mass:.4f} g/mol")
                
            elif operation == 'balance':
                if len(parts) < 2:
                    print("Usage: chem balance <equation>")
                    print("Example: chem balance H2 + O2 = H2O")
                    return
                    
                equation = ' '.join(parts[1:])
                print("Chemical equation balancing is not yet fully implemented")
                print("Would balance equation:", equation)
                
            else:
                print(f"Unknown chemistry operation: {operation}")
                
        except Exception as e:
            print(f"Error performing chemistry calculation: {e}")

    def physics_calculator(self, args: str):
        """Perform physics calculations."""
        try:
            if not args:
                print("Usage: physics <operation> <args>")
                print("Operations: constant, kinematic")
                print("Available constants:", ", ".join(self.physical_constants.keys()))
                return
                
            parts = args.split()
            operation = parts[0].lower()
            
            if operation == 'constant':
                if len(parts) < 2:
                    print("Usage: physics constant <name>")
                    return
                    
                const_name = parts[1].lower()
                if const_name not in self.physical_constants:
                    print(f"Constant not found: {const_name}")
                    return
                    
                print(f"{const_name}: {self.physical_constants[const_name]}")
                
            elif operation == 'kinematic':
                if len(parts) < 2:
                    print("Usage: physics kinematic <equation> <values>")
                    print("Equations: d=v0t+0.5at^2, v=v0+at, v^2=v0^2+2ad")
                    return
                    
                eq = parts[1].lower()
                values = {}
                for part in parts[2:]:
                    if '=' in part:
                        var, val = part.split('=')
                        values[var] = float(val)
                        
                if eq == 'd=v0t+0.5at^2':
                    if 'd' not in values and all(v in values for v in ['v0', 't', 'a']):
                        d = values['v0'] * values['t'] + 0.5 * values['a'] * values['t'] ** 2
                        print(f"Displacement (d): {d:.4f}")
                    else:
                        print("Need v0, t, and a to solve for d")
                        
                elif eq == 'v=v0+at':
                    if 'v' not in values and all(v in values for v in ['v0', 'a', 't']):
                        v = values['v0'] + values['a'] * values['t']
                        print(f"Final velocity (v): {v:.4f}")
                    else:
                        print("Need v0, a, and t to solve for v")
                        
                elif eq == 'v^2=v0^2+2ad':
                    if 'v' not in values and all(v in values for v in ['v0', 'a', 'd']):
                        v = math.sqrt(values['v0'] ** 2 + 2 * values['a'] * values['d'])
                        print(f"Final velocity (v): {v:.4f}")
                    else:
                        print("Need v0, a, and d to solve for v")
                        
                else:
                    print(f"Unknown kinematic equation: {eq}")
                    
            else:
                print(f"Unknown physics operation: {operation}")
                
        except Exception as e:
            print(f"Error performing physics calculation: {e}")

    def engineering_calculator(self, args: str):
        """Perform engineering calculations."""
        try:
            if not args:
                print("Usage: eng <operation> <args>")
                print("Operations: list, formula")
                print("Available tools:", ", ".join(self.engineering_tools.keys()))
                return
                
            parts = args.split()
            operation = parts[0].lower()
            
            if operation == 'list':
                print("\nEngineering Tools:")
                for tool in self.engineering_tools:
                    print(f"- {tool}: {self.engineering_tools[tool]['description']}")
                    
            elif operation == 'formula':
                if len(parts) < 2:
                    print("Usage: eng formula <tool>")
                    return
                    
                tool = parts[1].lower()
                if tool not in self.engineering_tools:
                    print(f"Tool not found: {tool}")
                    return
                    
                print(f"\n{tool.title()}:")
                print(f"Formula: {self.engineering_tools[tool]['formula']}")
                print(f"Description: {self.engineering_tools[tool]['description']}")
                
            else:
                print(f"Unknown engineering operation: {operation}")
                
        except Exception as e:
            print(f"Error performing engineering calculation: {e}")

    def math_puzzles(self, args: str):
        """Show math puzzles and brain teasers."""
        try:
            if not args:
                print("Usage: puzzle <operation>")
                print("Operations: list, show, solve")
                return
                
            parts = args.split()
            operation = parts[0].lower()
            
            if operation == 'list':
                print("\nMath Puzzles:")
                for i, puzzle in enumerate(self.math_puzzles_list, 1):
                    print(f"{i}. {puzzle['question']}")
                    
            elif operation == 'show':
                if len(parts) < 2:
                    print("Usage: puzzle show <number>")
                    return
                    
                try:
                    index = int(parts[1]) - 1
                    if 0 <= index < len(self.math_puzzles_list):
                        puzzle = self.math_puzzles_list[index]
                        print(f"\nPuzzle {index + 1}:")
                        print(puzzle['question'])
                        print(f"\nHint: {puzzle['hint']}")
                    else:
                        print("Invalid puzzle number")
                except ValueError:
                    print("Please enter a valid puzzle number")
                    
            elif operation == 'solve':
                if len(parts) < 3:
                    print("Usage: puzzle solve <number> <answer>")
                    return
                    
                try:
                    index = int(parts[1]) - 1
                    answer = ' '.join(parts[2:])
                    
                    if 0 <= index < len(self.math_puzzles_list):
                        puzzle = self.math_puzzles_list[index]
                        if answer.lower() == puzzle['answer'].lower():
                            print("Correct! Well done!")
                        else:
                            print(f"Sorry, that's incorrect. The answer is: {puzzle['answer']}")
                    else:
                        print("Invalid puzzle number")
                except ValueError:
                    print("Please enter a valid puzzle number")
                    
            else:
                print(f"Unknown puzzle operation: {operation}")
                
        except Exception as e:
            print(f"Error with math puzzles: {e}")

    def math_quiz(self, args: str):
        """Start an interactive math quiz."""
        try:
            if not args:
                print("Starting math quiz...")
                print("Type 'quit' to end the quiz at any time")
                self.quiz_active = True
                
                operations = ['+', '-', '*', '/']
                score = 0
                total = 5  # Number of questions
                
                for i in range(total):
                    if not self.quiz_active:
                        break
                        
                    a = random.randint(1, 10)
                    b = random.randint(1, 10)
                    op = random.choice(operations)
                    
                    question = f"{a} {op} {b}"
                    if op == '+':
                        answer = a + b
                    elif op == '-':
                        answer = a - b
                    elif op == '*':
                        answer = a * b
                    else:
                        answer = round(a / b, 2)
                        
                    while True:
                        user_input = input(f"\nQuestion {i+1}/{total}: What is {question}? ")
                        if user_input.lower() == 'quit':
                            self.quiz_active = False
                            break
                            
                        try:
                            user_answer = float(user_input)
                            if abs(user_answer - answer) < 0.01:  # Allow for floating point imprecision
                                print("Correct!")
                                score += 1
                            else:
                                print(f"Sorry, the correct answer is {answer}")
                            break
                        except ValueError:
                            print("Please enter a number or 'quit'")
                            
                if self.quiz_active:
                    print(f"\nQuiz complete! Your score: {score}/{total} ({score/total*100:.0f}%)")
                else:
                    print(f"\nQuiz ended early. Your score: {score}/{i} ({score/i*100:.0f}% if completed)")
                    
                self.quiz_active = False
                
            elif args.lower() == 'stop':
                self.quiz_active = False
                print("Quiz will end after current question")
            else:
                print("Usage: quiz (to start) or quiz stop (to end)")
                
        except Exception as e:
            print(f"Error with math quiz: {e}")
            self.quiz_active = False

    def switch_mode(self, args: str):
        """Switch calculator mode."""
        try:
            if not args:
                self.show_calculator_menu()
                return
                
            mode = args.strip().lower()
            
            if mode in ('scientific', '1'):
                self.current_mode = CalculatorMode.SCIENTIFIC
                print("Switched to Scientific Calculator mode")
            elif mode in ('graphing', '2'):
                self.current_mode = CalculatorMode.GRAPHING
                print("Switched to Graphing Calculator mode")
            elif mode in ('matrix', '3'):
                self.current_mode = CalculatorMode.MATRIX
                print("Switched to Matrix Calculator mode")
            elif mode in ('statistical', '4'):
                self.current_mode = CalculatorMode.STATISTICAL
                print("Switched to Statistical Calculator mode")
            elif mode in ('financial', '5'):
                self.current_mode = CalculatorMode.FINANCIAL
                print("Switched to Financial Calculator mode")
            elif mode in ('programming', '6'):
                self.current_mode = CalculatorMode.PROGRAMMING
                print("Switched to Programming Calculator mode")
            elif mode in ('unit', 'converter', '7'):
                self.current_mode = CalculatorMode.UNIT_CONVERTER
                print("Switched to Unit Converter mode")
            elif mode in ('date', 'time', '8'):
                self.current_mode = CalculatorMode.DATE_TIME
                print("Switched to Date/Time Calculator mode")
            elif mode in ('game', '9'):
                self.current_mode = CalculatorMode.GAME_THEORY
                print("Switched to Game Theory Calculator mode")
            elif mode in ('chemistry', 'chem', '10'):
                self.current_mode = CalculatorMode.CHEMISTRY
                print("Switched to Chemistry Calculator mode")
            elif mode in ('physics', '11'):
                self.current_mode = CalculatorMode.PHYSICS
                print("Switched to Physics Calculator mode")
            elif mode in ('engineering', 'eng', '12'):
                self.current_mode = CalculatorMode.ENGINEERING
                print("Switched to Engineering Calculator mode")
            elif mode in ('main', '13'):
                self.current_mode = CalculatorMode.SCIENTIFIC
                print("Switched to Main Calculator mode")
            else:
                print(f"Unknown mode: {mode}")
                print("Available modes: scientific, graphing, matrix, statistical, financial, programming, unit, date, game, chemistry, physics, engineering")
                
        except Exception as e:
            print(f"Error switching mode: {e}")

    def exit_cli(self, _=None):
        """Exit the calculator."""
        print("Exiting Blacksky Math Calculator...")
        sys.exit(0)

    def process_input(self, user_input: str):
        """Process user input and execute appropriate command or evaluation."""
        try:
            # Add to history
            self.history.append(user_input)
            
            # Check for empty input
            if not user_input.strip():
                return
                
            # Check for commands
            if user_input.startswith('!'):
                cmd = user_input[1:].split(maxsplit=1)
                cmd_name = cmd[0].lower()
                cmd_args = cmd[1] if len(cmd) > 1 else ""
                
                # Check for aliases
                cmd_name = self.aliases.get(cmd_name, cmd_name)
                
                if cmd_name in self.commands:
                    self.commands[cmd_name](cmd_args)
                else:
                    print(f"Unknown command: {cmd_name}")
                return
                
            # Check for variable assignment
            if '=' in user_input:
                var, expr = user_input.split('=', 1)
                var = var.strip()
                expr = expr.strip()
                
                try:
                    value = self.evaluate_expression(expr)
                    self.variables[var] = value
                    print(f"{var} = {value}")
                except Exception as e:
                    print(f"Error assigning variable: {e}")
                return
                
            # Evaluate expression
            try:
                result = self.evaluate_expression(user_input)
                print(result)
            except Exception as e:
                print(f"Error evaluating expression: {e}")
                
        except Exception as e:
            print(f"Error processing input: {e}")

    def run_cli(self):
        """Run the calculator in command-line interface mode."""
        print("Blacksky Math Calculator (type '!help' for help, '!exit' to quit)")
        
        while True:
            try:
                # Get input with appropriate prompt based on mode
                prompt = {
                    CalculatorMode.SCIENTIFIC: "SCI> ",
                    CalculatorMode.GRAPHING: "GRAPH> ",
                    CalculatorMode.MATRIX: "MATRIX> ",
                    CalculatorMode.STATISTICAL: "STAT> ",
                    CalculatorMode.FINANCIAL: "FIN> ",
                    CalculatorMode.PROGRAMMING: "PROG> ",
                    CalculatorMode.UNIT_CONVERTER: "UNIT> ",
                    CalculatorMode.DATE_TIME: "DATE> ",
                    CalculatorMode.GAME_THEORY: "GAME> ",
                    CalculatorMode.CHEMISTRY: "CHEM> ",
                    CalculatorMode.PHYSICS: "PHYS> ",
                    CalculatorMode.ENGINEERING: "ENG> "
                }.get(self.current_mode, ">>> ")
                
                user_input = input(prompt).strip()
                self.process_input(user_input)
                
            except KeyboardInterrupt:
                print("\nType '!exit' to quit or continue calculating")
            except EOFError:
                self.exit_cli()
            except Exception as e:
                print(f"Unexpected error: {e}")

if __name__ == "__main__":
    calculator = BlackskyMath()
    calculator.run_cli()