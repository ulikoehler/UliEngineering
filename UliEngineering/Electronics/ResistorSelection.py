#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Sequence
import itertools
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args
from UliEngineering.Electronics.VoltageDivider import voltage_divider_voltage
from UliEngineering.Electronics.Resistors import ESeries, standard_resistors, power_dissipated_in_resistor_by_current, series_resistors, current_through_resistor

__all__ = [
    'ResistorSelection',
    'ResistorSeriesWeights',
    'ResistorSeriesCostFunctor',
    'ResistorAroundValueCostFunctor',
    'ResistorPowerCostFunctor',
    'resistor_selection_error_matrix',
    'feedback_network_error',
    'select_resistors',
]

@dataclass
class ResistorSelection:
    """Result of resistor selection containing the resistor values, error, and total cost."""
    r1: float
    r2: float
    error: float
    total_cost: float

@dataclass
class ResistorSeriesWeights:
    """Weights for different resistor series."""
    E6: float = 0.95
    E12: float = 1.0
    E24: float = 2.0
    E48: float = 4.0
    E96: float = 8.0
    E192: float = 16.0
    non_series: float = 100.0

class ResistorSeriesCostFunctor(object):
    """
    Cost functor that assigns costs to resistors based on their E-series membership.
    Uses precomputed lookup structures for fast evaluation.
    """
    
    def __init__(self, weights=None, tolerance=0.001):
        """
        Initialize the cost functor.
        
        Parameters
        ----------
        weights : ResistorSeriesWeights, optional
            Weights for different E-series. If None, uses default weights.
        tolerance : float
            Tolerance for matching resistor values (as fraction, e.g. 0.001 = 0.1%)
        """
        self.weights = weights if weights is not None else ResistorSeriesWeights()
        self.tolerance = tolerance
        
        # Precompute lookup sets for fast membership testing
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build optimized lookup tables for E-series membership testing."""
        # Use standard_resistors() to generate complete lists
        self.e6_values = np.array(sorted(list(standard_resistors(sequence=ESeries.E6))))
        self.e12_values = np.array(sorted(list(standard_resistors(sequence=ESeries.E12))))
        self.e24_values = np.array(sorted(list(standard_resistors(sequence=ESeries.E24))))
        self.e48_values = np.array(sorted(list(standard_resistors(sequence=ESeries.E48))))
        self.e96_values = np.array(sorted(list(standard_resistors(sequence=ESeries.E96))))
        self.e192_values = np.array(sorted(list(standard_resistors(sequence=ESeries.E192))))
    
    def _is_in_series(self, value, series_values):
        """Check if a value is in a series within tolerance using binary search."""
        value = normalize_numeric(value)
        
        # Use binary search to find closest values
        idx = np.searchsorted(series_values, value)
        
        # Check value at idx and idx-1
        candidates = []
        if idx < len(series_values):
            candidates.append(series_values[idx])
        if idx > 0:
            candidates.append(series_values[idx-1])
            
        # Check if any candidate is within tolerance
        for candidate in candidates:
            relative_error = abs(value - candidate) / candidate
            if relative_error <= self.tolerance:
                return True
        return False
    
    def __call__(self, resistor_value):
        """
        Evaluate the cost of a resistor value based on its E-series membership.
        
        Parameters
        ----------
        resistor_value : float or Engineer string
            The resistor value to evaluate
            
        Returns
        -------
        float
            Cost value based on series membership
        """
        # Check series membership in order of preference (E6 first as most common/cheapest)
        if self._is_in_series(resistor_value, self.e6_values):
            return self.weights.E6
        if self._is_in_series(resistor_value, self.e12_values):
            return self.weights.E12
        if self._is_in_series(resistor_value, self.e24_values):
            return self.weights.E24
        if self._is_in_series(resistor_value, self.e48_values):
            return self.weights.E48
        if self._is_in_series(resistor_value, self.e96_values):
            return self.weights.E96
        if self._is_in_series(resistor_value, self.e192_values):
            return self.weights.E192
        else:
            return self.weights.non_series

def resistor_selection_error_matrix(error_function, r1_sequence, r2_sequence):
    """
    Compute an error matrix for selecting two resistors.
    
    Parameters:
    -----------
    error_function : callable
        Function that takes (r1, r2) and returns the percentage deviation 
        from the desired value
    r1_sequence : array-like
        Sequence of resistor values for the first resistor
    r2_sequence : array-like
        Sequence of resistor values for the second resistor
        
    Returns:
    --------
    numpy.ndarray
        2D array where rows represent r1_sequence and columns represent r2_sequence.
        Each element contains the error percentage for that resistor combination.
    """
    # Normalize resistor values to floats
    r1_values = np.array([normalize_numeric(r) for r in r1_sequence])
    r2_values = np.array([normalize_numeric(r) for r in r2_sequence])
    
    # Create meshgrid for broadcasting
    r1_mesh, r2_mesh = np.meshgrid(r1_values, r2_values, indexing='ij')
    
    # Vectorize the error function to work with numpy arrays
    vectorized_error = np.vectorize(error_function)
    
    # Compute error matrix using broadcasting
    error_matrix = vectorized_error(r1_mesh, r2_mesh)
    
    return error_matrix

@normalize_numeric_args
def feedback_network_error(r1, r2, input_voltage, target_voltage, load=None):
    """
    Calculate the percentage deviation of a feedback network output voltage
    from the target voltage.
    
    In a typical feedback network, r1 is the upper resistor (connected to input)
    and r2 is the lower resistor (connected to ground). The feedback voltage
    is taken from the junction between r1 and r2.
    
    Parameters
    ----------
    r1 : float or Engineer string
        Upper resistor value in Ohms
    r2 : float or Engineer string  
        Lower resistor value in Ohms
    input_voltage : float or Engineer string
        Input voltage to the feedback network
    target_voltage : float or Engineer string
        Desired output voltage at the feedback point
    load : float or Engineer string, optional
        Load resistance in parallel with r2. If None, no load is considered.
        
    Returns
    -------
    float
        Percentage deviation from target voltage (positive = higher, negative = lower)
    """
    # Use voltage divider function with optional load
    rload = load if load is not None else np.inf
    actual_voltage = voltage_divider_voltage(r1, r2, input_voltage, rload=rload)
    
    # Calculate percentage deviation
    deviation_percent = ((actual_voltage - target_voltage) / target_voltage) * 100
    
    return deviation_percent

class ResistorAroundValueCostFunctor(object):
    """
    Cost functor that evaluates how close a resistor value is to a target value
    using logarithmic criteria with configurable base.
    
    Returns the absolute difference in "orders of magnitude" between the
    resistor value and target value.
    """
    
    def __init__(self, target_value, base=10.0):
        """
        Initialize the functor.
        
        Parameters
        ----------
        target_value : float or Engineer string
            The target resistor value to compare against
        base : float, optional
            Base for logarithmic calculation. Default is 10.0 for orders of magnitude.
            Use 2.0 for powers of 2, e for natural logarithm, etc.
        """
        self.target_value = normalize_numeric(target_value)
        self.base = float(base)
        
        if self.target_value <= 0:
            raise ValueError("Target value must be positive")
        if self.base <= 0 or self.base == 1:
            raise ValueError("Base must be positive and not equal to 1")
    
    def __call__(self, resistor_value):
        """
        Evaluate how far a resistor value is from the target value.
        
        Parameters
        ----------
        resistor_value : float or Engineer string
            The resistor value to evaluate
            
        Returns
        -------
        float
            Absolute difference in logarithmic units (e.g., orders of magnitude).
            0 means exact match, 1 means 10x different (if base=10), 
            2 means 100x different (if base=10), etc.
        """
        value = normalize_numeric(resistor_value)
        
        if value <= 0:
            return float('inf')  # Invalid resistor value
        
        # Calculate logarithmic distance
        log_ratio = abs(np.log(value / self.target_value) / np.log(self.base))
        
        return log_ratio

class ResistorPowerCostFunctor(object):
    """
    Cost functor that evaluates resistor combinations based on power dissipation
    when connected in series with a given input voltage.
    
    Returns infinite cost if any resistor exceeds maximum power rating,
    otherwise returns a cost scaled from 0 to maximum_cost based on the
    highest power dissipation among the resistors.
    """
    
    def __init__(self, input_voltage, maximum_power, maximum_cost=100.0):
        """
        Initialize the power cost functor.
        
        Parameters
        ----------
        input_voltage : float or Engineer string
            Input voltage applied across the series resistor combination
        maximum_power : float or Engineer string
            Maximum allowable power dissipation for any single resistor
        maximum_cost : float, optional
            Maximum cost value to return when power is at the limit.
            Default is 100.0.
        """
        self.input_voltage = normalize_numeric(input_voltage)
        self.maximum_power = normalize_numeric(maximum_power)
        self.maximum_cost = float(maximum_cost)
        
        if self.input_voltage < 0:
            raise ValueError("Input voltage must be non-negative")
        if self.maximum_power <= 0:
            raise ValueError("Maximum power must be positive")
        if self.maximum_cost < 0:
            raise ValueError("Maximum cost must be non-negative")
    
    def __call__(self, r1, r2):
        """
        Evaluate the power-based cost for two resistors in series.
        
        Parameters
        ----------
        r1 : float or Engineer string
            First resistor value in Ohms
        r2 : float or Engineer string
            Second resistor value in Ohms
            
        Returns
        -------
        float
            Cost value: infinity if any resistor exceeds max power,
            otherwise 0 to maximum_cost based on highest power dissipation
        """
        r1_val = normalize_numeric(r1)
        r2_val = normalize_numeric(r2)
        
        if r1_val <= 0 or r2_val <= 0:
            return float('inf')  # Invalid resistor values
        
        # Calculate total series resistance and current using functions from Resistors.py
        total_resistance = series_resistors(r1_val, r2_val)
        current = current_through_resistor(total_resistance, self.input_voltage)
        
        # Calculate power dissipated in each resistor using functions from Resistors.py
        power_r1 = power_dissipated_in_resistor_by_current(r1_val, current)
        power_r2 = power_dissipated_in_resistor_by_current(r2_val, current)
        
        # Check if either resistor exceeds maximum power
        max_power_dissipated = max(power_r1, power_r2)
        if max_power_dissipated > self.maximum_power:
            return float('inf')
        
        # Scale cost from 0 to maximum_cost based on power utilization
        power_ratio = max_power_dissipated / self.maximum_power
        return power_ratio * self.maximum_cost


def select_resistors(
    error_function: Callable[[float, float], float],
    error_cutoff: float,
    r1_options: Sequence[float],
    r2_options: Sequence[float],
    cost_functions: List[Callable[[float, float, float], float]],
    cost_cutoff: float = 100.0
) -> List[ResistorSelection]:
    """
    Select optimal resistor combinations based on error and cost criteria.
    
    Args:
        error_function: Function that computes error given (r1, r2)
        error_cutoff: Maximum acceptable error value
        r1_options: Sequence of possible R1 values
        r2_options: Sequence of possible R2 values
        cost_functions: List of cost functions taking (r1, r2, error) as arguments
        cost_cutoff: Maximum acceptable total cost value. Default is 100.0.
    
    Returns:
        List of ResistorSelection objects sorted by total cost (ascending)
    """
    results = []
    
    # Compute all combinations and their errors
    for r1, r2 in itertools.product(r1_options, r2_options):
        error = error_function(r1, r2)
        
        # Only consider combinations below error cutoff
        if error <= error_cutoff:
            # Compute total cost as sum of all cost functions
            total_cost = sum(cost_func(r1, r2, error) for cost_func in cost_functions)
            
            # Only consider combinations below cost cutoff
            if total_cost <= cost_cutoff:
                results.append(ResistorSelection(
                    r1=r1,
                    r2=r2,
                    error=error,
                    total_cost=total_cost
                ))
    
    # Sort by total cost (ascending)
    results.sort(key=lambda x: x.total_cost)
    
    return results
