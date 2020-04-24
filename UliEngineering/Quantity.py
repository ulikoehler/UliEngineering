#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Units and exceptions related to units.
"""
from UliEngineering.EngineerIO import EngineerIO
from UliEngineering.Units import *
import operator

__all__ = ["Quantity"]

class Quantity(object):
    """
    Represents a value coupled with a unit,
    for example "1.23 V".
    
    Access the value by using .value    
    Access the unit by using .unit
    """
    def __init__(self, value, unit=None, io=EngineerIO.instance):
        """
        Either:
           Initialize with a string (using EngineerIO parsing): Quantity("1.23 V")
        Or:
           Initialize with a value and a unit ()
        """
        self._io = io
        if isinstance(value, (str, bytes)) and unit is None:
            # Parse !
            result = io.normalize(value)
            if not result.unit:
                raise ValueError(f"Can't parse '{value}' into a Quantity: Quantities can only be constructed if there is a unit present. If that string contains a unit, it might have been confused with a prefix - consider constructing the Quantity explicitly!")
            self.value = result.value
            self.unit = result.unit
        else: # Do not parse
            self.value = value
            self.unit = unit
    
    def __eq__(self, other):
        """
        Quantity equality operator.
        Quantities are equal to:
            - Quantity instances with the same value and unit
            - Strings that are exactly equal to this quantity's representation
            - Other objects that are equal to the quantity's value
        """
        if isinstance(other, Quantity):
            return self.value == other.value and self.unit == other.unit
        elif isinstance(other, str):
            return self.__repr__() == other
        else: # Other objects that are equal to the quantity's value
            return other == self.value
    
    def __compare(self, other, op):
        if isinstance(other, Quantity): # Quantities: Compare only if units match
            if self.unit != other.unit:
                raise InvalidUnitCombinationException(f"Can't compare Quantities with different units: {self} and {other}")
            return op(self.value, other.value)
        else: # Non-Quantities: Compare by vlaue
            return op(self.value, other)
    
    def __lt__(self, other):
        print("XXXS")
        return self.__compare(other, operator.lt)

    def __le__(self, other):
        return self.__compare(other, operator.le)
        
    def __gt__(self, other):
        return self.__compare(other, operator.gt)

    def __ge__(self, other):
        return self.__compare(other, operator.ge)

    def __repr__(self):
        return self._io.format(self.value, self.unit)
    
    def __abs__(self):
        return Quantity(abs(self.value), self.unit, self._io)