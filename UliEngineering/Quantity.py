#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Units and exceptions related to units.
"""
from UliEngineering.EngineerIO import EngineerIO
from UliEngineering.Units import *
import operator
import math

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
            - Quantity instances with the same value and unit (value is compared using math.isclose)
            - Strings that are exactly equal to this quantity's representation
            - Other objects that are equal to the quantity's value
        """
        if isinstance(other, Quantity):
            return math.isclose(self.value, other.value) and self.unit == other.unit
        elif isinstance(other, str):
            return self.__repr__() == other
        else: # Other objects that are equal to the quantity's value
            return other == self.value
    
    def __compare(self, other, op):
        """Perform a comparison on a Quantity, e.g. 'less than'"""
        if isinstance(other, Quantity): # Quantities: Compare only if units match
            if self.unit != other.unit:
                raise InvalidUnitCombinationException(f"Can't compare Quantities with different units: {self} and {other}")
            return op(self.value, other.value)
        else: # Non-Quantities: Compare by value
            return op(self.value, other)
    
    def __perform_addsub(self, other, op):
        """Perform an operation on a quantity (e.g. addition)"""
        if isinstance(other, Quantity): # Quantities: Compare only if units match
            if self.unit != other.unit:
                raise InvalidUnitCombinationException(f"Can't perform {op.__name__} operation on Quantities with different units: {self} and {other}")
            return Quantity(op(self.value, other.value), self.unit)
        else: # Non-Quantities: Compare by value
            return Quantity(op(self.value, other), self.unit)
    
    def __repr__(self):
        return self._io.format(self.value, self.unit)

    def __lt__(self, other):
        return self.__compare(other, operator.lt)

    def __le__(self, other):
        return self.__compare(other, operator.le)
        
    def __gt__(self, other):
        return self.__compare(other, operator.gt)

    def __ge__(self, other):
        return self.__compare(other, operator.ge)

    def __add__(self, other):
        return self.__perform_addsub(other, operator.add)

    def __sub__(self, other):
        return self.__perform_addsub(other, operator.sub)

    def __mul__(self, other):
        return self.__perform_arithmetic(other, operator.mul)
    
    def __truediv__(self, other):
        return self.__perform_arithmetic(other, operator.truediv)

    def __floordiv__(self, other):
        return self.__perform_arithmetic(other, operator.floordiv)

    def __mod__(self, other):
        return self.__perform_arithmetic(other, operator.mod)

    def __divmod__(self, other):
        return (self // other, self % other)

    def __pow__(self, other):
        return self.__perform_arithmetic(other, operator.pow)

    def __lshift__(self, other):
        return self.__perform_arithmetic(other, operator.lshift)

    def __rshift__(self, other):
        return self.__perform_arithmetic(other, operator.rshift)

    def __and__(self, other):
        return self.__perform_arithmetic(other, operator.and_)

    def __xor__(self, other):
        return self.__perform_arithmetic(other, operator.xor)

    def __or__(self, other):
        return self.__perform_arithmetic(other, operator.or_)
    
    def __abs__(self):
        return Quantity(abs(self.value), self.unit, self._io)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __rfloordiv__(self, other):
        return self.__floordiv__(other)

    def __rmod__(self, other):
        return self.__mod__(other)

    def __rdivmod__(self, other):
        return self.__divmod__(other)

    def __rpow__(self, other):
        return self.__pow__(other)

    def __rlshift__(self, other):
        return self.__lshift__(other)

    def __rrshift__(self, other):
        return self.__rshift__(other)

    def __rand__(self, other):
        return self.__and__(other)

    def __rxor__(self, other):
        return self.__xor__(other)
        
    def __ror__(self, other):
        return self.__or__(other)