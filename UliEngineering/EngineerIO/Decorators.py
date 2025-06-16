#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import inspect
from typing import Optional

from UliEngineering.EngineerIO import EngineerIO


def returns_unit(unit):
    """
    Decorator to annotate a function with a custom return unit string.
    Usage: @returns_unit("A")
    """
    def decorator(fn):
        fn._returns_unit = unit
        return fn
    return decorator

def normalize_numeric_args(func=None, *, exclude=None, instance:Optional[EngineerIO] = None):
    """
    Decorator that applies normalize_numeric to all arguments (args & kwargs) 
    of the decorated function before calling it.
    
    This allows functions to accept engineer notation strings and automatically
    convert them to numeric values.
    
    Parameters:
    -----------
    exclude : list of str, optional
        List of parameter names that should not be normalized
    
    Example:
        @normalize_numeric_args
        def add(a, b):
            return a + b
        
        result = add("1.5k", "2.3k")  # Will convert to add(1500.0, 2300.0)
        
        @normalize_numeric_args(exclude=['unit'])
        def calculate(value, unit):
            return value  # value is normalized, unit is left as string
    """
    if exclude is None:
        exclude = []
    exclude_set = set(exclude)
    
    if instance is None:
        instance = EngineerIO.instance()
    
    def decorator(func):
        # Get the function signature
        sig = inspect.signature(func)
        
        # Create new parameters with normalized default values
        new_params = []
        for param in sig.parameters.values():
            if param.name not in exclude_set and param.default != inspect.Parameter.empty and isinstance(param.default, str):
                # Normalize string default values
                try:
                    normalized_default = instance.normalize_numeric(param.default)
                    new_param = param.replace(default=normalized_default)
                except:
                    # If normalization fails, keep the original default
                    new_param = param
            else:
                new_param = param
            new_params.append(new_param)
        
        # Create new signature with normalized defaults
        new_sig = sig.replace(parameters=new_params)

        def wrapper(*args, **kwargs):
            # Get parameter names from signature
            param_names = list(sig.parameters.keys())
            
            # Normalize positional arguments (skip excluded ones)
            normalized_args = []
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else None
                if param_name in exclude_set:
                    normalized_args.append(arg)
                else:
                    normalized_args.append(instance.normalize_numeric(arg))
            normalized_args = tuple(normalized_args)
            
            # Normalize keyword arguments (skip excluded ones)
            normalized_kwargs = {}
            for key, value in kwargs.items():
                if key in exclude_set:
                    normalized_kwargs[key] = value
                else:
                    normalized_kwargs[key] = instance.normalize_numeric(value)
            
            # Bind arguments to new signature to get all parameters with defaults applied
            bound_args = new_sig.bind(*normalized_args, **normalized_kwargs)
            bound_args.apply_defaults()
            
            # Call the original function with all normalized arguments (including defaults)
            return func(*bound_args.args, **bound_args.kwargs)
        
        # Preserve function metadata
        functools.update_wrapper(wrapper, func)
        wrapper.__signature__ = new_sig
        wrapper._returns_unit = getattr(func, "_returns_unit", None)
        
        return wrapper
    
    # Handle both @normalize_numeric_args and @normalize_numeric_args(exclude=[...])
    if func is None:
        return decorator
    else:
        return decorator(func)