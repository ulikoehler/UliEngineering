#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import inspect
from typing import Annotated, Any, Callable, Optional, get_args, get_origin, get_type_hints

from UliEngineering.EngineerIO import EngineerIO
from UliEngineering.Units import InvalidUnitInContextException, Unit


Normalizer = Callable[[Any], Any]


def _normalize_length_arg(value):
    from UliEngineering.EngineerIO.Length import normalize_length
    return normalize_length(value)


def _normalize_area_arg(value):
    from UliEngineering.EngineerIO.Area import normalize_area
    return normalize_area(value)


def _normalize_volume_arg(value):
    from UliEngineering.EngineerIO.Volume import normalize_volume
    return normalize_volume(value)


_UNIT_NORMALIZER_MAP = {
    "m": _normalize_length_arg,
    "m²": _normalize_area_arg,
    "m^2": _normalize_area_arg,
    "m³": _normalize_volume_arg,
    "m^3": _normalize_volume_arg,
}


def returns_unit(unit):
    """
    Decorator to annotate a function with a custom return unit string.

    Usage: @returns_unit("A").
    """
    def decorator(fn):
        fn._returns_unit = unit
        return fn
    return decorator

def _normalize_scalar_arg(arg, normalizer):
    if isinstance(arg, tuple):
        return tuple(normalizer(value) for value in arg)
    if isinstance(arg, list):
        return [normalizer(value) for value in arg]
    return normalizer(arg)


def _wrap_registered_unit_normalizer(normalizer, unit):
    def wrapped(value):
        try:
            return normalizer(value)
        except InvalidUnitInContextException:
            raise
        except Exception as exc:
            raise InvalidUnitInContextException(
                f"Invalid unit: Expected {unit.unit} in source string '{value}'"
            ) from exc
    return wrapped


def _extract_normalizer_from_annotation(annotation, instance: EngineerIO):
    if annotation is inspect.Parameter.empty:
        return instance.normalize_numeric

    metadata = ()
    if get_origin(annotation) is Annotated:
        _, *metadata = get_args(annotation)
    elif isinstance(annotation, Unit) or callable(annotation):
        metadata = (annotation,)

    unit_metadata = [meta for meta in metadata if isinstance(meta, Unit)]
    converter_metadata = [meta for meta in metadata if callable(meta)]

    if unit_metadata and converter_metadata:
        raise TypeError("Use either a unit or a converter callable in Annotated metadata, not both")
    if len(unit_metadata) > 1:
        raise TypeError("Only one unit metadata entry is supported per argument")
    if len(converter_metadata) > 1:
        raise TypeError("Only one converter callable is supported per argument")

    if unit_metadata:
        unit = unit_metadata[0]
        registered_normalizer = _UNIT_NORMALIZER_MAP.get(unit.unit)
        if registered_normalizer is not None:
            return _wrap_registered_unit_normalizer(registered_normalizer, unit)
        return lambda value: instance.normalize_numeric_verify_unit(value, unit)
    if converter_metadata:
        return converter_metadata[0]
    return instance.normalize_numeric


def _build_param_normalizers(func, sig, exclude_set, instance: EngineerIO):
    type_hints = get_type_hints(func, globalns=func.__globals__, include_extras=True)
    param_normalizers = {}
    for name, param in sig.parameters.items():
        if name in exclude_set:
            continue
        annotation = type_hints.get(name, param.annotation)
        param_normalizers[name] = _extract_normalizer_from_annotation(annotation, instance)
    return param_normalizers


def normalize_args(func=None, *, exclude=None, instance:Optional[EngineerIO] = None):
    """
    Decorator that normalizes arguments before calling the wrapped function.

    Parameters can declare how they should be normalized using type annotations.
    The most pythonic form is ``typing.Annotated`` metadata:

    - ``Annotated[float, Hz]`` verifies the unit and scales to the canonical unit
    - ``Annotated[float, normalize_area]`` delegates to a specialized converter

    Parameters without metadata still use ``normalize_numeric()`` for backwards
    compatibility with existing functions.

    Parameters:
    -----------
    exclude : list of str, optional
        List of parameter names that should not be normalized

    Example:
        @normalize_args
        def add(a, b):
            return a + b

        result = add("1.5k", "2.3k")  # Will convert to add(1500.0, 2300.0)

        @normalize_args
        def rotation_speed(speed: Annotated[float, Hz]):
            return speed

        @normalize_args(exclude=['unit'])
        def calculate(value, unit):
            return value  # value is normalized, unit is left as string
    """
    if exclude is None:
        exclude = []
    exclude_set = set(exclude)
    
    if instance is None:
        instance = EngineerIO.instance()

    def decorator(func):
        sig = inspect.signature(func)
        param_normalizers = _build_param_normalizers(func, sig, exclude_set, instance)

        new_params = []
        for param in sig.parameters.values():
            if param.name not in exclude_set and param.default != inspect.Parameter.empty:
                try:
                    normalized_default = _normalize_scalar_arg(param.default, param_normalizers[param.name])
                    new_param = param.replace(default=normalized_default)
                except Exception:
                    new_param = param
            else:
                new_param = param
            new_params.append(new_param)

        new_sig = sig.replace(parameters=new_params)

        def wrapper(*args, **kwargs):
            param_names = list(sig.parameters.keys())

            normalized_args = []
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else None
                if param_name in exclude_set:
                    normalized_args.append(arg)
                else:
                    normalizer = param_normalizers.get(param_name, instance.normalize_numeric)
                    normalized_args.append(_normalize_scalar_arg(arg, normalizer))
            normalized_args = tuple(normalized_args)

            normalized_kwargs = {}
            for key, value in kwargs.items():
                if key in exclude_set:
                    normalized_kwargs[key] = value
                else:
                    normalizer = param_normalizers.get(key, instance.normalize_numeric)
                    normalized_kwargs[key] = _normalize_scalar_arg(value, normalizer)

            bound_args = new_sig.bind(*normalized_args, **normalized_kwargs)
            bound_args.apply_defaults()

            return func(*bound_args.args, **bound_args.kwargs)

        functools.update_wrapper(wrapper, func)
        wrapper.__signature__ = new_sig
        wrapper._returns_unit = getattr(func, "_returns_unit", None)

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def normalize_numeric_args(func=None, *, exclude=None, instance:Optional[EngineerIO] = None):
    """
    Decorator that applies normalize_numeric to all arguments (args & kwargs)
    of the decorated function before calling it.
    """
    if exclude is None:
        exclude = []
    exclude_set = set(exclude)

    if instance is None:
        instance = EngineerIO.instance()

    def decorator(func):
        sig = inspect.signature(func)

        new_params = []
        for param in sig.parameters.values():
            if param.name not in exclude_set and param.default != inspect.Parameter.empty and isinstance(param.default, str):
                try:
                    normalized_default = instance.normalize_numeric(param.default)
                    new_param = param.replace(default=normalized_default)
                except Exception:
                    new_param = param
            else:
                new_param = param
            new_params.append(new_param)

        new_sig = sig.replace(parameters=new_params)

        def wrapper(*args, **kwargs):
            param_names = list(sig.parameters.keys())

            normalized_args = []
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else None
                if param_name in exclude_set:
                    normalized_args.append(arg)
                else:
                    normalized_args.append(instance.normalize_numeric(arg))
            normalized_args = tuple(normalized_args)

            normalized_kwargs = {}
            for key, value in kwargs.items():
                if key in exclude_set:
                    normalized_kwargs[key] = value
                else:
                    normalized_kwargs[key] = instance.normalize_numeric(value)

            bound_args = new_sig.bind(*normalized_args, **normalized_kwargs)
            bound_args.apply_defaults()

            return func(*bound_args.args, **bound_args.kwargs)

        functools.update_wrapper(wrapper, func)
        wrapper.__signature__ = new_sig
        wrapper._returns_unit = getattr(func, "_returns_unit", None)

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)