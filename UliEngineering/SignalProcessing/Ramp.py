#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal processing module for generating ramp signals with configurable smoothing (acceleration).
"""
import numpy as np
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Exceptions import OperationImpossibleException

__all__ = ["periodic_ramp"]

def periodic_ramp(frequency, samplerate, amplitude=1.0, offset=0.0, 
                  rise_time=None, fall_time=None, high_time=None, 
                  acceleration=None, length=1.0, phaseshift=0.0,
                  continuity="C1"):
    """
    Generate a periodic ramp (trapezoidal) signal with smoothed corners (knees) defined by acceleration.
    
    The signal consists of 4 segments per period:
    1. Rise (Low -> High)
    2. High Hold
    3. Fall (High -> Low)
    4. Low Hold
    
    The sum of rise_time, high_time, fall_time, and low_time (calculated) must equal 1/frequency.
    
    :param frequency: The fundamental frequency of the signal in Hz.
    :param samplerate: The sampling rate in Hz.
    :param amplitude: The peak-to-peak amplitude. The signal swings from offset to offset + amplitude.
    :param offset: The low level of the signal.
    :param rise_time: Duration of the rising edge in seconds. If None, defaults to 10% of period.
    :param fall_time: Duration of the falling edge in seconds. If None, defaults to rise_time.
    :param high_time: Duration of the high hold in seconds. If None, defaults to (Period - Rise - Fall) / 2.
    :param acceleration: The acceleration used for smoothing the corners. 
                         Units are AmplitudeUnits / s^2.
                         If None, infinite acceleration is assumed (linear ramps, sharp corners).
                         Must be high enough to achieve the amplitude within the rise/fall times.
    :param length: Length of the generated signal in seconds.
    :param phaseshift: Initial phase shift in degrees.
    :param continuity: The type of continuity to guarantee.
                       "C0": Linear ramp (infinite acceleration at corners). Ignores acceleration parameter.
                       "C1": Continuous velocity (constant acceleration profile). Default if acceleration is provided.
                       "C2": Continuous acceleration (sinusoidal acceleration profile).
    
    :return: A numpy array containing the signal.
    """
    frequency = normalize_numeric(frequency)
    samplerate = normalize_numeric(samplerate)
    amplitude = normalize_numeric(amplitude)
    offset = normalize_numeric(offset)
    length = normalize_numeric(length)
    phaseshift = normalize_numeric(phaseshift)
    
    if continuity not in ["C0", "C1", "C2"]:
        raise ValueError(f"Continuity must be 'C0', 'C1', or 'C2', not '{continuity}'")

    period = 1.0 / frequency
    
    # Default timings
    if rise_time is None:
        rise_time = 0.1 * period
    else:
        rise_time = normalize_numeric(rise_time)
        
    if fall_time is None:
        fall_time = rise_time
    else:
        fall_time = normalize_numeric(fall_time)
        
    if high_time is None:
        remaining = period - rise_time - fall_time
        if remaining < 0:
            raise OperationImpossibleException(f"Rise ({rise_time}) + Fall ({fall_time}) times exceed period ({period})")
        high_time = remaining / 2.0
    else:
        high_time = normalize_numeric(high_time)
        
    low_time = period - rise_time - fall_time - high_time
    
    if low_time < -1e-9: # Tolerance for float errors
        raise OperationImpossibleException(f"Sum of segments exceeds period: Rise={rise_time}, High={high_time}, Fall={fall_time}, Period={period}")
    
    # Ensure low_time is not negative due to float precision
    low_time = max(0.0, low_time)
    
    # Validate acceleration
    # Min acceleration required for a rise/fall of duration T and amplitude A:
    # The most efficient profile is triangular velocity (bang-bang acceleration).
    # Distance = A. Time = T.
    # A = 2 * (0.5 * a * (T/2)^2) = a * T^2 / 4  => a_min = 4 * A / T^2
    
    if acceleration is not None and continuity != "C0":
        acceleration = normalize_numeric(acceleration)
        
        if continuity == "C2":
            # For C2 (sinusoidal acceleration), the efficiency is lower.
            # t1^2 - D*t1 + (pi*H)/(2*a) = 0
            # Discriminant D^2 - 2*pi*H/a >= 0 => a >= 2*pi*H/D^2
            factor = 2 * np.pi
        else: # C1
            factor = 4.0
            
        min_acc_rise = factor * amplitude / (rise_time ** 2) if rise_time > 0 else 0
        min_acc_fall = factor * amplitude / (fall_time ** 2) if fall_time > 0 else 0
        
        if acceleration < min_acc_rise:
            raise OperationImpossibleException(f"Acceleration {acceleration} is too low for rise time {rise_time} with {continuity} continuity. Min required: {min_acc_rise}")
        if acceleration < min_acc_fall:
            raise OperationImpossibleException(f"Acceleration {acceleration} is too low for fall time {fall_time} with {continuity} continuity. Min required: {min_acc_fall}")
            
    # Generate time array
    num_samples = int(length * samplerate)
    t = np.arange(num_samples) / samplerate
    
    # Apply phase shift
    t_shifted = t + (phaseshift / 360.0) * period
    
    # Map to [0, Period)
    t_cycle = np.mod(t_shifted, period)
    
    # Compute signal
    # We can use piecewise, or compute indices
    
    # Pre-calculate transition times
    t_end_rise = rise_time
    t_end_high = t_end_rise + high_time
    t_end_fall = t_end_high + fall_time
    # t_end_low = period
    
    y = np.zeros_like(t)
    
    # Segment 1: Rise
    mask_rise = (t_cycle < t_end_rise)
    if np.any(mask_rise):
        y[mask_rise] = _compute_edge(t_cycle[mask_rise], rise_time, amplitude, acceleration, continuity, rising=True)
        
    # Segment 2: High
    mask_high = (t_cycle >= t_end_rise) & (t_cycle < t_end_high)
    y[mask_high] = amplitude
    
    # Segment 3: Fall
    mask_fall = (t_cycle >= t_end_high) & (t_cycle < t_end_fall)
    if np.any(mask_fall):
        # Time relative to start of fall
        t_rel = t_cycle[mask_fall] - t_end_high
        y[mask_fall] = _compute_edge(t_rel, fall_time, amplitude, acceleration, continuity, rising=False)
        
    # Segment 4: Low (already 0 initialized, but add offset later)
    
    return y + offset

def _compute_edge(t, duration, amplitude, acceleration, continuity, rising=True):
    """
    Compute the value of a rising or falling edge at times t (0 to duration).
    """
    if duration <= 0:
        return np.full_like(t, amplitude if rising else 0.0)
        
    if acceleration is None or continuity == "C0":
        # Linear
        val = (t / duration) * amplitude
    else:
        # S-curve
        if continuity == "C2":
            # Sinusoidal acceleration
            # t1^2 - D*t1 + (pi*H)/(2*a) = 0
            term = duration**2 - (2 * np.pi * amplitude) / acceleration
        else: # C1
            # Constant acceleration
            # t1^2 - D*t1 + H/a = 0
            term = duration**2 - 4 * amplitude / acceleration
            
        if term < 0:
             # Should have been caught by validation, but handle float issues
             t1 = duration / 2.0
        else:
            t1 = (duration - np.sqrt(term)) / 2.0
            
        t2 = duration - t1 # End of constant velocity phase
        
        val = np.zeros_like(t)
        
        # Phase 1: Acceleration
        mask1 = t < t1
        if continuity == "C2":
            # x(t) = (a*t1/pi) * (t - (t1/pi)*sin(pi*t/t1))
            # Note: acceleration parameter 'a' is used as peak acceleration
            # But wait, we solved for t1 assuming 'a' is peak acceleration.
            # Let's verify:
            # x(t1) = a*t1^2/pi.
            # v(t1) = 2*a*t1/pi.
            # This matches the derivation.
            t_phase1 = t[mask1]
            val[mask1] = (acceleration * t1 / np.pi) * (t_phase1 - (t1 / np.pi) * np.sin(np.pi * t_phase1 / t1))
        else:
            val[mask1] = 0.5 * acceleration * t[mask1]**2
        
        # Phase 2: Constant velocity
        mask2 = (t >= t1) & (t < t2)
        if continuity == "C2":
            x_t1 = acceleration * t1**2 / np.pi
            v_max = 2 * acceleration * t1 / np.pi
        else:
            x_t1 = 0.5 * acceleration * t1**2
            v_max = acceleration * t1
            
        val[mask2] = x_t1 + v_max * (t[mask2] - t1)
        
        # Phase 3: Deceleration
        mask3 = t >= t2
        # Symmetry: x(t) = A - x_accel(D - t)
        t_phase3_rem = duration - t[mask3] # Time remaining
        if continuity == "C2":
            x_rem = (acceleration * t1 / np.pi) * (t_phase3_rem - (t1 / np.pi) * np.sin(np.pi * t_phase3_rem / t1))
            val[mask3] = amplitude - x_rem
        else:
            val[mask3] = amplitude - 0.5 * acceleration * t_phase3_rem**2
        
    if not rising:
        return amplitude - val
    return val
