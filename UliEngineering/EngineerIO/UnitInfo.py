#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit information dataclass for UliEngineering
"""
from dataclasses import dataclass, field
from typing import Dict, List, Union

@dataclass
class UnitAlias:
    """
    Represents a mapping of aliases to a canonical unit.
    This is used for units that don't have their own conversion factor
    but are alternative names for existing units.
    
    Attributes:
    -----------
    canonical : str
        The canonical unit that these aliases map to
    aliases : List[str]
        List of alternative names that should map to the canonical unit
    """
    canonical: str
    aliases: List[str] = field(default_factory=list)
    
    def matches_alias(self, alias_string: str) -> bool:
        """
        Check if the given string is one of the aliases
        
        Parameters:
        -----------
        alias_string : str
            The alias string to check
            
        Returns:
        --------
        bool
            True if the string is in the aliases list
        """
        return alias_string in self.aliases

@dataclass
class UnitInfo:
    """
    Represents information about a unit including its canonical form,
    aliases, and conversion factor.
    
    Attributes:
    -----------
    canonical : str
        The canonical/standard form of the unit (e.g., 'Ω', 's', 'A')
    factor : float
        The multiplication factor to convert to base SI units
    aliases : List[str]
        List of alternative representations for this unit
    """
    canonical: str
    factor: float = field(default=1.0)
    aliases: List[str] = field(default_factory=list)
    
    def matches(self, unit_string: str) -> bool:
        """
        Check if the given unit string matches this unit (canonical or alias)
        
        Parameters:
        -----------
        unit_string : str
            The unit string to check
            
        Returns:
        --------
        bool
            True if the unit string matches this unit
        """
        return unit_string == self.canonical or unit_string in self.aliases
    
    def get_all_representations(self) -> List[str]:
        """
        Get all possible representations of this unit (canonical + aliases)
        
        Returns:
        --------
        List[str]
            List containing canonical form and all aliases
        """
        return [self.canonical] + self.aliases


@dataclass
class EngineerIOConfiguration:
    units: List[Union[UnitInfo, UnitAlias]] = field()
    unit_prefixes: List[str] = field()
    si_prefix_map: Dict[str, float] = field()
    
    @classmethod
    def default(cls) -> 'EngineerIOConfiguration':
        """
        Returns a default configuration with standard units and prefixes.
        
        Returns:
        --------
        EngineerIOConfiguration
            Default configuration instance
        """
        # Late import to avoid circular dependency issuess
        from UliEngineering.EngineerIO.Defaults import default_si_prefix_map, default_unit_infos, default_unit_prefixes
        return cls(
            units=default_unit_infos(),
            unit_prefixes=default_unit_prefixes(),
            si_prefix_map=default_si_prefix_map()
        )
