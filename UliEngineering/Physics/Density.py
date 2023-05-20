#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.EngineerIO import normalize_numeric, Unit

__all__ = ["Densities", "density_by_volume_and_weight"]

"""
Pre-defined densities for various materials in kg/m³
"""
Densities = {
    # Various (pure) metals
    "Aluminium": 2700., # Source: https://en.wikipedia.org/wiki/Aluminium
    "Titanium": 4506., # Source: https://en.wikipedia.org/wiki/Titanium
    # Various alloys
    "Brass-CuZn10": 8800., # Source: https://www.metal-rolling-services.com/cuzn10-en
    "Brass-CuZn10": 8750., # Source: https://www.metal-rolling-services.com/cuzn15-en
    "Brass-CuZn30": 8550., # Source: https://www.metal-rolling-services.com/cuzn30-en
    "Brass-CuZn33": 8500., # Source: https://www.metal-rolling-services.com/cuzn33-en
    "Brass-CuZn36": 8450., # Source: https://www.metal-rolling-services.com/cuzn36-en
    "Brass-CuZn37": 8440., # Source: https://www.metal-rolling-services.com/cuzn37-en
    # Steel
    "S235JR": 7850., # Source: https://de.materials4me.com/media/pdf/e7/7d/30/Werkstoffdatenblatt_zum_Werkstoff_S235JR.pdf
    # Stainless steels
    "1.4003": 7700., # Source: https://www.edelstahl-rostfrei.de/fileadmin/user_upload/ISER/downloads/MB_822.pdf
    "1.4016": 7700., # Source: (same as above)
    "1.4511": 7700., # Source: (same as above)
    "1.4521": 7700., # Source: (same as above)
    "1.4509": 7700., # Source: (same as above)
    "1.4310": 7900., # Source: (same as above)
    "1.4318": 7900., # Source: (same as above)
    "1.4307": 7900., # Source: (same as above)
    "1.4305": 7900., # Source: (same as above)
    "1.4541": 7900., # Source: (same as above)
    "1.4401": 8000., # Source: (same as above)
    "1.4571": 8000., # Source: (same as above)
    "1.4437": 8000., # Source: (same as above)
    "1.4435": 8000., # Source: (same as above)
    "1.4439": 8000., # Source: (same as above)
    "1.4567": 7900., # Source: (same as above)
    "1.4539": 8000., # Source: (same as above)
    "1.4578": 8000., # Source: (same as above)
    "1.4547": 8000., # Source: (same as above)
    "1.4529": 8100., # Source: (same as above)
    "1.4565": 8000., # Source: (same as above)
    "1.4362": 7800., # Source: (same as above)
    "1.4462": 7800., # Source: (same as above)
    "1.4301": 7900., # Source: https://www.dew-stahl.com/fileadmin/files/dew-stahl.com/documents/Publikationen/Werkstoffdatenblaetter/RSH/1.4301_de.pdf
    "1.4404": 8000., # Source: https://www.dew-stahl.com/fileadmin/files/dew-stahl.com/documents/Publikationen/Werkstoffdatenblaetter/RSH/1.4404_en.pdf
    # Various types of polymers
    "POM": 1410., # Source: https://www.polyplastics.com/Gidb/GradeInfoDownloadAction.do?gradeId=1771&fileNo=1&langId=1&_LOCALE=ENGLISH
    "PTFE": 2200., # Source: https://en.wikipedia.org/wiki/Polytetrafluoroethylene
    "PEEK": 1320., # Source: https://en.wikipedia.org/wiki/Polyether_ether_ketone
}

def density_by_volume_and_weight(volume, weight) -> Unit("kg/m³"):
    """
    Calculates the density of a material by its volume and weight.

    Parameters
    ----------
    volume : float
        Volume of the material in m³.
    weight : float
        Weight of the material in kg.

    Returns
    -------
    float
        Density of the material in kg/m³.
    """
    weight = normalize_numeric(weight)
    volume = normalize_numeric(volume)
    return weight / volume