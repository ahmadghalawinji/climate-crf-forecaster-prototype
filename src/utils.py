import re
import pandas as pd

# ISO3 → country name mapping (subset covering HEPA countries)
ISO3_TO_NAME = {
    "AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria", "HRV": "Croatia",
    "CYP": "Cyprus", "CZE": "Czechia", "DNK": "Denmark", "EST": "Estonia",
    "FIN": "Finland", "FRA": "France", "DEU": "Germany", "GRC": "Greece",
    "HUN": "Hungary", "ISL": "Iceland", "IRL": "Ireland", "ITA": "Italy",
    "LVA": "Latvia", "LIE": "Liechtenstein", "LTU": "Lithuania", "LUX": "Luxembourg",
    "MLT": "Malta", "MDA": "Moldova", "MCO": "Monaco", "MNE": "Montenegro",
    "NLD": "Netherlands", "MKD": "North Macedonia", "NOR": "Norway", "POL": "Poland",
    "PRT": "Portugal", "ROU": "Romania", "SMR": "San Marino", "SRB": "Serbia",
    "SVK": "Slovakia", "SVN": "Slovenia", "ESP": "Spain", "SWE": "Sweden",
    "CHE": "Switzerland", "TUR": "Turkey", "UKR": "Ukraine", "GBR": "United Kingdom",
    "ALB": "Albania", "AND": "Andorra", "ARM": "Armenia", "AZE": "Azerbaijan",
    "BLR": "Belarus", "BIH": "Bosnia and Herzegovina", "GEO": "Georgia",
    "KAZ": "Kazakhstan", "XKX": "Kosovo", "RUS": "Russia",
}

COUNTRY_NORMALIZE = {
    "united states of america": "United States",
    "russian federation": "Russia",
    "republic of korea": "South Korea",
    "korea, rep.": "South Korea",
    "czech republic": "Czechia",
    "türkiye": "Turkey",
    "slovak republic": "Slovakia",
    "north macedonia": "North Macedonia",
}

def iso3_to_country(code):
    return ISO3_TO_NAME.get(str(code).upper(), code)

def normalize_country_name(name):
    if not isinstance(name, str):
        return name
    n = name.strip()
    return COUNTRY_NORMALIZE.get(n.lower(), n)