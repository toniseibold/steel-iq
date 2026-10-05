"""Shared geographic scope for European post-simulation analysis."""

EU27_ISO3 = frozenset(
    {
        "AUT",
        "BEL",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "SWE",
    }
)

EUROPE_ANALYSIS_ISO3 = EU27_ISO3 | frozenset({"ISL", "LIE", "NOR", "CHE", "GBR"})
EUROPE_ANALYSIS_LABEL = "EU27 + EFTA + UK"
EUROPE_ANALYSIS_DESCRIPTION = "EU27 plus Iceland, Liechtenstein, Norway, Switzerland and the United Kingdom"
