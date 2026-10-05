from steelo.adapters.dataprocessing.postprocessing.analysis_scope import (
    EUROPE_ANALYSIS_ISO3,
    EU27_ISO3,
)


def test_europe_analysis_scope_is_eu27_plus_efta_and_uk() -> None:
    assert len(EU27_ISO3) == 27
    assert EUROPE_ANALYSIS_ISO3 - EU27_ISO3 == {"ISL", "LIE", "NOR", "CHE", "GBR"}
    assert not ({"ALB", "BIH", "MDA", "MKD", "MNE", "SRB", "TUR", "UKR"} & EUROPE_ANALYSIS_ISO3)
