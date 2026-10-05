import pandas as pd

from steelo.adapters.dataprocessing.postprocessing.route_classification import (
    DRI_COAL,
    DRI_GAS,
    DRI_HYDROGEN,
    carbon_route,
    source_route_lookup,
    technology_route_label,
)


def test_dri_route_distinguishes_hydrogen_gas_and_coal():
    assert technology_route_label("DRI", "hydrogen") == DRI_HYDROGEN
    assert technology_route_label("DRI", "natural_gas") == DRI_GAS
    assert technology_route_label("DRI", "coal") == DRI_COAL


def test_carbon_route_uses_reductant_and_explicit_ccs_technology():
    assert carbon_route("DRI", "hydrogen") == "green"
    assert carbon_route("DRI+CCS", "natural_gas") == "blue"
    assert carbon_route("BF", "coke+pci") == "grey"
    assert carbon_route("BF+CCS", "coke+pci") == "blue"


def test_source_route_lookup_keeps_one_record_per_furnace():
    plants = pd.DataFrame(
        [
            {"year": 2050, "furnace_group_id": "h2", "technology": "DRI", "chosen_reductant": "hydrogen"},
            {"year": 2050, "furnace_group_id": "h2", "technology": "DRI", "chosen_reductant": "hydrogen"},
            {"year": 2050, "furnace_group_id": "gas", "technology": "DRI", "chosen_reductant": "natural_gas"},
        ]
    )

    result = source_route_lookup(plants, 2050).set_index("furnace_group_id")

    assert len(result) == 2
    assert result.loc["h2", "technology_route"] == DRI_HYDROGEN
    assert result.loc["gas", "technology_route"] == DRI_GAS
