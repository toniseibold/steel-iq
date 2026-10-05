from pathlib import Path

import pandas as pd
import pytest

from steelo.adapters.dataprocessing.postprocessing.europe_scenario_post_analysis import ScenarioInput
from steelo.adapters.dataprocessing.postprocessing.production_cost_emissions_matrix import summarise_scenario


def test_primary_steel_adds_iron_cost_and_emissions_to_conversion_step(tmp_path: Path):
    rows = [
        {
            "year": 2050,
            "region": "Europe",
            "iso3": "DEU",
            "furnace_group_id": "iron-1",
            "product": "iron",
            "production": 100.0,
            "unit_production_cost": 200.0,
            "feedstock": "io_high",
            "demand": 150.0,
            "total_cost - material and allocation": 10_000.0,
            "emissions_rs-inspired_direct_ghg": 100.0,
            "emissions_rs-inspired_indirect_ghg": 50.0,
        },
        {
            "year": 2050,
            "region": "Europe",
            "iso3": "DEU",
            "furnace_group_id": "steel-1",
            "product": "steel",
            "production": 100.0,
            "unit_production_cost": 300.0,
            "feedstock": "pig_iron",
            "demand": 75.0,
            "total_cost - material and allocation": 15_000.0,
            "emissions_rs-inspired_direct_ghg": 20.0,
            "emissions_rs-inspired_indirect_ghg": 10.0,
        },
        {
            "year": 2050,
            "region": "Europe",
            "iso3": "DEU",
            "furnace_group_id": "steel-1",
            "product": "steel",
            "production": 100.0,
            "unit_production_cost": 300.0,
            "feedstock": "scrap",
            "demand": 25.0,
            "total_cost - material and allocation": 2_500.0,
            "emissions_rs-inspired_direct_ghg": 20.0,
            "emissions_rs-inspired_indirect_ghg": 10.0,
        },
    ]
    plants_csv = tmp_path / "post_processed_test.csv"
    pd.DataFrame(rows).to_csv(plants_csv, index=False)
    scenario = ScenarioInput(0, "E-W", "primary", tmp_path, plants_csv)

    result = summarise_scenario(scenario).set_index(["geography", "product_stage"])

    for geography in ("Global", "EU"):
        iron = result.loc[(geography, "iron")]
        primary = result.loc[(geography, "primary_steel")]
        secondary = result.loc[(geography, "secondary_steel")]
        assert iron["avg_cost_usd_per_t"] == pytest.approx(200.0)
        assert primary["avg_cost_usd_per_t"] == pytest.approx(325.0)
        assert secondary["avg_cost_usd_per_t"] == pytest.approx(225.0)
        assert primary["direct_emissions_tco2e_per_t"] == pytest.approx(1.2)
        assert secondary["direct_emissions_tco2e_per_t"] == pytest.approx(0.2)
        assert primary["direct_indirect_emissions_tco2e_per_t"] == pytest.approx(1.8)
        assert secondary["direct_indirect_emissions_tco2e_per_t"] == pytest.approx(0.3)
