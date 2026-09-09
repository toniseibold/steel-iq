from pathlib import Path

import pandas as pd
import pytest

from steelo.adapters.dataprocessing.postprocessing.regional_emissions_price_analysis import (
    build_summary,
    write_outputs,
)


def test_regional_metrics_are_production_weighted_and_feedstocks_are_not_duplicated(tmp_path: Path):
    results_dir = tmp_path / "results"
    run_dir = results_dir / "master_input_0" / "sim_1"
    (run_dir / "data").mkdir(parents=True)
    # An incomplete scenario must not prevent completed scenarios from being analysed.
    (results_dir / "master_input_incomplete" / "sim_1").mkdir(parents=True)
    rows = [
        {
            "year": 2030,
            "region": "Europe",
            "product": "steel",
            "furnace_group_id": "fg1",
            "production": 100.0,
            "unit_production_cost": 500.0,
            "emissions_rs-inspired_direct_ghg": 100.0,
            "emissions_rs-inspired_indirect_ghg": 50.0,
        },
        {
            "year": 2030,
            "region": "Europe",
            "product": "steel",
            "furnace_group_id": "fg1",  # second feedstock row for the same furnace
            "production": 100.0,
            "unit_production_cost": 500.0,
            "emissions_rs-inspired_direct_ghg": 100.0,
            "emissions_rs-inspired_indirect_ghg": 50.0,
        },
        {
            "year": 2030,
            "region": "Europe",
            "product": "steel",
            "furnace_group_id": "fg2",
            "production": 300.0,
            "unit_production_cost": 700.0,
            "emissions_rs-inspired_direct_ghg": 600.0,
            "emissions_rs-inspired_indirect_ghg": 300.0,
        },
        {
            "year": 2030,
            "region": "CIS",
            "product": "iron",
            "furnace_group_id": "fg3",
            "production": 200.0,
            "unit_production_cost": 300.0,
            "emissions_rs-inspired_direct_ghg": 300.0,
            "emissions_rs-inspired_indirect_ghg": 100.0,
        },
    ]
    pd.DataFrame(rows).to_csv(run_dir / "post_processed_test.csv", index=False)
    pd.DataFrame([{"year": 2030, "steel_price_usd_per_t": 800.0, "iron_price_usd_per_t": 450.0}]).to_csv(
        run_dir / "data" / "market_prices_2030_2050.csv", index=False
    )

    summary = build_summary(results_dir, [2030])
    steel = summary[(summary["region"] == "Europe") & (summary["product"] == "steel")].iloc[0]
    iron = summary[(summary["region"] == "CIS") & (summary["product"] == "iron")].iloc[0]

    assert steel["production_mt"] == pytest.approx(400 / 1e6)
    assert steel["avg_ghg_intensity_tco2e_per_t"] == pytest.approx(1050 / 400)
    assert steel["avg_unit_production_cost_usd_per_t"] == pytest.approx(650.0)
    assert steel["global_market_price_usd_per_t"] == pytest.approx(800.0)
    assert iron["avg_ghg_intensity_tco2e_per_t"] == pytest.approx(2.0)

    created = write_outputs(summary, tmp_path / "output", [2030])
    assert len(created) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in created)
