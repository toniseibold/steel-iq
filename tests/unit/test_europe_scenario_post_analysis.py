from pathlib import Path

import pandas as pd
import pytest

from steelo.adapters.dataprocessing.postprocessing.europe_scenario_post_analysis import (
    ScenarioInput,
    _europe_iso3,
    _steel_origin_by_year,
    aggregate_trade_partners,
    calculate_primary_iron_self_sufficiency,
    discover_scenario_inputs,
    load_europe_steel_demand,
    load_europe_technology_portfolio,
)


def test_europe_scope_does_not_expand_from_model_region(tmp_path: Path):
    path = tmp_path / "post_processed_test.csv"
    pd.DataFrame(
        [
            {"iso3": "DEU", "region": "Europe"},
            {"iso3": "GBR", "region": "Europe"},
            {"iso3": "SRB", "region": "Europe"},
        ]
    ).to_csv(path, index=False)
    scenario = ScenarioInput(0, "E-W", "primary", tmp_path, path)

    scope = _europe_iso3(scenario)

    assert {"DEU", "GBR"} <= scope
    assert "SRB" not in scope


def test_scenarios_only_use_supplied_results_directory_and_missing_run_is_skipped(tmp_path: Path):
    primary = tmp_path / "250USD"
    other_results = tmp_path / "180USD"
    for root, index in ((primary, 0), (primary, 1), (other_results, 2)):
        run = root / f"master_input_{index}" / "sim_1"
        (run / "TM").mkdir(parents=True)
        pd.DataFrame([{"year": 2050}]).to_csv(run / "post_processed_test.csv", index=False)

    scenarios = discover_scenario_inputs(primary, skip_runs=[11])

    assert [(item.index, item.title, item.source) for item in scenarios] == [
        (0, "E-W", "primary"),
        (1, "EC-W", "primary"),
    ]


def test_primary_secondary_split_conserves_furnace_production(tmp_path: Path):
    rows = [
        {
            "year": 2050,
            "region": "Europe",
            "iso3": "DEU",
            "furnace_group_id": "fg1",
            "product": "steel",
            "production": 100.0,
            "feedstock": "scrap",
            "demand": 30.0,
        },
        {
            "year": 2050,
            "region": "Europe",
            "iso3": "DEU",
            "furnace_group_id": "fg1",
            "product": "steel",
            "production": 100.0,
            "feedstock": "pig_iron",
            "demand": 70.0,
        },
    ]
    path = tmp_path / "plants.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    result = _steel_origin_by_year(path).iloc[0]

    assert result["steel_secondary_mt"] == pytest.approx(30.0 / 1e6)
    assert result["steel_primary_mt"] == pytest.approx(70.0 / 1e6)
    assert result["steel_primary_mt"] + result["steel_secondary_mt"] == pytest.approx(100.0 / 1e6)


def test_load_europe_steel_demand_uses_fixed_scope_scenario_and_units(tmp_path: Path):
    workbook = tmp_path / "master.xlsx"
    rows = [
        ["Germany", "DEU", " Crude steel consumption for forming [kt] ", "BAU", "kt", 1_000.0, 1_100.0],
        ["United Kingdom", "GBR", "Crude steel consumption for forming [kt]", "BAU", "kt", 500.0, 600.0],
        ["Serbia", "SRB", "Crude steel consumption for forming [kt]", "BAU", "kt", 900.0, 900.0],
        ["Germany", "DEU", "Crude steel consumption for forming [kt]", "Other", "kt", 5_000.0, 5_000.0],
        ["Germany", "DEU", "Total available scrap", "BAU", "kt", 8_000.0, 8_000.0],
    ]
    pd.DataFrame(
        rows,
        columns=["Country", "ISO-3 code", "Metric", "Scenario", "Unit", 2030, 2040],
    ).to_excel(workbook, sheet_name="Demand and scrap availability", index=False)

    result = load_europe_steel_demand(workbook, [2030, 2040]).set_index("year")

    assert result.loc[2030, "steel_demand_mt"] == pytest.approx(1.5)
    assert result.loc[2040, "steel_demand_mt"] == pytest.approx(1.7)


def test_primary_iron_self_sufficiency_uses_iron_and_preserves_uncapped_ratio():
    production = pd.DataFrame(
        [
            {"scenario_index": 0, "scenario_title": "E-W", "year": 2030, "iron_production_mt": 40.0},
            {"scenario_index": 0, "scenario_title": "E-W", "year": 2040, "iron_production_mt": 120.0},
        ]
    )
    demand = pd.DataFrame(
        [
            {"year": 2030, "steel_demand_mt": 100.0},
            {"year": 2040, "steel_demand_mt": 100.0},
        ]
    )

    result = calculate_primary_iron_self_sufficiency(production, demand).set_index("year")

    assert result.loc[2030, "primary_iron_to_demand_ratio"] == pytest.approx(0.4)
    assert result.loc[2030, "primary_iron_self_sufficiency_pct"] == pytest.approx(40.0)
    assert result.loc[2040, "primary_iron_to_demand_ratio"] == pytest.approx(1.2)
    assert result.loc[2040, "primary_iron_self_sufficiency_pct"] == pytest.approx(100.0)


def test_small_partner_flows_are_combined_as_rest():
    flows = pd.DataFrame(
        [
            {
                "scenario_index": 0,
                "scenario_title": "E-W",
                "year": 2050,
                "direction": "import",
                "country_from": "USA",
                "country_to": "DEU",
                "commodity": "steel",
                "volume_t": 2_000_000.0,
            },
            {
                "scenario_index": 0,
                "scenario_title": "E-W",
                "year": 2050,
                "direction": "import",
                "country_from": "CAN",
                "country_to": "DEU",
                "commodity": "steel",
                "volume_t": 400_000.0,
            },
            {
                "scenario_index": 0,
                "scenario_title": "E-W",
                "year": 2050,
                "direction": "import",
                "country_from": "MEX",
                "country_to": "DEU",
                "commodity": "steel",
                "volume_t": 600_000.0,
            },
        ]
    )

    result = aggregate_trade_partners(flows, threshold_t=1e6).set_index("country_plot")

    assert result.loc["USA", "volume_t"] == pytest.approx(2_000_000.0)
    assert result.loc["Rest", "volume_t"] == pytest.approx(1_000_000.0)


def test_technology_portfolio_counts_each_furnace_once_and_classifies_routes(tmp_path: Path):
    common = {"year": 2050, "region": "Europe", "iso3": "DEU"}
    rows = [
        {
            **common,
            "furnace_group_id": "dri-h2",
            "product": "iron",
            "technology": "DRI",
            "chosen_reductant": "hydrogen",
            "capacity": 100.0,
            "production": 80.0,
            "feedstock": "io_high",
            "demand": 90.0,
            "carbon_breakdown - co2_stored": 0.2,
            "carbon_breakdown - co2_utilised": 0.1,
        },
        {
            **common,
            "furnace_group_id": "dri-h2",
            "product": "iron",
            "technology": "DRI",
            "chosen_reductant": "hydrogen",
            "capacity": 100.0,
            "production": 80.0,
            "feedstock": "scrap",
            "demand": 10.0,
            "carbon_breakdown - co2_stored": 0.3,
            "carbon_breakdown - co2_utilised": 0.0,
        },
        {
            **common,
            "furnace_group_id": "eaf-scrap",
            "product": "steel",
            "technology": "EAF",
            "chosen_reductant": None,
            "capacity": 200.0,
            "production": 150.0,
            "feedstock": "scrap",
            "demand": 160.0,
            "carbon_breakdown - co2_stored": 0.0,
            "carbon_breakdown - co2_utilised": 0.0,
        },
        {
            **common,
            "furnace_group_id": "eaf-mixed",
            "product": "steel",
            "technology": "EAF",
            "chosen_reductant": None,
            "capacity": 300.0,
            "production": 240.0,
            "feedstock": "scrap",
            "demand": 120.0,
            "carbon_breakdown - co2_stored": 0.0,
            "carbon_breakdown - co2_utilised": 0.0,
        },
        {
            **common,
            "furnace_group_id": "eaf-mixed",
            "product": "steel",
            "technology": "EAF",
            "chosen_reductant": None,
            "capacity": 300.0,
            "production": 240.0,
            "feedstock": "pig_iron",
            "demand": 120.0,
            "carbon_breakdown - co2_stored": 0.0,
            "carbon_breakdown - co2_utilised": 0.0,
        },
        {
            "year": 2050,
            "region": "North America",
            "iso3": "USA",
            "furnace_group_id": "outside-europe",
            "product": "steel",
            "technology": "EAF",
            "chosen_reductant": None,
            "capacity": 999.0,
            "production": 999.0,
            "feedstock": "scrap",
            "demand": 999.0,
            "carbon_breakdown - co2_stored": 0.0,
            "carbon_breakdown - co2_utilised": 0.0,
        },
    ]
    path = tmp_path / "post_processed_test.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    scenario = ScenarioInput(0, "E-W", "primary", tmp_path, path)

    result = load_europe_technology_portfolio([scenario]).set_index("technology_route")

    assert result.loc["DRI hydrogen", "capacity_t"] == pytest.approx(100.0)
    assert result.loc["DRI hydrogen", "production_t"] == pytest.approx(80.0)
    assert result.loc["DRI hydrogen", "co2_stored_t"] == pytest.approx(40.0)
    assert result.loc["EAF scrap", "capacity_t"] == pytest.approx(200.0)
    assert result.loc["EAF", "capacity_t"] == pytest.approx(300.0)
    assert result["capacity_t"].sum() == pytest.approx(600.0)
