from pathlib import Path

import pandas as pd
import pytest

from steelo.adapters.dataprocessing.postprocessing.analysis_scope import EUROPE_ANALYSIS_LABEL
from steelo.adapters.dataprocessing.postprocessing.world_scenario_analysis import (
    COMMODITY_COLOURS,
    SCENARIO_GRID_ORDER,
    SCENARIO_GRID_POSITION,
    ScenarioRun,
    _fixed_colours,
    aggregate_trade,
    calculate_production_metrics,
)


def test_shared_scenario_grid_uses_requested_row_major_order():
    assert SCENARIO_GRID_ORDER == (
        12,
        14,
        13,
        15,
        4,
        6,
        5,
        7,
        8,
        10,
        9,
        11,
        0,
        2,
        1,
        3,
    )
    assert SCENARIO_GRID_POSITION[0] == (3, 0)
    assert SCENARIO_GRID_POSITION[10] == (2, 1)
    assert SCENARIO_GRID_POSITION[5] == (1, 2)
    assert SCENARIO_GRID_POSITION[15] == (0, 3)


def test_aggregate_trade_uses_external_flows_and_signed_balances():
    trade = pd.DataFrame(
        [
            {
                "source_continent": "Europe",
                "destination_continent": "Asia",
                "commodity": "steel",
                "allocated_volume": 3_000_000.0,
                "source_tech": "EAF",
            },
            {
                "source_continent": "Asia",
                "destination_continent": "Europe",
                "commodity": "steel",
                "allocated_volume": 2_000_000.0,
                "source_tech": "BOF",
            },
            {
                "source_continent": "Europe",
                "destination_continent": "Europe",
                "commodity": "steel",
                "allocated_volume": 4_000_000.0,
                "source_tech": "EAF",
            },
        ]
    )

    flows, balances, production = aggregate_trade(trade, "steel")

    assert flows["volume_t"].sum() == 5_000_000.0
    europe = balances[balances["continent"] == "Europe"].set_index("direction")["volume_t"]
    assert europe["outgoing"] == 3_000_000.0
    assert europe["incoming"] == -2_000_000.0
    assert production["volume_t"].sum() == 9_000_000.0


def test_commodity_colours_do_not_change_when_other_categories_are_absent():
    all_colours = _fixed_colours(["io_high", "io_mid", "io_low"], COMMODITY_COLOURS)
    subset_colours = _fixed_colours(["io_high", "io_low"], COMMODITY_COLOURS)

    assert subset_colours["io_high"] == all_colours["io_high"] == "#e31a1c"
    assert subset_colours["io_low"] == all_colours["io_low"] == "#fcae91"


def test_intermediate_trade_keeps_green_and_grey_routes_separate():
    trade = pd.DataFrame(
        [
            {
                "source_continent": "Europe",
                "destination_continent": "Asia",
                "commodity": "hbi_high",
                "carbon_route": "green",
                "allocated_volume": 3.0,
            },
            {
                "source_continent": "Europe",
                "destination_continent": "Asia",
                "commodity": "hbi_high",
                "carbon_route": "grey",
                "allocated_volume": 2.0,
            },
        ]
    )

    flows, balances, _production = aggregate_trade(trade, "intermediates")

    assert set(flows["carbon_route"]) == {"green", "grey"}
    assert set(balances["carbon_route"]) == {"green", "grey"}


def test_metrics_are_production_weighted_and_do_not_duplicate_feedstocks(tmp_path: Path):
    rows = [
        {
            "year": 2050,
            "iso3": "DEU",
            "furnace_group_id": "eu",
            "product": "steel",
            "production": 100.0,
            "unit_production_cost": 500.0,
            "emissions_rs-inspired_direct_ghg": 100.0,
            "emissions_rs-inspired_indirect_ghg": 50.0,
        },
        {
            "year": 2050,
            "iso3": "DEU",
            "furnace_group_id": "eu",
            "product": "steel",
            "production": 100.0,
            "unit_production_cost": 500.0,
            "emissions_rs-inspired_direct_ghg": 100.0,
            "emissions_rs-inspired_indirect_ghg": 50.0,
        },
        {
            "year": 2050,
            "iso3": "USA",
            "furnace_group_id": "global",
            "product": "steel",
            "production": 300.0,
            "unit_production_cost": 700.0,
            "emissions_rs-inspired_direct_ghg": 600.0,
            "emissions_rs-inspired_indirect_ghg": 300.0,
        },
    ]
    plants_csv = tmp_path / "post_processed_test.csv"
    pd.DataFrame(rows).to_csv(plants_csv, index=False)
    run = ScenarioRun(0, "E-W", tmp_path, plants_csv)

    result = calculate_production_metrics(run, [2050])
    eu = result[result["scope"] == EUROPE_ANALYSIS_LABEL].iloc[0]
    global_row = result[result["scope"] == "Global"].iloc[0]

    assert eu["carbon_intensity_tco2e_per_t"] == pytest.approx(1.5)
    assert eu["average_production_cost_usd_per_t"] == pytest.approx(500.0)
    assert global_row["carbon_intensity_tco2e_per_t"] == pytest.approx(1050.0 / 400.0)
    assert global_row["average_production_cost_usd_per_t"] == pytest.approx(650.0)
