from pathlib import Path

import pandas as pd
import pytest

from steelo.adapters.dataprocessing.postprocessing.emissions_subsidy_timeseries import (
    _plot_emissions,
    _plot_eu_fiscal_balance,
    build_parser,
    summarise_scenario,
)
from steelo.adapters.dataprocessing.postprocessing.europe_scenario_post_analysis import ScenarioInput
from steelo.adapters.dataprocessing.postprocessing.world_scenario_analysis import DIRECT_EMISSIONS, INDIRECT_EMISSIONS


def test_summarise_scenario_counts_emissions_and_subsidies_once(tmp_path: Path) -> None:
    rows = [
        {
            "year": 2025,
            "iso3": "DEU",
            "region": "Europe",
            "furnace_group_id": "eu-steel",
            "product": "steel",
            "production": 100.0,
            "capacity": 200.0,
            "decision_year": 2025,
            "investment_decision": "retrofit_technology_switch",
            DIRECT_EMISSIONS: 20.0,
            INDIRECT_EMISSIONS: 5.0,
            "unit_carbon_cost": 4.0,
            "unit_subsidy_hydrogen": 3.0,
            "unit_subsidy_electricity": 2.0,
            "unit_subsidy_capex": 10.0,
        },
        {
            "year": 2025,
            "iso3": "DEU",
            "region": "Europe",
            "furnace_group_id": "eu-steel",
            "product": "steel",
            "production": 100.0,
            "capacity": 200.0,
            "decision_year": 2025,
            "investment_decision": "retrofit_technology_switch",
            DIRECT_EMISSIONS: 20.0,
            INDIRECT_EMISSIONS: 5.0,
            "unit_carbon_cost": 4.0,
            "unit_subsidy_hydrogen": 3.0,
            "unit_subsidy_electricity": 2.0,
            "unit_subsidy_capex": 10.0,
        },
        {
            "year": 2025,
            "iso3": "USA",
            "region": "North America",
            "furnace_group_id": "us-iron",
            "product": "iron",
            "production": 50.0,
            "capacity": 80.0,
            "decision_year": 0,
            "investment_decision": "none",
            DIRECT_EMISSIONS: 30.0,
            INDIRECT_EMISSIONS: 10.0,
            "unit_carbon_cost": 5.0,
            "unit_subsidy_hydrogen": 1.0,
            "unit_subsidy_electricity": 0.0,
            "unit_subsidy_capex": 8.0,
        },
        {
            "year": 2026,
            "iso3": "DEU",
            "region": "Europe",
            "furnace_group_id": "eu-steel",
            "product": "steel",
            "production": 120.0,
            "capacity": 200.0,
            "decision_year": 0,
            "investment_decision": "none",
            DIRECT_EMISSIONS: 18.0,
            INDIRECT_EMISSIONS: 6.0,
            "unit_carbon_cost": 6.0,
            "unit_subsidy_hydrogen": 3.0,
            "unit_subsidy_electricity": 2.0,
            "unit_subsidy_capex": 10.0,
        },
    ]
    csv = tmp_path / "post_processed_test.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    scenario = ScenarioInput(0, "E-W", "primary", tmp_path, csv)

    result = summarise_scenario(scenario)
    global_2025 = result[(result["scope"] == "Global") & (result["year"] == 2025)].iloc[0]
    eu_2025 = result[(result["scope"] == "EU") & (result["year"] == 2025)].iloc[0]
    eu_2026 = result[(result["scope"] == "EU") & (result["year"] == 2026)].iloc[0]

    assert global_2025["direct_tco2e"] == pytest.approx(50.0)
    assert global_2025["direct_indirect_tco2e"] == pytest.approx(65.0)
    assert eu_2025["direct_tco2e"] == pytest.approx(20.0)
    assert eu_2025["hydrogen_subsidy_usd"] == pytest.approx(300.0)
    assert eu_2025["electricity_subsidy_usd"] == pytest.approx(200.0)
    assert eu_2025["capex_subsidy_usd"] == pytest.approx(2_000.0)
    assert eu_2025["carbon_tax_revenue_usd"] == pytest.approx(400.0)
    assert eu_2025["total_subsidy_spending_usd"] == pytest.approx(2_500.0)
    assert eu_2025["net_fiscal_cost_usd"] == pytest.approx(2_100.0)
    assert eu_2026["capex_subsidy_usd"] == pytest.approx(0.0)
    assert eu_2026["carbon_tax_revenue_usd"] == pytest.approx(720.0)
    assert eu_2026["net_fiscal_cost_usd"] == pytest.approx(-120.0)
    assert eu_2026["cumulative_direct_tco2e"] == pytest.approx(38.0)

    fiscal_plot = tmp_path / "fiscal.png"
    _plot_eu_fiscal_balance(result, fiscal_plot)
    assert fiscal_plot.is_file()


def test_all_scenarios_are_included_by_default() -> None:
    assert build_parser().parse_args(["results"]).skip_runs == []


def test_eu_scope_excludes_non_eu_model_europe_country(tmp_path: Path) -> None:
    rows = []
    for iso3, furnace in (("GBR", "included"), ("SRB", "excluded")):
        rows.append(
            {
                "year": 2025,
                "iso3": iso3,
                "region": "Europe",
                "furnace_group_id": furnace,
                "product": "steel",
                "production": 100.0,
                "capacity": 100.0,
                "decision_year": 0,
                "investment_decision": "none",
                DIRECT_EMISSIONS: 10.0,
                INDIRECT_EMISSIONS: 2.0,
                "unit_carbon_cost": 4.0,
                "unit_subsidy_hydrogen": 0.0,
                "unit_subsidy_electricity": 1.0,
                "unit_subsidy_capex": 0.0,
            }
        )
    csv = tmp_path / "post_processed_test.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    scenario = ScenarioInput(0, "E-W", "primary", tmp_path, csv)

    result = summarise_scenario(scenario)
    eu = result[result["scope"].eq("EU")].iloc[0]

    assert eu["direct_tco2e"] == pytest.approx(10.0)
    assert eu["electricity_subsidy_usd"] == pytest.approx(100.0)
    assert eu["carbon_tax_revenue_usd"] == pytest.approx(400.0)


def test_annual_emissions_use_common_5500_mt_axis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = pd.DataFrame(
        {
            "scenario_index": [0],
            "scope": ["Global"],
            "year": [2050],
            "direct_tco2e": [1e9],
            "direct_indirect_tco2e": [2e9],
        }
    )
    limits: list[tuple[float, float]] = []
    from matplotlib.axes import Axes

    original = Axes.set_ylim

    def record_limits(self: Axes, *args: object, **kwargs: object) -> tuple[float, float]:
        if len(args) >= 2:
            limits.append((float(args[0]), float(args[1])))
        elif len(args) == 1 and isinstance(args[0], (tuple, list)):
            limits.append((float(args[0][0]), float(args[0][1])))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_ylim", record_limits)
    _plot_emissions(summary, "Global", False, tmp_path / "annual.png")
    assert (0.0, 5500.0) in limits
