from pathlib import Path

import pandas as pd
import pytest

from steelo.h2_dri_quota.config import H2DRIQuotaConfig
from steelo.h2_dri_quota.reporting import generate_h2_dri_quota_report


def test_report_tracks_destinations_and_cost_comparisons(tmp_path: Path):
    run_dir = tmp_path / "run"
    tm_dir = run_dir / "TM"
    data_dir = run_dir / "data"
    quota_dir = data_dir / "h2_dri_quota"
    tm_dir.mkdir(parents=True)
    data_dir.mkdir()
    plants = pd.DataFrame(
        [
            [2030, "DEU", "Germany", "p1", "h2", "DRI", "iron", "hydrogen", 120.0, 300.0],
            [2030, "FRA", "France", "p2", "other-i", "BF", "iron", "coke", 100.0, 250.0],
            [2030, "DEU", "Germany", "p3", "steel-h2", "EAF", "steel", "", 90.0, 500.0],
            [2030, "FRA", "France", "p4", "steel-other", "EAF", "steel", "", 100.0, 450.0],
            [2030, "CHN", "China", "p5", "steel-world", "BOF", "steel", "", 200.0, 400.0],
        ],
        columns=[
            "year",
            "iso3",
            "country",
            "plant_id",
            "furnace_group_id",
            "technology",
            "product",
            "chosen_reductant",
            "production",
            "unit_production_cost",
        ],
    )
    post = run_dir / "post_processed.csv"
    plants.to_csv(post, index=False)
    pd.DataFrame(
        [
            {
                "commodity": "dri_high",
                "source_id": "h2",
                "destination_id": "steel-h2",
                "destination_type": "Plant-FurnaceGroup",
                "destination_location": "Location(iso3='DEU')",
                "allocated_volume": 117.0923,
            },
            {
                "commodity": "steel",
                "source_id": "steel-h2",
                "destination_id": "Germany",
                "destination_type": "DemandCenter",
                "destination_location": "Location(iso3='DEU')",
                "allocated_volume": 1_000.0,
            },
        ]
    ).to_csv(tm_dir / "steel_trade_allocations_2030.csv", index=False)
    pd.DataFrame([{"year": 2030, "iron_price_usd_per_t": 310.0, "steel_price_usd_per_t": 510.0}]).to_csv(
        data_dir / "market_prices_2025_2060.csv", index=False
    )

    summary_path, flow_path, plot_path = generate_h2_dri_quota_report(
        post_processed_csv=post,
        tm_dir=tm_dir,
        output_dir=quota_dir,
        config=H2DRIQuotaConfig(enabled=True),
    )

    summary = pd.read_csv(summary_path).iloc[0]
    flows = pd.read_csv(flow_path)
    assert summary["quota_met"]
    assert summary["h2_dri_unit_production_cost_usd_t"] == pytest.approx(300.0)
    assert summary["h2_dri_downstream_steel_unit_production_cost_usd_t"] == pytest.approx(500.0)
    assert summary["other_eu_steel_unit_production_cost_usd_t"] == pytest.approx(450.0)
    assert flows.iloc[0]["destination_furnace_group_id"] == "steel-h2"
    assert plot_path.exists()


def test_global_report_tracks_row_project_and_european_eaf_delivery(tmp_path: Path):
    run_dir = tmp_path / "run"
    tm_dir = run_dir / "TM"
    quota_dir = run_dir / "data" / "h2_dri_quota"
    tm_dir.mkdir(parents=True)
    quota_dir.mkdir(parents=True)
    post = run_dir / "post_processed.csv"
    pd.DataFrame(
        [
            {
                "year": 2030,
                "iso3": "CHN",
                "country": "China",
                "latitude": 37.75,
                "longitude": 84.0,
                "plant_id": "p-cn",
                "furnace_group_id": "h2-cn",
                "technology": "DRI",
                "product": "iron",
                "chosen_reductant": "hydrogen",
                "capacity": 200.0,
                "status": "operating",
                "production": 120.0,
                "unit_production_cost": 275.0,
            },
            {
                "year": 2030,
                "iso3": "DEU",
                "country": "Germany",
                "latitude": 51.0,
                "longitude": 10.0,
                "plant_id": "p-de",
                "furnace_group_id": "eaf-de",
                "technology": "EAF",
                "product": "steel",
                "chosen_reductant": "",
                "capacity": 200.0,
                "status": "operating",
                "production": 100.0,
                "unit_production_cost": 500.0,
            },
        ]
    ).to_csv(post, index=False)
    pd.DataFrame(
        [
            {
                "commodity": "hbi_high",
                "source_id": "h2-cn",
                "destination_id": "eaf-de",
                "destination_type": "Plant-FurnaceGroup",
                "destination_location": "Location(iso3='DEU')",
                "allocated_volume": 117.0923,
            },
            {
                "commodity": "steel",
                "source_id": "eaf-de",
                "destination_id": "Germany",
                "destination_type": "DemandCenter",
                "destination_location": "Location(iso3='DEU')",
                "allocated_volume": 500.0,
            },
        ]
    ).to_csv(tm_dir / "steel_trade_allocations_2030.csv", index=False)
    pd.DataFrame(
        [
            {
                "planning_year": 2026,
                "quota_year": 2030,
                "operation_start_year": 2030,
                "plant_id": "p-cn",
                "furnace_group_id": "h2-cn",
                "iso3": "CHN",
                "technology": "DRI",
                "chosen_reductant": "hydrogen",
                "candidate_scope": "global",
                "latest_npv_usd": -50.0,
                "indicative_policy_support_npv_usd": 50.0,
                "capacity_t": 200.0,
            }
        ]
    ).to_csv(quota_dir / "forced_projects.csv", index=False)

    summary_path, flow_path, _plot_path = generate_h2_dri_quota_report(
        post_processed_csv=post,
        tm_dir=tm_dir,
        output_dir=quota_dir,
        config=H2DRIQuotaConfig(enabled=True, candidate_scope="global"),
    )

    summary = pd.read_csv(summary_path).iloc[0]
    flows = pd.read_csv(flow_path)
    projects = pd.read_csv(quota_dir / "forced_project_outcomes.csv")
    assert summary["quota_met"]
    assert summary["candidate_scope"] == "global"
    assert summary["quota_scope_h2_dri_production_t"] == pytest.approx(120.0)
    assert summary["quota_h2_dri_delivered_to_eu_eaf_t"] == pytest.approx(117.0923)
    assert flows.iloc[0]["source_iso3"] == "CHN"
    assert projects.iloc[0]["country"] == "China"
    assert projects.iloc[0]["capacity_t"] == pytest.approx(200.0)
    assert projects.iloc[0]["latest_npv_usd"] == pytest.approx(-50.0)
    assert projects.iloc[0]["delivered_to_eu_eaf_t"] == pytest.approx(117.0923)
    assert projects.iloc[0]["destination_countries"] == "DEU (0.000 Mt)"
