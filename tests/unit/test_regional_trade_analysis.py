import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from steelo.adapters.dataprocessing.postprocessing.regional_trade_analysis import (
    aggregate_flows,
    parse_location,
    run_regional_trade_analysis,
)


def _location(lat: float, lon: float, country: str, region: str, iso3: str) -> str:
    return (
        f"Location(lat={lat}, lon={lon}, country='{country}', "
        f"region='{region}', iso3='{iso3}', distance_to_other_iso3=None)"
    )


def _trade_row(
    source_id: str,
    source: str,
    source_tech: str,
    destination_id: str,
    destination: str,
    volume: float,
) -> dict[str, object]:
    return {
        "commodity": "steel",
        "source_type": "Plant-FurnaceGroup",
        "source_id": source_id,
        "source_location": source,
        "capacity_at_source": volume,
        "source_tech": source_tech,
        "destination_type": "DemandCenter",
        "destination_id": destination_id,
        "destination_location": destination,
        "allocated_volume": volume,
        "allocation_cost": 1.0,
        "demand_at_destination": volume,
        "supply_at_source": "N/A",
    }


def test_parse_location_supports_model_repr():
    location = parse_location(_location(51.2, 10.4, "Germany", "Europe", "DEU"))

    assert location.iso3 == "DEU"
    assert location.region == "Europe"
    assert location.lat == pytest.approx(51.2)
    assert location.lon == pytest.approx(10.4)


def test_aggregate_flows_uses_countries_in_europe_and_regions_elsewhere():
    trade = pd.DataFrame(
        [
            {
                "source_iso3": "DEU",
                "source_region": "Europe",
                "destination_iso3": "FRA",
                "destination_region": "Europe",
                "allocated_volume": 100.0,
            },
            {
                "source_iso3": "USA",
                "source_region": "North America",
                "destination_iso3": "DEU",
                "destination_region": "Europe",
                "allocated_volume": 50.0,
            },
            {
                "source_iso3": "FRA",
                "source_region": "Europe",
                "destination_iso3": "USA",
                "destination_region": "North America",
                "allocated_volume": 70.0,
            },
        ]
    )

    germany = aggregate_flows(trade, "germany")
    europe = aggregate_flows(trade, "europe")

    assert set(map(tuple, germany[["source", "destination", "volume"]].to_records(index=False))) == {
        ("DEU", "FRA", 100.0),
        ("North America", "DEU", 50.0),
    }
    assert set(map(tuple, europe[["source", "destination", "volume"]].to_records(index=False))) == {
        ("North America", "Europe", 50.0),
        ("Europe", "North America", 70.0),
    }


def test_run_regional_trade_analysis_creates_maps_and_tables(tmp_path: Path):
    run_dir = tmp_path / "run"
    tm_dir = run_dir / "TM"
    tm_dir.mkdir(parents=True)
    locations = {
        "DEU": _location(51.0, 10.0, "DEU", "unknown", "DEU"),
        "FRA": _location(46.0, 2.0, "FRA", "unknown", "FRA"),
        # Supplier-style locations from real output may use the country name as
        # region and leave ISO3 blank.
        "USA": _location(39.0, -98.0, "USA", "USA", ""),
        "BRA": _location(-10.0, -52.0, "Brazil", "Brazil", ""),
    }
    trade = pd.DataFrame(
        [
            _trade_row("de_bf", locations["DEU"], "BF-BOF", "fr_demand", locations["FRA"], 100.0),
            _trade_row("us_eaf", locations["USA"], "EAF", "de_demand", locations["DEU"], 50.0),
            _trade_row("de_bf", locations["DEU"], "BF-BOF", "br_demand", locations["BRA"], 30.0),
            _trade_row("fr_eaf", locations["FRA"], "EAF", "us_demand", locations["USA"], 70.0),
        ]
    )
    trade.to_csv(tm_dir / "steel_trade_allocations_2030.csv", index=False)

    # de_bf deliberately appears twice, as it does when the post-processor has
    # one row per feedstock. Its production must be counted only once.
    plants = pd.DataFrame(
        [
            {
                "year": 2030,
                "iso3": "DEU",
                "region": "Europe",
                "furnace_group_id": "de_bf",
                "technology": "BF-BOF",
                "product": "steel",
                "production": 130.0,
                "feedstock": "hot_metal",
                "demand": 100.0,
            },
            {
                "year": 2030,
                "iso3": "DEU",
                "region": "Europe",
                "furnace_group_id": "de_bf",
                "technology": "BF-BOF",
                "product": "steel",
                "production": 130.0,
                "feedstock": "scrap",
                "demand": 30.0,
            },
            {
                "year": 2030,
                "iso3": "FRA",
                "region": "Europe",
                "furnace_group_id": "fr_eaf",
                "technology": "EAF",
                "product": "steel",
                "production": 70.0,
                "feedstock": "scrap",
                "demand": 70.0,
            },
            {
                "year": 2030,
                "iso3": "USA",
                "region": "North America",
                "furnace_group_id": "us_eaf",
                "technology": "EAF",
                "product": "steel",
                "production": 50.0,
                "feedstock": "scrap",
                "demand": 50.0,
            },
        ]
    )
    plants.to_csv(run_dir / "post_processed_2030-01-01_00-00.csv", index=False)

    shapes = gpd.GeoDataFrame(
        {"ISO_A3": ["DEU", "FRA", "USA", "BRA"], "NAME": ["Germany", "France", "United States", "Brazil"]},
        geometry=[box(5, 47, 15, 55), box(-5, 42, 8, 51), box(-125, 25, -65, 50), box(-74, -34, -34, 5)],
        crs="EPSG:4326",
    )
    shape_dir = tmp_path / "natural_earth"
    shape_dir.mkdir()
    shape_path = shape_dir / "ne_110m_admin_0_countries.shp"
    shapes.to_file(shape_path)
    data_dir = tmp_path / "prepared_data"
    fixtures_dir = data_dir / "fixtures"
    fixtures_dir.mkdir(parents=True)
    mappings = [
        {"Country": "Germany", "ISO 3-letter code": "DEU", "region_for_outputs": "Europe"},
        {"Country": "France", "ISO 3-letter code": "FRA", "region_for_outputs": "Europe"},
        {"Country": "United States", "ISO 3-letter code": "USA", "region_for_outputs": "North America"},
        {"Country": "Brazil", "ISO 3-letter code": "BRA", "region_for_outputs": "Latin America"},
    ]
    (fixtures_dir / "country_mappings.json").write_text(json.dumps(mappings), encoding="utf-8")
    (run_dir / "simulation_config.json").write_text(
        json.dumps({"countries_shapefile_dir": str(shape_dir), "data_dir": str(data_dir)}), encoding="utf-8"
    )

    created = run_regional_trade_analysis(run_dir, year=2030)

    assert len(created) == 10
    assert all(path.is_file() and path.stat().st_size > 0 for path in created)
    germany_flows = pd.read_csv(run_dir / "regional_trade_analysis/2030/germany_trade_steel_flows.csv")
    assert germany_flows["volume"].sum() == pytest.approx(180.0)
    germany_nodes = pd.read_csv(run_dir / "regional_trade_analysis/2030/germany_trade_steel_nodes.csv")
    germany = germany_nodes[germany_nodes["node"] == "DEU"].iloc[0]
    assert germany["production_total"] == pytest.approx(130.0)
    germany_transition = pd.read_csv(
        run_dir / "regional_trade_analysis/2030/germany_steel_production_origin_transition.csv"
    ).iloc[0]
    assert germany_transition["primary_production"] == pytest.approx(100.0)
    assert germany_transition["secondary_production"] == pytest.approx(30.0)
    assert germany_transition["net_trade"] == pytest.approx(-80.0)
    assert germany_transition["trade_direction"] == "net exporter"
    europe_transition = pd.read_csv(
        run_dir / "regional_trade_analysis/2030/europe_steel_production_origin_transition.csv"
    ).iloc[0]
    assert europe_transition["primary_production"] == pytest.approx(100.0)
    assert europe_transition["secondary_production"] == pytest.approx(100.0)
    assert europe_transition["net_trade"] == pytest.approx(-50.0)
