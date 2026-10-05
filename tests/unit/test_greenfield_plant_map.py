import importlib
from pathlib import Path
import sys

import pandas as pd

from PIL import Image

from steelo.adapters.dataprocessing.postprocessing.greenfield_plant_map import (
    discover_greenfield_runs,
    draw_greenfield_evolution_gif,
    draw_greenfield_map_matrix,
    summarise_greenfield_capacity,
)


def test_module_can_be_imported_directly_from_an_interactive_path(monkeypatch) -> None:
    module_dir = Path(sys.modules[draw_greenfield_map_matrix.__module__].__file__).parent
    monkeypatch.syspath_prepend(str(module_dir))
    sys.modules.pop("greenfield_plant_map", None)

    module = importlib.import_module("greenfield_plant_map")

    assert callable(module.run_greenfield_maps)
    sys.modules.pop("greenfield_plant_map", None)


def test_greenfield_summary_excludes_expansions_and_uses_first_operating_year(tmp_path: Path):
    run_dir = tmp_path / "sim"
    run_dir.mkdir()
    common = {
        "plant_id": "P-new",
        "furnace_group_id": "P-new",
        "iso3": "CAN",
        "country": "Canada",
        "region": "North America",
        "product": "iron",
        "technology": "DRI",
        "capacity": 2_000_000,
        "latitude": 50.0,
        "longitude": -100.0,
    }
    rows = [
        {"year": 2030, "plant_group_id": "indi_CAN", **common},
        {"year": 2031, "plant_group_id": "indi_CAN", **common},
        {
            "year": 2031,
            "plant_group_id": "indi_CAN",
            **{**common, "furnace_group_id": "P-new_expansion", "capacity": 3_000_000},
        },
        {
            "year": 2030,
            "plant_group_id": "E-existing",
            "plant_id": "P-old",
            "furnace_group_id": "P-old_4",
            "iso3": "DEU",
            "country": "Germany",
            "region": "Europe",
            "product": "steel",
            "technology": "EAF",
            "capacity": 1_000_000,
            "latitude": 51.0,
            "longitude": 10.0,
        },
    ]
    pd.DataFrame(rows).to_csv(run_dir / "plant_agent_fleet_decisions.csv", index=False)
    pd.DataFrame([{"year": 2030, "furnace_group_id": "P-new", "chosen_reductant": "hydrogen"}]).to_csv(
        run_dir / "post_processed_test.csv", index=False
    )

    result = summarise_greenfield_capacity(run_dir)

    assert len(result) == 1
    assert result.iloc[0]["plant_id"] == "P-new"
    assert result.iloc[0]["operational_year"] == 2030
    assert result.iloc[0]["capacity_mtpa"] == 2.0
    assert result.iloc[0]["technology_route"] == "DRI (hydrogen)"


def test_legacy_coordinates_are_recovered_from_trade_csv(tmp_path: Path):
    run_dir = tmp_path / "sim"
    (run_dir / "TM").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "year": 2040,
                "plant_group_id": "indi_BRA",
                "plant_id": "P-new",
                "furnace_group_id": "P-new",
                "iso3": "BRA",
                "product": "steel",
                "technology": "EAF",
                "capacity": 2_500_000,
            }
        ]
    ).to_csv(run_dir / "plant_agent_fleet_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_id": "P-new",
                "source_location": "Location(lat=-23.5, lon=-46.6, country='BRA', iso3='BRA')",
            }
        ]
    ).to_csv(run_dir / "TM" / "steel_trade_allocations_2040.csv", index=False)

    result = summarise_greenfield_capacity(run_dir)

    assert result.iloc[0]["latitude"] == -23.5
    assert result.iloc[0]["longitude"] == -46.6


def test_scenario_discovery_uses_titles_and_skips_run_11(tmp_path: Path):
    for index in (0, 11, 15):
        run_dir = tmp_path / f"master_input_{index}" / "sim_20250101_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "plant_agent_fleet_decisions.csv").touch()

    runs = discover_greenfield_runs(tmp_path)

    assert [(run.index, run.title) for run in runs] == [(0, "E-W"), (15, "ECS-WCS")]


def test_evolution_gif_contains_one_cumulative_frame_per_year(tmp_path: Path):
    summary = pd.DataFrame(
        [
            {
                "operational_year": 2026,
                "product": "iron",
                "technology": "DRI",
                "capacity_mtpa": 2.0,
                "latitude": 50.0,
                "longitude": 10.0,
            }
        ]
    )
    output = tmp_path / "evolution.gif"

    draw_greenfield_evolution_gif(
        summary,
        None,
        output,
        scenario_title="Test",
        year_from=2025,
        year_to=2027,
        frame_duration_ms=10,
    )

    with Image.open(output) as image:
        assert image.n_frames == 3


def test_map_matrix_uses_four_by_four_scenario_layout(tmp_path: Path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    Image.new("RGB", (20, 10), "red").save(first)
    Image.new("RGB", (20, 10), "blue").save(last)
    output = tmp_path / "matrix.png"

    draw_greenfield_map_matrix({0: first, 15: last}, output, cell_size=(100, 50))

    with Image.open(output) as matrix:
        assert matrix.size == (400, 200)
        assert matrix.getpixel((10, 160)) == (255, 0, 0)
        assert matrix.getpixel((390, 10)) == (0, 0, 255)
