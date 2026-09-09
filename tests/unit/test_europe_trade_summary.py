from pathlib import Path

import pandas as pd
import pytest

from steelo.adapters.dataprocessing.postprocessing.europe_trade_summary import (
    summarise_europe_trade,
    summarise_europe_trade_flows,
)


def _location(iso3: str) -> str:
    return f"Location(lat=0, lon=0, country='{iso3}', region='unknown', iso3='{iso3}')"


def test_summarise_europe_trade_by_year_and_commodity(tmp_path: Path):
    run_dir = tmp_path / "run"
    tm_dir = run_dir / "TM"
    tm_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"iso3": "DEU", "region": "Europe"},
            {"iso3": "FRA", "region": "Europe"},
            {"iso3": "USA", "region": "North America"},
        ]
    ).to_csv(run_dir / "post_processed_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "commodity": "steel",
                "source_location": _location("USA"),
                "destination_location": _location("DEU"),
                "allocated_volume": 10.0,
            },
            {
                "commodity": "steel",
                "source_location": _location("DEU"),
                "destination_location": _location("USA"),
                "allocated_volume": 4.0,
            },
            {
                "commodity": "steel",
                "source_location": _location("DEU"),
                "destination_location": _location("FRA"),
                "allocated_volume": 99.0,
            },
            {
                "commodity": "scrap",
                "source_location": _location("USA"),
                "destination_location": _location("FRA"),
                "allocated_volume": 7.0,
            },
            # Cyprus has no furnace row, but still belongs to the model's Europe region.
            {
                "commodity": "scrap",
                "source_location": _location("USA"),
                "destination_location": _location("CYP"),
                "allocated_volume": 3.0,
            },
        ]
    ).to_csv(tm_dir / "steel_trade_allocations_2030.csv", index=False)

    result = summarise_europe_trade(run_dir).set_index(["year", "commodity"])

    assert result.loc[(2030, "steel"), "imports"] == pytest.approx(10.0)
    assert result.loc[(2030, "steel"), "exports"] == pytest.approx(4.0)
    assert result.loc[(2030, "steel"), "net_imports"] == pytest.approx(6.0)
    assert result.loc[(2030, "scrap"), "imports"] == pytest.approx(10.0)
    assert result.loc[(2030, "scrap"), "exports"] == pytest.approx(0.0)

    flows = summarise_europe_trade_flows(run_dir)
    steel = flows[flows["commodity"].eq("steel")].set_index(["country_from", "country_to"])
    assert steel.loc[("USA", "DEU"), "direction"] == "import"
    assert steel.loc[("USA", "DEU"), "volume"] == pytest.approx(10.0)
    assert steel.loc[("DEU", "USA"), "direction"] == "export"
    assert steel.loc[("DEU", "USA"), "volume"] == pytest.approx(4.0)
    assert ("DEU", "FRA") not in steel.index
