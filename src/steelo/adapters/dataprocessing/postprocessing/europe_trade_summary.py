"""Summarise annual commodity imports to and exports from Europe."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import pandas as pd
import pycountry

from .analysis_scope import EUROPE_ANALYSIS_DESCRIPTION, EUROPE_ANALYSIS_ISO3


_LOCATION_VALUE = re.compile(r"(?P<key>iso3|country)=(?P<value>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")
_YEAR = re.compile(r"steel_trade_allocations_(\d{4})\.csv$")


def _location_iso3(value: object) -> str:
    """Return ISO3 from a serialized model Location."""
    fields = {
        match.group("key"): str(ast.literal_eval(match.group("value"))).strip()
        for match in _LOCATION_VALUE.finditer(str(value))
    }
    iso3 = fields.get("iso3", "").upper()
    if len(iso3) == 3:
        return iso3
    country = fields.get("country", "")
    if len(country) == 3:
        return country.upper()
    try:
        return pycountry.countries.lookup(country).alpha_3
    except LookupError:
        return ""


def _europe_iso3(run_dir: Path) -> set[str]:
    """Return the fixed comparison scope; ``run_dir`` is retained for API compatibility."""
    _ = run_dir
    return set(EUROPE_ANALYSIS_ISO3)


def summarise_europe_trade_flows(run_dir: Path) -> pd.DataFrame:
    f"""Return bilateral flows crossing the {EUROPE_ANALYSIS_DESCRIPTION} boundary.

    ``country_from`` and ``country_to`` are ISO3 country codes. Intra-European
    and wholly non-European flows are excluded.
    """
    run_dir = Path(run_dir)
    europe_iso3 = _europe_iso3(run_dir)
    results: list[pd.DataFrame] = []
    for path in sorted((run_dir / "TM").glob("steel_trade_allocations_*.csv")):
        match = _YEAR.search(path.name)
        if not match:
            continue
        trade = pd.read_csv(path)
        required = {"commodity", "source_location", "destination_location", "allocated_volume"}
        if missing := required - set(trade.columns):
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

        trade["country_from"] = trade["source_location"].map(_location_iso3)
        trade["country_to"] = trade["destination_location"].map(_location_iso3)
        unresolved = trade[trade["country_from"].eq("") | trade["country_to"].eq("")]
        if not unresolved.empty:
            raise ValueError(f"Could not resolve countries in {len(unresolved)} rows of {path}")

        source_europe = trade["country_from"].isin(europe_iso3)
        destination_europe = trade["country_to"].isin(europe_iso3)
        boundary = trade.loc[source_europe ^ destination_europe].copy()
        boundary["direction"] = destination_europe[source_europe ^ destination_europe].map(
            {True: "import", False: "export"}
        )
        boundary["year"] = int(match.group(1))
        results.append(
            boundary.groupby(["year", "direction", "country_from", "country_to", "commodity"], as_index=False)[
                "allocated_volume"
            ]
            .sum()
            .rename(columns={"allocated_volume": "volume"})
        )

    if not results:
        raise FileNotFoundError(f"No steel_trade_allocations_*.csv found in {run_dir / 'TM'}")
    return pd.concat(results, ignore_index=True).sort_values(
        ["year", "direction", "country_from", "country_to", "commodity"], ignore_index=True
    )


def _commodity_totals(flows: pd.DataFrame) -> pd.DataFrame:
    totals = flows.pivot_table(
        index=["year", "commodity"], columns="direction", values="volume", aggfunc="sum", fill_value=0.0
    ).rename(columns={"import": "imports", "export": "exports"})
    for column in ("imports", "exports"):
        if column not in totals:
            totals[column] = 0.0
    result = totals.reset_index()
    result["net_imports"] = result["imports"] - result["exports"]
    return result[["year", "commodity", "imports", "exports", "net_imports"]]


def summarise_europe_trade(run_dir: Path) -> pd.DataFrame:
    """Sum external European imports and exports by year and commodity."""
    return _commodity_totals(summarise_europe_trade_flows(run_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Simulation run directory containing TM/")
    parser.add_argument("--output", type=Path, help="Output CSV (default: RUN_DIR/europe_trade_summary.csv)")
    parser.add_argument(
        "--flows-output", type=Path, help="Bilateral output CSV (default: RUN_DIR/europe_trade_flows.csv)"
    )
    args = parser.parse_args()
    output = args.output or args.run_dir / "europe_trade_summary.csv"
    flows_output = args.flows_output or args.run_dir / "europe_trade_flows.csv"
    flows = summarise_europe_trade_flows(args.run_dir)
    summary = _commodity_totals(flows)
    flows.to_csv(flows_output, index=False)
    summary.to_csv(output, index=False)
    print(f"Wrote {len(flows)} bilateral flows to {flows_output}")
    print(f"Wrote {len(summary)} rows to {output}")


if __name__ == "__main__":
    main()
