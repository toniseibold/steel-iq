"""Compare regional iron/steel emissions intensity with cost and market price."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .europe_competitiveness_summary import RunArtifacts, discover_runs


DEFAULT_YEARS = (2030, 2050)
PRODUCTS = ("iron", "steel")
DIRECT_EMISSIONS = "emissions_rs-inspired_direct_ghg"
INDIRECT_EMISSIONS = "emissions_rs-inspired_indirect_ghg"


def _market_prices(run: RunArtifacts) -> pd.DataFrame:
    candidates = sorted((run.run_dir / "data").glob("market_prices_*.csv"))
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(candidates[-1])


def summarise_run(run: RunArtifacts, years: Iterable[int] = DEFAULT_YEARS) -> pd.DataFrame:
    """Calculate production-weighted regional metrics for one simulation run.

    Emissions intensity includes RS-inspired direct and indirect GHG emissions.
    The underlying emissions columns contain annual tonnes, so regional intensity
    is total emissions divided by total production rather than a plant-count mean.
    """
    plants = pd.read_csv(run.post_processed_csv)
    required = {
        "year",
        "region",
        "product",
        "production",
        "unit_production_cost",
        "furnace_group_id",
        DIRECT_EMISSIONS,
        INDIRECT_EMISSIONS,
    }
    if missing := required - set(plants.columns):
        raise ValueError(f"{run.post_processed_csv} is missing columns: {', '.join(sorted(missing))}")

    # Post-processing has one row per furnace/feedstock; count each furnace once.
    plants = plants.drop_duplicates(["year", "furnace_group_id"]).copy()
    plants["year"] = pd.to_numeric(plants["year"], errors="coerce")
    plants["production"] = pd.to_numeric(plants["production"], errors="coerce").fillna(0.0)
    plants["unit_production_cost"] = pd.to_numeric(plants["unit_production_cost"], errors="coerce")
    plants[DIRECT_EMISSIONS] = pd.to_numeric(plants[DIRECT_EMISSIONS], errors="coerce").fillna(0.0)
    plants[INDIRECT_EMISSIONS] = pd.to_numeric(plants[INDIRECT_EMISSIONS], errors="coerce").fillna(0.0)
    selected_years = {int(year) for year in years}
    plants = plants[
        plants["year"].isin(selected_years)
        & plants["product"].astype(str).str.casefold().isin(PRODUCTS)
        & plants["production"].gt(0)
    ].copy()

    prices = _market_prices(run)
    price_lookup: dict[tuple[int, str], float] = {}
    if not prices.empty and "year" in prices:
        for row in prices.itertuples(index=False):
            year = int(row.year)
            price_lookup[(year, "iron")] = float(getattr(row, "iron_price_usd_per_t", math.nan))
            price_lookup[(year, "steel")] = float(getattr(row, "steel_price_usd_per_t", math.nan))

    records: list[dict[str, object]] = []
    for (year, product, region), group in plants.groupby(["year", "product", "region"], dropna=False):
        production = float(group["production"].sum())
        valid_cost = group["unit_production_cost"].notna() & group["production"].gt(0)
        average_cost = (
            float(
                np.average(group.loc[valid_cost, "unit_production_cost"], weights=group.loc[valid_cost, "production"])
            )
            if valid_cost.any()
            else math.nan
        )
        direct = float(group[DIRECT_EMISSIONS].sum())
        indirect = float(group[INDIRECT_EMISSIONS].sum())
        records.append(
            {
                "scenario_name": run.scenario_name,
                "master_input_name": run.master_input_name,
                "year": int(year),
                "product": str(product).casefold(),
                "region": str(region),
                "production_mt": production / 1e6,
                "direct_emissions_mtco2e": direct / 1e6,
                "indirect_emissions_mtco2e": indirect / 1e6,
                "avg_ghg_intensity_tco2e_per_t": (direct + indirect) / production,
                "avg_unit_production_cost_usd_per_t": average_cost,
                "global_market_price_usd_per_t": price_lookup.get((int(year), str(product).casefold()), math.nan),
            }
        )
    return pd.DataFrame.from_records(records)


def build_summary(results_dir: Path, years: Iterable[int] = DEFAULT_YEARS) -> pd.DataFrame:
    """Build the regional table for every scenario below ``results_dir``."""
    frames = [summarise_run(run, years) for run in discover_runs(results_dir)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(f"No regional emissions records found below {results_dir}")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["master_input_name", "year", "product", "region"], ignore_index=True
    )


def _plot_scenario(data: pd.DataFrame, output_path: Path, years: Iterable[int]) -> None:
    years = tuple(int(year) for year in years)
    fig, axes = plt.subplots(
        len(PRODUCTS),
        len(years),
        figsize=(7 * len(years), 5.5 * len(PRODUCTS)),
        dpi=180,
        squeeze=False,
    )
    colors = plt.get_cmap("tab20")
    regions = sorted(data["region"].dropna().unique())
    region_colors = {region: colors(index % 20) for index, region in enumerate(regions)}

    for row_index, product in enumerate(PRODUCTS):
        for column_index, year in enumerate(years):
            axis = axes[row_index, column_index]
            subset = data[(data["product"] == product) & (data["year"] == year)].dropna(
                subset=["avg_ghg_intensity_tco2e_per_t", "avg_unit_production_cost_usd_per_t"]
            )
            if subset.empty:
                axis.text(0.5, 0.5, "No production data", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            sizes = 45.0 + 12.0 * np.sqrt(subset["production_mt"].clip(lower=0.0))
            for size, item in zip(sizes, subset.itertuples(index=False)):
                axis.scatter(
                    item.avg_ghg_intensity_tco2e_per_t,
                    item.avg_unit_production_cost_usd_per_t,
                    s=size,
                    color=region_colors[item.region],
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.5,
                )
                axis.annotate(
                    item.region,
                    (item.avg_ghg_intensity_tco2e_per_t, item.avg_unit_production_cost_usd_per_t),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
            market_prices = subset["global_market_price_usd_per_t"].dropna()
            if not market_prices.empty:
                market_price = float(market_prices.iloc[0])
                axis.axhline(market_price, color="#333333", linestyle="--", linewidth=1.2)
                axis.text(
                    0.99,
                    market_price,
                    f" Global market price: ${market_price:,.0f}/t",
                    transform=axis.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=8,
                )
            axis.set_title(f"{product.title()} — {year}")
            axis.set_xlabel("Average GHG intensity (tCO₂e/t)")
            axis.set_ylabel("Average production cost (USD/t)")
            axis.grid(alpha=0.25)

    scenario = str(data["scenario_name"].iloc[0])
    fig.suptitle(f"Regional emissions intensity and cost — {scenario}", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Bubble area increases with regional production; dashed line is the global market price.",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(summary: pd.DataFrame, output_dir: Path, years: Iterable[int] = DEFAULT_YEARS) -> list[Path]:
    """Write the combined CSV and one four-panel chart per scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "regional_emissions_and_prices.csv"
    summary.to_csv(csv_path, index=False)
    created = [csv_path]
    for scenario, subset in summary.groupby("scenario_name", sort=True):
        safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in scenario)
        plot_path = output_dir / f"{safe_name}_regional_emissions_and_prices.png"
        _plot_scenario(subset, plot_path, years)
        created.append(plot_path)
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Directory containing master_input_* scenario folders")
    parser.add_argument(
        "--output-dir", type=Path, help="Destination (default: RESULTS_DIR/regional_emissions_price_summary)"
    )
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.results_dir / "regional_emissions_price_summary"
    summary = build_summary(args.results_dir.resolve(), args.years)
    for path in write_outputs(summary, output_dir.resolve(), args.years):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
