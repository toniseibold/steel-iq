"""Plot annual/cumulative sector emissions and annual subsidy costs for scenario runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .analysis_scope import EUROPE_ANALYSIS_DESCRIPTION, EUROPE_ANALYSIS_LABEL
from .europe_scenario_post_analysis import ScenarioInput, _europe_iso3, discover_scenario_inputs
from .world_scenario_analysis import (
    DIRECT_EMISSIONS,
    INDIRECT_EMISSIONS,
    SCENARIO_GRID_ORDER,
    SCENARIO_TITLES,
)


PRODUCTS = ("iron", "steel")
SCOPES = ("Global", "EU")
SCOPE_DISPLAY_NAMES = {"Global": "Global", "EU": EUROPE_ANALYSIS_LABEL}
SUBSIDY_COLUMNS = {
    "hydrogen": "unit_subsidy_hydrogen",
    "electricity": "unit_subsidy_electricity",
    "capex": "unit_subsidy_capex",
}
SUBSIDY_COLOURS = {
    "hydrogen": "#1f78b4",
    "electricity": "#fdbf6f",
    "capex": "#6a3d9a",
}
CARBON_TAX_REVENUE_COLOUR = "#d73027"
NET_FISCAL_COLOUR = "#222222"
CARBON_COST_COLUMN = "unit_carbon_cost"
REQUIRED_COLUMNS = {
    "year",
    "iso3",
    "furnace_group_id",
    "product",
    "production",
    "capacity",
    "decision_year",
    "investment_decision",
    DIRECT_EMISSIONS,
    INDIRECT_EMISSIONS,
    CARBON_COST_COLUMN,
    *SUBSIDY_COLUMNS.values(),
}


def _prepare_plants(path: Path, year_from: int | None, year_to: int | None) -> pd.DataFrame:
    plants = pd.read_csv(path, usecols=lambda column: column in REQUIRED_COLUMNS, low_memory=False)
    if missing := REQUIRED_COLUMNS - set(plants.columns):
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    plants = plants.drop_duplicates(["year", "furnace_group_id"]).copy()
    numeric = [
        "year",
        "production",
        "capacity",
        "decision_year",
        DIRECT_EMISSIONS,
        INDIRECT_EMISSIONS,
        CARBON_COST_COLUMN,
        *SUBSIDY_COLUMNS.values(),
    ]
    for column in numeric:
        plants[column] = pd.to_numeric(plants[column], errors="coerce")
    plants = plants[plants["year"].notna()].copy()
    plants["year"] = plants["year"].astype(int)
    if year_from is not None:
        plants = plants[plants["year"].ge(year_from)]
    if year_to is not None:
        plants = plants[plants["year"].le(year_to)]
    plants["iso3"] = plants["iso3"].fillna("").astype(str).str.upper()
    plants["product"] = plants["product"].fillna("").astype(str).str.casefold()
    plants["investment_decision"] = plants["investment_decision"].fillna("none").astype(str).str.casefold()
    plants[numeric[1:]] = plants[numeric[1:]].fillna(0.0)
    return plants


def _scope_summary(plants: pd.DataFrame, countries: set[str] | None) -> pd.DataFrame:
    scoped = plants if countries is None else plants[plants["iso3"].isin(countries)]
    scoped = scoped[scoped["product"].isin(PRODUCTS)].copy()
    if scoped.empty:
        return pd.DataFrame()

    scoped["direct_tco2e"] = scoped[DIRECT_EMISSIONS]
    scoped["indirect_tco2e"] = scoped[INDIRECT_EMISSIONS]
    scoped["direct_indirect_tco2e"] = scoped["direct_tco2e"] + scoped["indirect_tco2e"]
    # ``unit_carbon_cost`` is the carbon charge paid per tonne of furnace output.
    # Keep revenue positive in the data; the fiscal plot displays it below zero.
    scoped["carbon_tax_revenue_usd"] = scoped[CARBON_COST_COLUMN].clip(lower=0.0) * scoped["production"]
    # Energy subsidy columns are USD/t output. They are paid on annual production.
    scoped["hydrogen_subsidy_usd"] = scoped[SUBSIDY_COLUMNS["hydrogen"]] * scoped["production"]
    scoped["electricity_subsidy_usd"] = scoped[SUBSIDY_COLUMNS["electricity"]] * scoped["production"]
    # CAPEX support is USD/t capacity and persists on the furnace record. Count it once,
    # in the year in which an actual investment decision is recorded.
    investment = scoped["decision_year"].eq(scoped["year"]) & ~scoped["investment_decision"].eq("none")
    scoped["capex_subsidy_usd"] = np.where(
        investment,
        scoped[SUBSIDY_COLUMNS["capex"]] * scoped["capacity"],
        0.0,
    )

    totals = scoped.groupby("year", as_index=False).agg(
        direct_tco2e=("direct_tco2e", "sum"),
        indirect_tco2e=("indirect_tco2e", "sum"),
        direct_indirect_tco2e=("direct_indirect_tco2e", "sum"),
        carbon_tax_revenue_usd=("carbon_tax_revenue_usd", "sum"),
        hydrogen_subsidy_usd=("hydrogen_subsidy_usd", "sum"),
        electricity_subsidy_usd=("electricity_subsidy_usd", "sum"),
        capex_subsidy_usd=("capex_subsidy_usd", "sum"),
    )
    for product in PRODUCTS:
        product_totals = (
            scoped[scoped["product"].eq(product)]
            .groupby("year")[["direct_tco2e", "indirect_tco2e", "direct_indirect_tco2e"]]
            .sum()
            .add_prefix(f"{product}_")
        )
        totals = totals.merge(product_totals, left_on="year", right_index=True, how="left")
    totals = totals.fillna(0.0).sort_values("year").reset_index(drop=True)
    totals["total_subsidy_usd"] = totals[["hydrogen_subsidy_usd", "electricity_subsidy_usd", "capex_subsidy_usd"]].sum(
        axis=1
    )
    subsidy_components = totals[["hydrogen_subsidy_usd", "electricity_subsidy_usd", "capex_subsidy_usd"]]
    totals["total_subsidy_spending_usd"] = subsidy_components.clip(lower=0.0).sum(axis=1)
    totals["net_fiscal_cost_usd"] = totals["total_subsidy_spending_usd"] - totals["carbon_tax_revenue_usd"]
    totals["cumulative_direct_tco2e"] = totals["direct_tco2e"].cumsum()
    totals["cumulative_direct_indirect_tco2e"] = totals["direct_indirect_tco2e"].cumsum()
    return totals


def summarise_scenario(
    scenario: ScenarioInput,
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    include_ukraine: bool = False,
) -> pd.DataFrame:
    """Return annual global and EU iron-and-steel emissions and subsidy expenditure."""
    plants = _prepare_plants(scenario.plants_csv, year_from, year_to)
    europe = _europe_iso3(scenario, include_ukraine)
    frames: list[pd.DataFrame] = []
    for scope, countries in (("Global", None), ("EU", europe)):
        frame = _scope_summary(plants, countries)
        if frame.empty:
            continue
        frame.insert(0, "scope", scope)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "scenario_title", scenario.title)
    result.insert(0, "scenario_index", scenario.index)
    return result


def build_summary(
    scenarios: Iterable[ScenarioInput],
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    include_ukraine: bool = False,
) -> pd.DataFrame:
    frames = [
        summarise_scenario(
            scenario,
            year_from=year_from,
            year_to=year_to,
            include_ukraine=include_ukraine,
        )
        for scenario in scenarios
    ]
    if not frames:
        raise ValueError("No completed scenarios were available")
    return pd.concat(frames, ignore_index=True)


def _plot_emissions(summary: pd.DataFrame, scope: str, cumulative: bool, output_path: Path) -> None:
    direct = "cumulative_direct_tco2e" if cumulative else "direct_tco2e"
    total = "cumulative_direct_indirect_tco2e" if cumulative else "direct_indirect_tco2e"
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=180, sharex=True, constrained_layout=True)
    for position, (index, axis) in enumerate(zip(SCENARIO_GRID_ORDER, axes.flat)):
        selected = summary[summary["scenario_index"].eq(index) & summary["scope"].eq(scope)].sort_values("year")
        if selected.empty:
            axis.axis("off")
            axis.set_title(f"{SCENARIO_TITLES[index]} — unavailable", fontsize=10)
            continue
        axis.plot(selected["year"], selected[direct] / 1e6, color="#d73027", linewidth=1.8, label="Direct")
        axis.plot(
            selected["year"],
            selected[total] / 1e6,
            color="#4575b4",
            linewidth=1.8,
            label="Direct + indirect",
        )
        axis.set_title(SCENARIO_TITLES[index], fontsize=11)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        axis.tick_params(labelsize=8)
        if not cumulative:
            axis.set_ylim(0.0, 5500.0)
        if position % 4 == 0:
            axis.set_ylabel("MtCO₂e")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=False)
    period = "Cumulative" if cumulative else "Annual"
    fig.suptitle(f"{period} iron + steel production emissions — {SCOPE_DISPLAY_NAMES[scope]}", fontsize=17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_subsidies(summary: pd.DataFrame, scope: str, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=180, sharex=True, constrained_layout=True)
    for position, (index, axis) in enumerate(zip(SCENARIO_GRID_ORDER, axes.flat)):
        selected = summary[summary["scenario_index"].eq(index) & summary["scope"].eq(scope)].sort_values("year")
        if selected.empty:
            axis.axis("off")
            axis.set_title(f"{SCENARIO_TITLES[index]} — unavailable", fontsize=10)
            continue
        bottom_positive = np.zeros(len(selected))
        bottom_negative = np.zeros(len(selected))
        for subsidy in ("hydrogen", "electricity", "capex"):
            values = selected[f"{subsidy}_subsidy_usd"].to_numpy(dtype=float) / 1e9
            bottoms = np.where(values >= 0, bottom_positive, bottom_negative)
            axis.bar(
                selected["year"],
                values,
                bottom=bottoms,
                color=SUBSIDY_COLOURS[subsidy],
                width=0.8,
                label=subsidy.title(),
            )
            bottom_positive += np.maximum(values, 0.0)
            bottom_negative += np.minimum(values, 0.0)
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(SCENARIO_TITLES[index], fontsize=11)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        axis.tick_params(labelsize=8)
        if position % 4 == 0:
            axis.set_ylabel("billion USD/year")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=3, frameon=False)
    fig.suptitle(f"Annual iron + steel subsidies by type — {SCOPE_DISPLAY_NAMES[scope]}", fontsize=17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_eu_fiscal_balance(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot European subsidy spending, carbon-tax revenue, and their signed sum."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=180, sharex=True, constrained_layout=True)
    for position, (index, axis) in enumerate(zip(SCENARIO_GRID_ORDER, axes.flat)):
        selected = summary[summary["scenario_index"].eq(index) & summary["scope"].eq("EU")].sort_values("year")
        if selected.empty:
            axis.axis("off")
            axis.set_title(f"{SCENARIO_TITLES[index]} — unavailable", fontsize=10)
            continue

        subsidy_bottom = np.zeros(len(selected))
        for subsidy in ("hydrogen", "electricity", "capex"):
            spending = selected[f"{subsidy}_subsidy_usd"].clip(lower=0.0).to_numpy(dtype=float) / 1e9
            axis.bar(
                selected["year"],
                spending,
                bottom=subsidy_bottom,
                color=SUBSIDY_COLOURS[subsidy],
                width=0.8,
            )
            subsidy_bottom += spending

        revenue = selected["carbon_tax_revenue_usd"].to_numpy(dtype=float) / 1e9
        axis.bar(
            selected["year"],
            -revenue,
            color=CARBON_TAX_REVENUE_COLOUR,
            width=0.8,
        )
        axis.plot(
            selected["year"],
            selected["net_fiscal_cost_usd"] / 1e9,
            color=NET_FISCAL_COLOUR,
            linewidth=1.6,
            marker="o",
            markersize=2.5,
            zorder=4,
        )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(SCENARIO_TITLES[index], fontsize=11)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        axis.tick_params(labelsize=8)
        if position % 4 == 0:
            axis.set_ylabel("billion USD/year")

    handles = [
        Patch(facecolor=SUBSIDY_COLOURS[subsidy], label=f"{subsidy.title()} subsidy")
        for subsidy in ("hydrogen", "electricity", "capex")
    ]
    handles.extend(
        [
            Patch(facecolor=CARBON_TAX_REVENUE_COLOUR, label="Carbon-tax revenue (negative)"),
            Line2D([0], [0], color=NET_FISCAL_COLOUR, marker="o", linewidth=1.6, label="Net fiscal cost"),
        ]
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=5, frameon=False)
    fig.suptitle(
        f"Annual iron + steel subsidies and carbon-tax revenue — {EUROPE_ANALYSIS_LABEL}",
        fontsize=17,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_methodology(path: Path, *, include_ukraine: bool) -> None:
    path.write_text(
        "\n".join(
            [
                "Emissions and subsidy time-series methodology",
                "",
                "- Sector totals include both iron-making and steelmaking furnace groups.",
                "- Furnace/feedstock duplicates are removed before summation.",
                "- Direct and indirect totals use the RS-inspired GHG result columns.",
                "- Cumulative emissions are the running sum from the first selected model year.",
                "- Annual emissions plots use a common 0-5500 MtCO2e axis; cumulative plots retain their larger scale.",
                "- Hydrogen/electricity costs equal annual production times the recorded USD/t subsidy.",
                "- CAPEX costs equal capacity times the recorded USD/t-capacity subsidy and are booked once,",
                "  in a year with a recorded investment decision; continuing eligibility is not booked again.",
                "- Carbon-tax revenue equals non-negative unit carbon cost times annual furnace production.",
                "- In the EU fiscal-balance plot, subsidies are positive, carbon-tax revenue is negative,",
                "  and the net line equals subsidy spending minus carbon-tax revenue.",
                f"- EU scope means {EUROPE_ANALYSIS_DESCRIPTION}.",
                f"- Ukraine is {'included' if include_ukraine else 'excluded'}; Turkey remains outside EU (MENA).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_analysis(
    results_dir: Path,
    *,
    output_dir: Path | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    skip_runs: Iterable[int] = (),
    include_ukraine: bool = False,
) -> list[Path]:
    results_dir = results_dir.resolve()
    destination = (output_dir or results_dir / "emissions_subsidy_timeseries").resolve()
    scenarios = discover_scenario_inputs(results_dir, skip_runs=skip_runs)
    summary = build_summary(
        scenarios,
        year_from=year_from,
        year_to=year_to,
        include_ukraine=include_ukraine,
    )
    destination.mkdir(parents=True, exist_ok=True)
    emissions_columns = [
        column
        for column in summary.columns
        if "tco2e" in column or column in {"scenario_index", "scenario_title", "scope", "year"}
    ]
    subsidy_columns = [
        column
        for column in summary.columns
        if "subsidy" in column or column in {"scenario_index", "scenario_title", "scope", "year"}
    ]
    emissions_csv = destination / "emissions_totals_by_year.csv"
    subsidies_csv = destination / "subsidy_costs_by_year.csv"
    fiscal_csv = destination / "eu_subsidies_carbon_revenue_by_year.csv"
    summary[emissions_columns].to_csv(emissions_csv, index=False)
    summary[subsidy_columns].to_csv(subsidies_csv, index=False)
    fiscal_columns = [
        "scenario_index",
        "scenario_title",
        "year",
        "hydrogen_subsidy_usd",
        "electricity_subsidy_usd",
        "capex_subsidy_usd",
        "total_subsidy_spending_usd",
        "carbon_tax_revenue_usd",
        "net_fiscal_cost_usd",
    ]
    summary.loc[summary["scope"].eq("EU"), fiscal_columns].to_csv(fiscal_csv, index=False)
    outputs = [emissions_csv, subsidies_csv, fiscal_csv]
    for scope in SCOPES:
        slug = scope.casefold()
        annual = destination / f"annual_emissions_{slug}_scenario_matrix.png"
        cumulative = destination / f"cumulative_emissions_{slug}_scenario_matrix.png"
        subsidies = destination / f"annual_subsidies_{slug}_scenario_matrix.png"
        _plot_emissions(summary, scope, False, annual)
        _plot_emissions(summary, scope, True, cumulative)
        _plot_subsidies(summary, scope, subsidies)
        outputs.extend([annual, cumulative, subsidies])
    fiscal_plot = destination / "annual_eu_subsidies_carbon_revenue_scenario_matrix.png"
    _plot_eu_fiscal_balance(summary, fiscal_plot)
    outputs.append(fiscal_plot)
    methodology = destination / "methodology.txt"
    write_methodology(methodology, include_ukraine=include_ukraine)
    outputs.append(methodology)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Directory containing master_input_* scenarios")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument("--skip-runs", type=int, nargs="*", default=[])
    parser.add_argument("--include-ukraine", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in run_analysis(
        args.results_dir,
        output_dir=args.output_dir,
        year_from=args.year_from,
        year_to=args.year_to,
        skip_runs=args.skip_runs,
        include_ukraine=args.include_ukraine,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
