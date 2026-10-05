"""European production and trade comparison for the 16-run scenario matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import AsinhNorm
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from .analysis_scope import EUROPE_ANALYSIS_ISO3, EUROPE_ANALYSIS_LABEL
from .europe_trade_summary import _location_iso3
from .route_classification import (
    CARBON_ROUTE_COMMODITIES,
    CARBON_ROUTE_HATCHES,
    CARBON_ROUTE_LABELS,
    source_route_lookup,
)
from .world_scenario_analysis import (
    COMMODITY_COLOURS,
    SCENARIO_COLUMN_LABELS,
    SCENARIO_GRID_ORDER,
    SCENARIO_ROW_LABELS,
    SCENARIO_TITLES,
)


X_LABELS = ("E", "EC", "ES", "ECS")
Y_LABELS = ("W", "WC", "WS", "WCS")
POLICY_LABELS = ("", "Carbon Tax", "Subsidies", "C+S")
GRID_COLUMN_POLICY_LABELS = ("", "Subsidies", "Carbon Tax", "C+S")
GRID_ROW_POLICY_LABELS = ("C+S", "Carbon Tax", "Subsidies", "")
METALLIC_INPUTS = {
    "scrap",
    "hot_metal",
    "pig_iron",
    "dri_low",
    "dri_mid",
    "dri_high",
    "hbi_low",
    "hbi_mid",
    "hbi_high",
    "electrolytic_iron",
    "liquid_iron",
}
LOCATION_COUNTRY_ALIASES = {
    "bosnia & herzegovina": "BIH",
}
IRON_TECHNOLOGY_COLOURS = {
    "BF": "#4c78a8",
    "BF_CHARCOAL": "#9c755f",
    "DRI hydrogen": "#72b7b2",
    "DRI natural_gas": "#f2cf5b",
    "E-WIN": "#b279a2",
    "SR": "#f28e2c",
}
STEEL_TECHNOLOGY_COLOURS = {
    "BOF": "#e45756",
    "EAF": "#54a24b",
    "EAF scrap": "#a6dda0",
}
CARBON_STORED = "carbon_breakdown - co2_stored"
CARBON_UTILISED = "carbon_breakdown - co2_utilised"
DEMAND_SHEET = "Demand and scrap availability"
STEEL_DEMAND_METRIC = "crude steel consumption for forming [kt]"
DEFAULT_MASTER_EXCEL = Path.home() / ".steelo" / "data_cache" / "master-input-v2.0.0" / "master_input_0.xlsx"


@dataclass(frozen=True)
class ScenarioInput:
    index: int
    title: str
    source: str
    run_dir: Path
    plants_csv: Path


def _completed_run(scenario_dir: Path) -> tuple[Path, Path] | None:
    candidates = []
    latest = scenario_dir / "latest"
    if latest.is_dir():
        candidates.append(latest)
    candidates.extend(sorted(scenario_dir.glob("sim_*"), reverse=True))
    for run_dir in candidates:
        post_processed = sorted(run_dir.glob("post_processed_*.csv"))
        if post_processed and (run_dir / "TM").is_dir():
            return run_dir.resolve(), post_processed[-1].resolve()
    return None


def discover_scenario_inputs(
    results_dir: Path,
    *,
    skip_runs: Iterable[int] = (),
) -> list[ScenarioInput]:
    """Resolve completed scenarios strictly below the supplied results directory."""
    skipped = set(skip_runs)
    scenarios: list[ScenarioInput] = []
    for index, title in enumerate(SCENARIO_TITLES):
        if index in skipped:
            continue
        completed = _completed_run(results_dir / f"master_input_{index}")
        if completed is not None:
            run_dir, plants_csv = completed
            scenarios.append(ScenarioInput(index, title, "primary", run_dir, plants_csv))
    if not scenarios:
        raise FileNotFoundError(f"No completed scenarios found below {results_dir}")
    return scenarios


def _europe_iso3(scenario: ScenarioInput, include_ukraine: bool = False) -> set[str]:
    """Return the fixed comparison scope, independent of model region labels."""
    _ = scenario  # Kept in the signature for callers that already pass a scenario.
    countries = set(EUROPE_ANALYSIS_ISO3)
    if include_ukraine:
        countries.add("UKR")
    else:
        countries.discard("UKR")
    return countries


def _trade_location_iso3(value: object) -> str:
    iso3 = _location_iso3(value)
    if iso3:
        return iso3
    text = str(value).casefold()
    return next((code for name, code in LOCATION_COUNTRY_ALIASES.items() if name in text), "")


def _steel_origin_by_year(plants_csv: Path) -> pd.DataFrame:
    """Split steel production by scrap share without duplicating furnace rows."""
    columns = ["year", "region", "iso3", "furnace_group_id", "product", "production", "feedstock", "demand"]
    plants = pd.read_csv(plants_csv, usecols=lambda column: column in columns, low_memory=False)
    missing = set(columns) - set(plants.columns)
    if missing:
        raise ValueError(f"{plants_csv} is missing columns: {', '.join(sorted(missing))}")
    steel = plants[plants["product"].astype(str).str.casefold().eq("steel")].copy()
    steel["feedstock"] = steel["feedstock"].fillna("").astype(str).str.casefold()
    steel["demand"] = pd.to_numeric(steel["demand"], errors="coerce").fillna(0.0).clip(lower=0.0)
    metadata = steel.drop_duplicates(["year", "furnace_group_id"])[
        ["year", "region", "iso3", "furnace_group_id", "production"]
    ].copy()
    metadata["production"] = pd.to_numeric(metadata["production"], errors="coerce").fillna(0.0)
    metallic = steel[steel["feedstock"].isin(METALLIC_INPUTS)].copy()
    metallic["scrap_demand"] = metallic["demand"].where(metallic["feedstock"].eq("scrap"), 0.0)
    shares = metallic.groupby(["year", "furnace_group_id"], as_index=False).agg(
        metallic_demand=("demand", "sum"), scrap_demand=("scrap_demand", "sum")
    )
    data = metadata.merge(shares, on=["year", "furnace_group_id"], how="left").fillna(
        {"metallic_demand": 0.0, "scrap_demand": 0.0}
    )
    data["secondary_share"] = np.where(
        data["metallic_demand"].gt(0), data["scrap_demand"] / data["metallic_demand"], 0.0
    )
    data["secondary_share"] = data["secondary_share"].clip(0.0, 1.0)
    data["steel_secondary_mt"] = data["production"] * data["secondary_share"] / 1e6
    data["steel_primary_mt"] = data["production"] * (1.0 - data["secondary_share"]) / 1e6
    return data


def load_production_matrix(
    scenarios: Iterable[ScenarioInput], years: Iterable[int], *, include_ukraine: bool = False
) -> pd.DataFrame:
    """Return European iron, total steel, primary steel, and secondary steel production."""
    years = tuple(int(year) for year in years)
    records: list[dict[str, object]] = []
    for scenario in scenarios:
        europe = _europe_iso3(scenario, include_ukraine)
        iron_columns = ["year", "iso3", "furnace_group_id", "product", "production"]
        iron = pd.read_csv(scenario.plants_csv, usecols=iron_columns, low_memory=False)
        iron = iron.drop_duplicates(["year", "furnace_group_id"])
        iron = iron[
            iron["iso3"].astype(str).str.upper().isin(europe) & iron["product"].astype(str).str.casefold().eq("iron")
        ].copy()
        iron["production"] = pd.to_numeric(iron["production"], errors="coerce").fillna(0.0)
        iron = iron.groupby("year")["production"].sum() / 1e6
        origin = _steel_origin_by_year(scenario.plants_csv)
        origin = origin[origin["iso3"].astype(str).str.upper().isin(europe)]
        origin = origin.groupby("year", as_index=True)[["steel_primary_mt", "steel_secondary_mt"]].sum()
        for year in years:
            split = origin.loc[year] if year in origin.index else pd.Series(dtype=float)
            primary = float(split.get("steel_primary_mt", 0.0))
            secondary = float(split.get("steel_secondary_mt", 0.0))
            records.append(
                {
                    "scenario_index": scenario.index,
                    "scenario_title": scenario.title,
                    "source": scenario.source,
                    "year": year,
                    "x": X_LABELS[scenario.index % 4],
                    "y": Y_LABELS[scenario.index // 4],
                    "iron_production_mt": float(iron.get(year, 0.0)),
                    "steel_production_mt": primary + secondary,
                    "steel_primary_mt": primary,
                    "steel_secondary_mt": secondary,
                }
            )
    return pd.DataFrame.from_records(records)


def load_europe_steel_demand(
    master_excel: Path,
    years: Iterable[int],
    *,
    demand_scenario: str = "BAU",
    include_ukraine: bool = False,
) -> pd.DataFrame:
    """Load total steel demand for the fixed EU27 + EFTA + UK analysis scope.

    Demand is read from the master workbook in kt/a and returned in both tonnes
    and Mt/a. Metric and scenario matching ignore surrounding whitespace and
    letter case because historical workbooks contain padded metric labels.
    """
    master_excel = Path(master_excel).expanduser().resolve()
    if not master_excel.is_file():
        raise FileNotFoundError(f"Master Excel file not found: {master_excel}")

    requested_years = tuple(dict.fromkeys(int(year) for year in years))
    demand = pd.read_excel(master_excel, sheet_name=DEMAND_SHEET)
    required = {"ISO-3 code", "Metric", "Scenario", "Unit"}
    if missing := required - set(demand.columns):
        raise ValueError(f"{master_excel} sheet '{DEMAND_SHEET}' is missing: {', '.join(sorted(missing))}")

    year_columns = {int(column): column for column in demand.columns if str(column).strip().isdigit()}
    missing_years = [year for year in requested_years if year not in year_columns]
    if missing_years:
        raise ValueError(
            f"{master_excel} sheet '{DEMAND_SHEET}' has no demand columns for years: "
            f"{', '.join(map(str, missing_years))}"
        )

    metric = demand["Metric"].fillna("").astype(str).str.strip().str.casefold()
    scenario = demand["Scenario"].fillna("").astype(str).str.strip().str.casefold()
    selected = demand[
        metric.eq(STEEL_DEMAND_METRIC.casefold()) & scenario.eq(str(demand_scenario).strip().casefold())
    ].copy()
    if selected.empty:
        available = sorted(demand.loc[metric.eq(STEEL_DEMAND_METRIC.casefold()), "Scenario"].astype(str).unique())
        raise ValueError(
            f"No '{STEEL_DEMAND_METRIC}' rows for scenario '{demand_scenario}' in {master_excel}; "
            f"available scenarios: {available}"
        )

    scope = set(EUROPE_ANALYSIS_ISO3)
    if include_ukraine:
        scope.add("UKR")
    selected["iso3"] = selected["ISO-3 code"].fillna("").astype(str).str.strip().str.upper()
    selected = selected[selected["iso3"].isin(scope)].copy()

    unit_factors = {"t": 1.0, "kt": 1e3, "mt": 1e6}
    units = selected["Unit"].fillna("").astype(str).str.strip().str.casefold()
    unknown_units = sorted(set(units) - set(unit_factors))
    if unknown_units:
        raise ValueError(f"Unsupported steel-demand units in {master_excel}: {unknown_units}")
    factors = units.map(unit_factors)

    records = []
    for year in requested_years:
        values = pd.to_numeric(selected[year_columns[year]], errors="coerce").fillna(0.0)
        demand_t = float((values * factors).sum())
        records.append({"year": year, "steel_demand_t": demand_t, "steel_demand_mt": demand_t / 1e6})
    return pd.DataFrame.from_records(records)


def calculate_primary_iron_self_sufficiency(
    production: pd.DataFrame,
    steel_demand: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate primary-iron production as a share of total regional steel demand.

    ``primary_iron_self_sufficiency_pct`` is bounded to the requested 0--100%
    interpretation. ``primary_iron_to_demand_ratio`` remains uncapped so export
    surpluses are not hidden in the accompanying CSV.
    """
    production_columns = ["scenario_index", "scenario_title", "year", "iron_production_mt"]
    if missing := set(production_columns) - set(production.columns):
        raise ValueError(f"Production data is missing columns: {', '.join(sorted(missing))}")
    if missing := {"year", "steel_demand_mt"} - set(steel_demand.columns):
        raise ValueError(f"Demand data is missing columns: {', '.join(sorted(missing))}")

    result = production[production_columns].merge(
        steel_demand[["year", "steel_demand_mt"]], on="year", how="left", validate="many_to_one"
    )
    result["primary_iron_to_demand_ratio"] = np.where(
        result["steel_demand_mt"].gt(0),
        result["iron_production_mt"] / result["steel_demand_mt"],
        np.nan,
    )
    result["primary_iron_self_sufficiency_pct"] = (
        result["primary_iron_to_demand_ratio"].clip(lower=0.0, upper=1.0) * 100.0
    )
    return result.sort_values(["scenario_index", "year"]).reset_index(drop=True)


def plot_primary_iron_self_sufficiency_matrix(data: pd.DataFrame, output_path: Path) -> None:
    """Plot annual primary-iron self-sufficiency in the shared 4x4 scenario order."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), dpi=180, sharex=True, sharey=True)
    fig.suptitle(
        f"{EUROPE_ANALYSIS_LABEL} primary-iron self-sufficiency\n"
        "Regional primary iron production / regional steel demand",
        fontsize=17,
    )
    all_years = sorted(pd.to_numeric(data["year"], errors="coerce").dropna().astype(int).unique())
    year_ticks = [year for year in all_years if year % 5 == 0]
    if all_years and not year_ticks:
        year_ticks = all_years

    for position, (index, axis) in enumerate(zip(SCENARIO_GRID_ORDER, axes.flat)):
        selected = data[data["scenario_index"].eq(index)].sort_values("year")
        if selected.empty:
            axis.axis("off")
            axis.set_title(f"{SCENARIO_TITLES[index]} — unavailable")
            continue
        axis.plot(
            selected["year"],
            selected["primary_iron_self_sufficiency_pct"],
            color="#1f78b4",
            marker="o",
            markersize=2.8,
            linewidth=1.6,
        )
        axis.set_title(SCENARIO_TITLES[index])
        axis.set_ylim(0.0, 100.0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
        axis.grid(axis="both", linestyle="--", linewidth=0.5, color="grey", alpha=0.45)
        axis.set_axisbelow(True)
        if year_ticks:
            axis.set_xticks(year_ticks)
        if position % 4 == 0:
            axis.set_ylabel("Self-sufficiency")
        if position // 4 == 3:
            axis.set_xlabel("Year")
        else:
            axis.tick_params(axis="x", labelbottom=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_europe_technology_portfolio(
    scenarios: Iterable[ScenarioInput], *, include_ukraine: bool = False
) -> pd.DataFrame:
    """Aggregate European capacity and production once per furnace and classify technology routes.

    DRI is split by its chosen reductant. An EAF is labelled ``EAF scrap`` only if
    all of its positive metallic input is scrap; mixed-charge EAFs remain ``EAF``.
    Carbon breakdown contributions are summed over feedstock rows and converted to
    annual tonnes using the furnace's production before aggregation.
    """
    required = {
        "year",
        "iso3",
        "furnace_group_id",
        "product",
        "technology",
        "chosen_reductant",
        "capacity",
        "production",
        "feedstock",
        "demand",
    }
    requested = required | {CARBON_STORED, CARBON_UTILISED}
    frames: list[pd.DataFrame] = []
    for scenario in scenarios:
        plants = pd.read_csv(
            scenario.plants_csv,
            usecols=lambda column: column in requested,
            low_memory=False,
        )
        if missing := required - set(plants.columns):
            raise ValueError(f"{scenario.plants_csv} is missing columns: {', '.join(sorted(missing))}")
        for column in (CARBON_STORED, CARBON_UTILISED):
            if column not in plants:
                plants[column] = 0.0
        for column in ("year", "capacity", "production", "demand", CARBON_STORED, CARBON_UTILISED):
            plants[column] = pd.to_numeric(plants[column], errors="coerce").fillna(0.0)
        plants["iso3"] = plants["iso3"].fillna("").astype(str).str.upper()
        plants["product"] = plants["product"].fillna("").astype(str).str.casefold()
        plants["technology"] = plants["technology"].fillna("Unknown").astype(str)
        plants["chosen_reductant"] = (
            plants["chosen_reductant"].fillna("").astype(str).str.strip().str.casefold().str.replace(" ", "_")
        )
        plants["feedstock"] = plants["feedstock"].fillna("").astype(str).str.casefold()
        plants = plants[plants["iso3"].isin(_europe_iso3(scenario, include_ukraine))].copy()
        if plants.empty:
            continue

        keys = ["year", "furnace_group_id"]
        metadata = plants.drop_duplicates(keys)[
            keys + ["product", "technology", "chosen_reductant", "capacity", "production"]
        ].copy()
        carbon = plants.groupby(keys, as_index=False).agg(
            co2_stored_tco2_per_t=(CARBON_STORED, "sum"),
            co2_utilised_tco2_per_t=(CARBON_UTILISED, "sum"),
        )
        metallic = plants[plants["feedstock"].isin(METALLIC_INPUTS) & plants["demand"].gt(0)].copy()
        metallic["scrap_demand"] = metallic["demand"].where(metallic["feedstock"].eq("scrap"), 0.0)
        charge = metallic.groupby(keys, as_index=False).agg(
            metallic_demand_t=("demand", "sum"),
            scrap_demand_t=("scrap_demand", "sum"),
        )
        data = metadata.merge(carbon, on=keys, how="left").merge(charge, on=keys, how="left")
        data = data.fillna(
            {
                "co2_stored_tco2_per_t": 0.0,
                "co2_utilised_tco2_per_t": 0.0,
                "metallic_demand_t": 0.0,
                "scrap_demand_t": 0.0,
            }
        )
        data["technology_route"] = data["technology"]
        dri = data["technology"].str.casefold().eq("dri")
        data.loc[dri, "technology_route"] = "DRI " + data.loc[dri, "chosen_reductant"].replace("", "unknown")
        scrap_only = (
            data["technology"].str.casefold().eq("eaf")
            & data["metallic_demand_t"].gt(0)
            & np.isclose(data["metallic_demand_t"], data["scrap_demand_t"])
        )
        data.loc[scrap_only, "technology_route"] = "EAF scrap"
        data["co2_stored_t"] = data["co2_stored_tco2_per_t"] * data["production"]
        data["co2_utilised_t"] = data["co2_utilised_tco2_per_t"] * data["production"]
        summary = data.groupby(["year", "product", "technology_route"], as_index=False).agg(
            capacity_t=("capacity", "sum"),
            production_t=("production", "sum"),
            co2_stored_t=("co2_stored_t", "sum"),
            co2_utilised_t=("co2_utilised_t", "sum"),
            furnace_groups=("furnace_group_id", "nunique"),
        )
        summary.insert(0, "scenario_title", scenario.title)
        summary.insert(0, "scenario_index", scenario.index)
        frames.append(summary)
    if not frames:
        return pd.DataFrame(
            columns=[
                "scenario_index",
                "scenario_title",
                "year",
                "product",
                "technology_route",
                "capacity_t",
                "production_t",
                "co2_stored_t",
                "co2_utilised_t",
                "furnace_groups",
            ]
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["scenario_index", "year", "product", "technology_route"])
        .reset_index(drop=True)
    )


def _ordered_technologies(data: pd.DataFrame, product: str) -> list[str]:
    preferred = IRON_TECHNOLOGY_COLOURS if product == "iron" else STEEL_TECHNOLOGY_COLOURS
    available = set(data.loc[data["product"].eq(product), "technology_route"].astype(str))
    return [technology for technology in preferred if technology in available] + sorted(available - set(preferred))


def plot_europe_technology_capacity_matrix(
    portfolio: pd.DataFrame,
    years: Iterable[int],
    product: str,
    output_path: Path,
) -> None:
    """Plot installed capacity by technology in a 4×4 scenario matrix."""
    years = tuple(int(year) for year in years)
    selected = portfolio[portfolio["year"].isin(years) & portfolio["product"].eq(product)].copy()
    technologies = _ordered_technologies(selected, product)
    palette = IRON_TECHNOLOGY_COLOURS if product == "iron" else STEEL_TECHNOLOGY_COLOURS
    totals = selected.groupby(["scenario_index", "year"])["capacity_t"].sum() / 1e6
    ymax = max(float(totals.max()) * 1.12 if not totals.empty else 1.0, 1.0)
    product_label = "Ironmaking" if product == "iron" else "Steelmaking"
    fig, axes = _scenario_axes(
        f"{EUROPE_ANALYSIS_LABEL} {product_label.lower()} installed capacity by technology",
        figsize=(16, 14),
    )
    for position, (index, axis) in enumerate(zip(SCENARIO_GRID_ORDER, axes.flat)):
        scenario = selected[selected["scenario_index"].eq(index)]
        if scenario.empty:
            axis.axis("off")
            axis.set_title(f"{SCENARIO_TITLES[index]} — unavailable")
            continue
        table = scenario.pivot_table(
            index="year",
            columns="technology_route",
            values="capacity_t",
            aggfunc="sum",
            fill_value=0.0,
        )
        table = table.reindex(index=years, columns=technologies, fill_value=0.0) / 1e6
        table.plot(
            kind="bar",
            stacked=True,
            ax=axis,
            color=[palette.get(technology, "#bab0ac") for technology in technologies],
            legend=False,
            width=0.78,
        )
        axis.set_title(SCENARIO_TITLES[index])
        axis.set_xlabel("")
        axis.set_ylim(0.0, ymax)
        axis.set_xticklabels([str(year) for year in years], rotation=0)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, color="grey", alpha=0.45)
        axis.set_axisbelow(True)
        if position % 4 == 0:
            axis.set_ylabel("Installed capacity [Mt/a]")
        for x, total in enumerate(table.sum(axis=1)):
            if total > 0:
                axis.text(x, total + ymax * 0.012, f"{total:.0f}", ha="center", va="bottom", fontsize=7)
    handles = [Patch(facecolor=palette.get(technology, "#bab0ac"), label=technology) for technology in technologies]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def collect_europe_trade(
    scenarios: Iterable[ScenarioInput], *, include_ukraine: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return external European bilateral flows and commodity totals for all runs."""
    all_flows: list[pd.DataFrame] = []
    for scenario in scenarios:
        europe = _europe_iso3(scenario, include_ukraine)
        plant_columns = ["year", "furnace_group_id", "technology", "chosen_reductant"]
        plants = pd.read_csv(scenario.plants_csv, usecols=lambda column: column in plant_columns, low_memory=False)
        for path in sorted((scenario.run_dir / "TM").glob("steel_trade_allocations_*.csv")):
            try:
                year = int(path.stem.rsplit("_", 1)[1])
            except ValueError:
                continue
            trade = pd.read_csv(
                path,
                usecols=["commodity", "source_id", "source_location", "destination_location", "allocated_volume"],
            )
            trade["country_from"] = trade["source_location"].map(_trade_location_iso3)
            trade["country_to"] = trade["destination_location"].map(_trade_location_iso3)
            trade["carbon_route"] = "not_applicable"
            if "source_id" in trade:
                routes = source_route_lookup(plants, year).set_index("furnace_group_id")["carbon_route"]
                route_sensitive = trade["commodity"].astype(str).str.casefold().isin(CARBON_ROUTE_COMMODITIES)
                trade.loc[route_sensitive, "carbon_route"] = (
                    trade.loc[route_sensitive, "source_id"].astype(str).map(routes).fillna("unknown")
                )
            unresolved = trade["country_from"].eq("") | trade["country_to"].eq("")
            if unresolved.any():
                raise ValueError(f"Could not resolve countries in {int(unresolved.sum())} rows of {path}")
            source_europe = trade["country_from"].isin(europe)
            destination_europe = trade["country_to"].isin(europe)
            boundary = trade[source_europe ^ destination_europe].copy()
            boundary["direction"] = np.where(destination_europe[source_europe ^ destination_europe], "import", "export")
            boundary["year"] = year
            boundary["scenario_index"] = scenario.index
            boundary["scenario_title"] = scenario.title
            all_flows.append(
                boundary.groupby(
                    [
                        "scenario_index",
                        "scenario_title",
                        "year",
                        "direction",
                        "country_from",
                        "country_to",
                        "commodity",
                        "carbon_route",
                    ],
                    as_index=False,
                )["allocated_volume"]
                .sum()
                .rename(columns={"allocated_volume": "volume_t"})
            )
    if not all_flows:
        return pd.DataFrame(), pd.DataFrame()
    flows = pd.concat(all_flows, ignore_index=True)
    totals = flows.pivot_table(
        index=["scenario_index", "scenario_title", "year", "commodity", "carbon_route"],
        columns="direction",
        values="volume_t",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for column in ("import", "export"):
        if column not in totals:
            totals[column] = 0.0
    totals = totals.rename(columns={"import": "imports_t", "export": "exports_t"})
    totals["net_imports_t"] = totals["imports_t"] - totals["exports_t"]
    return flows, totals


def aggregate_trade_partners(flows: pd.DataFrame, threshold_t: float = 1e6) -> pd.DataFrame:
    """Aggregate small external partners into Rest for readable labels."""
    data = flows.copy()
    if "carbon_route" not in data:
        data["carbon_route"] = "not_applicable"
    data["partner_country"] = data["country_from"].where(data["direction"].eq("import"), data["country_to"])
    data = data.groupby(
        ["scenario_index", "scenario_title", "year", "direction", "commodity", "carbon_route", "partner_country"],
        as_index=False,
    )["volume_t"].sum()
    data["country_plot"] = data["partner_country"].where(data["volume_t"].gt(threshold_t), "Rest")
    return data.groupby(
        ["scenario_index", "scenario_title", "year", "direction", "commodity", "carbon_route", "country_plot"],
        as_index=False,
    )["volume_t"].sum()


def _scenario_axes(title: str, figsize: tuple[float, float] = (16, 15)) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(4, 4, figsize=figsize, dpi=180, sharex=False, sharey=False)
    fig.suptitle(title, fontsize=17)
    return fig, axes


def plot_production_matrices(data: pd.DataFrame, years: Iterable[int], output_path: Path) -> None:
    metrics = {
        "iron_production_mt": "Iron production",
        "steel_primary_mt": "Primary steel production",
        "steel_secondary_mt": "Secondary steel production",
    }
    years = tuple(years)
    fig, axes = plt.subplots(len(years), len(metrics), figsize=(15, 9), dpi=180, squeeze=False, constrained_layout=True)
    for column, (metric, label) in enumerate(metrics.items()):
        values = data[metric].dropna().to_numpy(dtype=float)
        vmax = max(float(values.max()) if values.size else 1.0, 1.0)
        positive = values[values > 0]
        linear_width = max(float(np.median(positive)) * 0.2 if positive.size else 1.0, 1.0)
        norm = AsinhNorm(linear_width=linear_width, vmin=0.0, vmax=vmax)
        for row, year in enumerate(years):
            axis = axes[row, column]
            selected = data[data["year"] == year]
            matrix = selected.pivot(index="y", columns="x", values=metric).reindex(
                index=SCENARIO_ROW_LABELS,
                columns=SCENARIO_COLUMN_LABELS,
            )
            cmap = plt.get_cmap("RdYlGn").copy()
            cmap.set_bad("#D9D9D9")
            image = axis.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap=cmap, norm=norm)
            axis.set_xticks(range(4), GRID_COLUMN_POLICY_LABELS, rotation=40, ha="right")
            axis.set_yticks(range(4), GRID_ROW_POLICY_LABELS)
            axis.set_title(f"{label} — {year}")
            axis.set_xlabel("Europe")
            if column == 0:
                axis.set_ylabel("World")
            for r in range(4):
                for c in range(4):
                    value = matrix.iloc[r, c]
                    text = f"{value:.1f}" if pd.notna(value) else "N/A"
                    axis.text(c, r, text, ha="center", va="center", fontsize=8)
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label("Production [Mt/a]", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_net_trade_matrix(
    totals: pd.DataFrame,
    years: Iterable[int],
    output_path: Path,
    *,
    include_ore: bool = False,
    label_threshold_mt: float = 20.0,
) -> None:
    years = tuple(years)
    data = totals.copy()
    if not include_ore:
        data = data[~data["commodity"].astype(str).str.startswith("io_")]
    commodities = sorted(data["commodity"].unique())
    colours = {commodity: COMMODITY_COLOURS.get(commodity, "#bdbdbd") for commodity in commodities}
    fig, axes = _scenario_axes(f"{EUROPE_ANALYSIS_LABEL} net trade by commodity")
    for position, (index, axis) in enumerate(zip(SCENARIO_GRID_ORDER, axes.flat)):
        selected = data[data["scenario_index"] == index]
        if selected.empty:
            axis.axis("off")
            axis.set_title(f"{SCENARIO_TITLES[index]} — unavailable")
            continue
        table = selected.pivot_table(
            index="year", columns=["commodity", "carbon_route"], values="net_imports_t", aggfunc="sum", fill_value=0.0
        )
        table = table.reindex(years, fill_value=0.0) / 1e6
        table.plot(
            kind="bar",
            stacked=True,
            ax=axis,
            color=[colours[commodity] for commodity, _route in table.columns],
            legend=False,
        )
        for container, (_commodity, route) in zip(axis.containers, table.columns):
            for patch in container.patches:
                patch.set_hatch(CARBON_ROUTE_HATCHES.get(route, ""))
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, color="grey", alpha=0.5)
        axis.set_axisbelow(True)
        axis.set_title(SCENARIO_TITLES[index])
        axis.set_xlabel("")
        if position % 4 == 0:
            axis.set_ylabel("Net imports [Mt/a]")
        if position // 4 != 3:
            axis.set_xticklabels([])
        for patch in axis.patches:
            height = patch.get_height()
            if abs(height) >= label_threshold_mt:
                axis.text(
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_y() + height / 2,
                    f"{height:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
    handles = [Patch(facecolor=colours[name], label=name.replace("_", " ").title()) for name in commodities]
    iron_routes = sorted(set(data["carbon_route"]) - {"not_applicable"})
    handles += [
        Patch(
            facecolor="white",
            edgecolor="#555555",
            hatch=CARBON_ROUTE_HATCHES.get(route, ""),
            label=f"Iron route — {CARBON_ROUTE_LABELS.get(route, route)}",
        )
        for route in iron_routes
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_trade_partner_matrix(
    partners: pd.DataFrame,
    year: int,
    output_path: Path,
    *,
    country_label_threshold_mt: float = 5.0,
) -> None:
    data = partners[partners["year"] == year].copy()
    commodities = sorted(data["commodity"].unique())
    colours = {commodity: COMMODITY_COLOURS.get(commodity, "#bdbdbd") for commodity in commodities}
    fig, axes = _scenario_axes(f"{EUROPE_ANALYSIS_LABEL} trade partners — {year}")
    for position, (index, axis) in enumerate(zip(SCENARIO_GRID_ORDER, axes.flat)):
        selected = data[data["scenario_index"] == index]
        if selected.empty:
            axis.axis("off")
            axis.set_title(f"{SCENARIO_TITLES[index]} — unavailable")
            continue
        label_keys: set[tuple[str, str, str]] = set()
        for direction in ("import", "export"):
            largest = selected[selected["direction"] == direction].nlargest(5, "volume_t")
            label_keys.update(
                (str(item.direction), str(item.commodity), str(item.country_plot))
                for item in largest.itertuples(index=False)
                if float(item.volume_t) / 1e6 >= country_label_threshold_mt
            )
        positive_bottom = negative_bottom = 0.0
        for commodity in commodities:
            commodity_rows = selected[selected["commodity"] == commodity]
            for direction in ("import", "export"):
                rows = commodity_rows[commodity_rows["direction"] == direction].copy()
                rows["is_rest"] = rows["country_plot"].eq("Rest")
                rows = rows.sort_values(["is_rest", "volume_t"], ascending=[True, False])
                for item in rows.itertuples(index=False):
                    volume_mt = float(item.volume_t) / 1e6
                    if direction == "import":
                        height, bottom = volume_mt, positive_bottom
                        positive_bottom += volume_mt
                    else:
                        height, bottom = -volume_mt, negative_bottom
                        negative_bottom -= volume_mt
                    axis.bar(0, height, bottom=bottom, color=colours[commodity], edgecolor="white", linewidth=0.45)
                    axis.patches[-1].set_hatch(CARBON_ROUTE_HATCHES.get(item.carbon_route, ""))
                    label_key = (str(item.direction), str(item.commodity), str(item.country_plot))
                    if label_key in label_keys:
                        axis.text(0, bottom + height / 2, item.country_plot, ha="center", va="center", fontsize=6)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, color="grey", alpha=0.4)
        axis.set_axisbelow(True)
        axis.set_xticks([0], [str(year)])
        axis.set_title(SCENARIO_TITLES[index])
        if position % 4 == 0:
            axis.set_ylabel("Imports (+) / exports (−) [Mt/a]")
    handles = [Patch(facecolor=colours[name], label=name.replace("_", " ").title()) for name in commodities]
    iron_routes = sorted(set(data["carbon_route"]) - {"not_applicable"})
    handles += [
        Patch(
            facecolor="white",
            edgecolor="#555555",
            hatch=CARBON_ROUTE_HATCHES.get(route, ""),
            label=f"Iron route — {CARBON_ROUTE_LABELS.get(route, route)}",
        )
        for route in iron_routes
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_europe_scenario_analysis(
    results_dir: Path,
    *,
    output_dir: Path | None = None,
    master_excel: Path = DEFAULT_MASTER_EXCEL,
    demand_scenario: str = "BAU",
    skip_runs: Iterable[int] = (),
    production_years: Iterable[int] = (2040, 2050),
    self_sufficiency_years: Iterable[int] = tuple(range(2025, 2061)),
    trade_years: Iterable[int] = (2030, 2035, 2040, 2045, 2050),
    partner_year: int = 2050,
    portfolio_years: Iterable[int] = (2030, 2040, 2050, 2060),
    include_ukraine: bool = False,
) -> list[Path]:
    results_dir = results_dir.resolve()
    destination = (output_dir or results_dir / "europe_scenario_post_analysis").resolve()
    scenarios = discover_scenario_inputs(
        results_dir,
        skip_runs=skip_runs,
    )
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "scenario_manifest.csv"
    pd.DataFrame(
        [
            {
                "scenario_index": item.index,
                "scenario_title": item.title,
                "source": item.source,
                "run_dir": str(item.run_dir),
            }
            for item in scenarios
        ]
    ).to_csv(manifest_path, index=False)

    production = load_production_matrix(scenarios, production_years, include_ukraine=include_ukraine)
    production_path = destination / "europe_production_matrix.csv"
    production_plot = destination / "europe_production_matrices.png"
    production.to_csv(production_path, index=False)
    plot_production_matrices(production, production_years, production_plot)

    self_sufficiency_years = tuple(int(year) for year in self_sufficiency_years)
    self_sufficiency_production = load_production_matrix(
        scenarios, self_sufficiency_years, include_ukraine=include_ukraine
    )
    steel_demand = load_europe_steel_demand(
        master_excel,
        self_sufficiency_years,
        demand_scenario=demand_scenario,
        include_ukraine=include_ukraine,
    )
    self_sufficiency = calculate_primary_iron_self_sufficiency(self_sufficiency_production, steel_demand)
    self_sufficiency_path = destination / "europe_primary_iron_self_sufficiency.csv"
    self_sufficiency_plot = destination / "europe_primary_iron_self_sufficiency_scenario_matrix.png"
    self_sufficiency.to_csv(self_sufficiency_path, index=False)
    plot_primary_iron_self_sufficiency_matrix(self_sufficiency, self_sufficiency_plot)

    portfolio = load_europe_technology_portfolio(scenarios, include_ukraine=include_ukraine)
    portfolio_years = tuple(int(year) for year in portfolio_years)
    capacity = portfolio[portfolio["year"].isin(portfolio_years)][
        [
            "scenario_index",
            "scenario_title",
            "year",
            "product",
            "technology_route",
            "capacity_t",
            "furnace_groups",
        ]
    ].copy()
    capacity["capacity_mt"] = capacity["capacity_t"] / 1e6
    year_slug = "_".join(str(year) for year in portfolio_years)
    capacity_path = destination / f"europe_technology_capacity_{year_slug}.csv"
    iron_capacity_plot = destination / "europe_iron_capacity_by_technology_scenario_matrix.png"
    steel_capacity_plot = destination / "europe_steel_capacity_by_technology_scenario_matrix.png"
    capacity.to_csv(capacity_path, index=False)
    plot_europe_technology_capacity_matrix(portfolio, portfolio_years, "iron", iron_capacity_plot)
    plot_europe_technology_capacity_matrix(portfolio, portfolio_years, "steel", steel_capacity_plot)

    flows, totals = collect_europe_trade(scenarios, include_ukraine=include_ukraine)
    flows_path = destination / "europe_trade_flows.csv"
    totals_path = destination / "europe_trade_summary.csv"
    flows.to_csv(flows_path, index=False)
    totals.to_csv(totals_path, index=False)
    net_trade_plot = destination / "europe_net_trade_scenario_matrix.png"
    plot_net_trade_matrix(totals, trade_years, net_trade_plot)

    partners = aggregate_trade_partners(flows)
    partners_path = destination / f"europe_trade_partners_{partner_year}.csv"
    partners[partners["year"] == partner_year].to_csv(partners_path, index=False)
    partner_plot = destination / f"europe_trade_partners_{partner_year}_scenario_matrix.png"
    plot_trade_partner_matrix(partners, partner_year, partner_plot)
    return [
        manifest_path,
        production_path,
        production_plot,
        self_sufficiency_path,
        self_sufficiency_plot,
        capacity_path,
        iron_capacity_plot,
        steel_capacity_plot,
        flows_path,
        totals_path,
        net_trade_plot,
        partners_path,
        partner_plot,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--master-excel", type=Path, default=DEFAULT_MASTER_EXCEL)
    parser.add_argument("--demand-scenario", default="BAU")
    parser.add_argument("--skip-runs", type=int, nargs="*", default=[])
    parser.add_argument("--production-years", type=int, nargs="+", default=[2040, 2050])
    parser.add_argument("--self-sufficiency-years", type=int, nargs="+", default=list(range(2025, 2061)))
    parser.add_argument("--trade-years", type=int, nargs="+", default=[2030, 2035, 2040, 2045, 2050])
    parser.add_argument("--partner-year", type=int, default=2050)
    parser.add_argument("--portfolio-years", type=int, nargs="+", default=[2030, 2040, 2050, 2060])
    parser.add_argument("--include-ukraine", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    created = run_europe_scenario_analysis(
        args.results_dir,
        output_dir=args.output_dir,
        master_excel=args.master_excel,
        demand_scenario=args.demand_scenario,
        skip_runs=args.skip_runs,
        production_years=args.production_years,
        self_sufficiency_years=args.self_sufficiency_years,
        trade_years=args.trade_years,
        partner_year=args.partner_year,
        portfolio_years=args.portfolio_years,
        include_ukraine=args.include_ukraine,
    )
    print(
        f"Created {len(created)} files in {(args.output_dir or args.results_dir / 'europe_scenario_post_analysis').resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
