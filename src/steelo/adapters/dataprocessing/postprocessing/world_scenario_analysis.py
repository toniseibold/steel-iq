"""Batch world-trade, emissions-intensity, and production-cost scenario analysis."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle, Wedge

from .analysis_scope import EUROPE_ANALYSIS_ISO3, EUROPE_ANALYSIS_LABEL
from .regional_trade_analysis import parse_location
from .route_classification import (
    CARBON_ROUTE_COMMODITIES,
    CARBON_ROUTE_HATCHES,
    CARBON_ROUTE_LABELS,
    source_route_lookup,
)


SCENARIO_TITLES = (
    "E-W",
    "EC-W",
    "ES-W",
    "ECS-W",
    "E-WC",
    "EC-WC",
    "ES-WC",
    "ECS-WC",
    "E-WS",
    "EC-WS",
    "ES-WS",
    "ECS-WS",
    "E-WCS",
    "EC-WCS",
    "ES-WCS",
    "ECS-WCS",
)
# Shared visual order for every 4×4 post-analysis matrix. Columns increase
# European ambition from left to right; rows increase world ambition from
# bottom to top. The bottom-left-to-top-right diagonal is therefore:
# E-W -> ES-WS -> EC-WC -> ECS-WCS.
SCENARIO_COLUMN_LABELS = ("E", "ES", "EC", "ECS")
SCENARIO_ROW_LABELS = ("WCS", "WC", "WS", "W")
SCENARIO_GRID_ORDER = (
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
SCENARIO_GRID_POSITION = {
    scenario_index: divmod(position, 4) for position, scenario_index in enumerate(SCENARIO_GRID_ORDER)
}
TRADE_GROUPS = {
    "ore": ("io_low", "io_mid", "io_high"),
    "intermediates": (
        "scrap",
        "dri_low",
        "dri_mid",
        "dri_high",
        "hbi_low",
        "hbi_mid",
        "hbi_high",
        "hot_metal",
        "pig_iron",
        "liquid_iron",
        "electrolytic_iron",
    ),
    "steel": ("steel",),
}
COMMODITY_COLOURS = {
    "electrolytic_iron": "#6a3d9a",
    "hbi_high": "#1f78b4",
    "hbi_mid": "#6baed6",
    "hbi_low": "#bdd7e7",
    "io_high": "#e31a1c",
    "io_mid": "#fb6a4a",
    "io_low": "#fcae91",
    "pig_iron": "#ff7f00",
    "scrap": "#33a02c",
    "steel": "#636363",
    "bio_pci": "#b2df8a",
    "co2_stored": "#b15928",
    # Additional traded intermediates use a separate, fixed teal/brown family.
    "dri_high": "#006d77",
    "dri_mid": "#41b6c4",
    "dri_low": "#a1dab4",
    "hot_metal": "#8c510a",
    "liquid_iron": "#bf812d",
}
TECHNOLOGY_COLOURS = {
    "BF": "#e41a1c",
    "BOF": "#984ea3",
    "DRI": "#377eb8",
    "EAF": "#4daf4a",
    "E-WIN": "#ff7f00",
    "SR": "#a65628",
    "Unknown": "#bdbdbd",
}
CONTINENT_POSITIONS = {
    "North America": (-105.0, 48.0),
    "South America": (-61.0, -18.0),
    "Europe": (13.0, 52.0),
    "Africa": (20.0, 4.0),
    "Asia": (100.0, 38.0),
    "Oceania": (137.0, -26.0),
}
DIRECT_EMISSIONS = "emissions_rs-inspired_direct_ghg"
INDIRECT_EMISSIONS = "emissions_rs-inspired_indirect_ghg"


@dataclass(frozen=True)
class ScenarioRun:
    index: int
    title: str
    run_dir: Path
    plants_csv: Path


def discover_scenario_runs(results_dir: Path, skip_runs: Iterable[int] = (11,)) -> list[ScenarioRun]:
    """Find completed runs and assign the requested 0–15 scenario titles."""
    skipped = set(skip_runs)
    runs: list[ScenarioRun] = []
    for directory in results_dir.glob("master_input_*"):
        match = re.fullmatch(r"master_input_(\d+)", directory.name)
        if not match:
            continue
        index = int(match.group(1))
        if index in skipped or not 0 <= index < len(SCENARIO_TITLES):
            continue
        completed = [
            (run_dir, sorted(run_dir.glob("post_processed_*.csv")))
            for run_dir in sorted(directory.glob("sim_*"), reverse=True)
        ]
        completed = [(run_dir, csvs) for run_dir, csvs in completed if csvs]
        if not completed:
            continue
        run_dir, csvs = completed[0]
        runs.append(ScenarioRun(index, SCENARIO_TITLES[index], run_dir, csvs[-1]))
    if not runs:
        raise FileNotFoundError(f"No completed scenario runs found below {results_dir}")
    return sorted(runs, key=lambda run: run.index)


def _continent_lookup() -> dict[str, str]:
    """Return ISO3-to-continent mapping from Cartopy's Natural Earth data."""
    path = shapereader.natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
    lookup: dict[str, str] = {}
    for record in shapereader.Reader(path).records():
        continent = str(record.attributes.get("CONTINENT", ""))
        for key in ("ISO_A3", "ADM0_A3", "SOV_A3"):
            iso3 = str(record.attributes.get(key, "")).upper()
            if len(iso3) == 3 and iso3 != "-99":
                lookup.setdefault(iso3, continent)
    lookup.update({"XKX": "Europe"})
    return lookup


def _fallback_continent(region: object) -> str:
    value = str(region).casefold()
    if "europe" in value:
        return "Europe"
    if any(token in value for token in ("africa", "mena")):
        return "Africa"
    if "america" in value:
        return "South America" if "latin" in value or "south" in value else "North America"
    if "oceania" in value:
        return "Oceania"
    return "Asia"


def prepare_trade(
    trade_csv: Path,
    group: str,
    continents: dict[str, str],
    plants: pd.DataFrame | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """Filter a trade group and attach source/destination continents."""
    trade = pd.read_csv(trade_csv)
    required = {"commodity", "source_location", "destination_location", "allocated_volume"}
    if missing := required - set(trade.columns):
        raise ValueError(f"{trade_csv} is missing columns: {', '.join(sorted(missing))}")
    trade["commodity"] = trade["commodity"].astype(str).str.casefold()
    trade = trade[trade["commodity"].isin(TRADE_GROUPS[group])].copy()
    trade["allocated_volume"] = pd.to_numeric(trade["allocated_volume"], errors="coerce").fillna(0.0)
    trade = trade[trade["allocated_volume"] > 0].copy()

    def geography(value: object) -> tuple[str, str]:
        location = parse_location(value)
        return location.iso3, continents.get(location.iso3, _fallback_continent(location.region))

    source = trade["source_location"].map(geography)
    destination = trade["destination_location"].map(geography)
    trade["source_iso3"] = source.map(lambda value: value[0])
    trade["source_continent"] = source.map(lambda value: value[1])
    trade["destination_iso3"] = destination.map(lambda value: value[0])
    trade["destination_continent"] = destination.map(lambda value: value[1])
    trade = trade[
        trade["source_continent"].isin(CONTINENT_POSITIONS) & trade["destination_continent"].isin(CONTINENT_POSITIONS)
    ].copy()
    trade["carbon_route"] = "not_applicable"
    if plants is not None and year is not None and "source_id" in trade:
        routes = source_route_lookup(plants, year).set_index("furnace_group_id")
        source_ids = trade["source_id"].astype(str)
        trade["source_route"] = source_ids.map(routes["technology_route"])
        route_sensitive = trade["commodity"].isin(CARBON_ROUTE_COMMODITIES)
        trade.loc[route_sensitive, "carbon_route"] = (
            source_ids[route_sensitive].map(routes["carbon_route"]).fillna("unknown")
        )
        if "source_tech" in trade:
            known_route = trade["source_route"].notna()
            trade.loc[known_route, "source_tech"] = trade.loc[known_route, "source_route"].astype(str)
    return trade


def aggregate_trade(trade: pd.DataFrame, group: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return intercontinental flows, signed balances, and allocated production."""
    external = trade[trade["source_continent"] != trade["destination_continent"]].copy()
    if "carbon_route" not in external:
        external["carbon_route"] = "not_applicable"
    flows = (
        external.groupby(["source_continent", "destination_continent", "carbon_route"], as_index=False)[
            "allocated_volume"
        ]
        .sum()
        .rename(columns={"allocated_volume": "volume_t"})
    )
    outgoing = external.groupby(["source_continent", "commodity", "carbon_route"], as_index=False)[
        "allocated_volume"
    ].sum()
    outgoing = outgoing.rename(columns={"source_continent": "continent", "allocated_volume": "volume_t"})
    outgoing["direction"] = "outgoing"
    incoming = external.groupby(["destination_continent", "commodity", "carbon_route"], as_index=False)[
        "allocated_volume"
    ].sum()
    incoming = incoming.rename(columns={"destination_continent": "continent", "allocated_volume": "volume_t"})
    incoming["direction"] = "incoming"
    incoming["volume_t"] *= -1.0
    balances = pd.concat([outgoing, incoming], ignore_index=True)

    production_label = trade["commodity"].astype(str)
    if group == "steel" and "source_tech" in trade:
        technology = trade["source_tech"].fillna("Unknown").astype(str)
        production_label = technology.where(technology.str.strip().ne(""), "Unknown")
    production = (
        trade.assign(production_label=production_label)
        .groupby(["source_continent", "production_label"], as_index=False)["allocated_volume"]
        .sum()
    )
    production = production.rename(columns={"source_continent": "continent", "allocated_volume": "volume_t"})
    return flows, balances, production


def _fixed_colours(names: Iterable[str], colours: dict[str, str]) -> dict[str, str]:
    """Select stable colors without reassigning them when a category is absent."""
    return {name: colours.get(name, "#bdbdbd") for name in sorted(set(names))}


def _add_continent_glyph(
    axis: plt.Axes,
    position: tuple[float, float],
    continent: str,
    balances: pd.DataFrame,
    production: pd.DataFrame,
    commodity_colours: dict[str, object],
    production_colours: dict[str, object],
    maximum_trade: float,
) -> None:
    drawing = DrawingArea(124, 100, clip=False)
    baseline, bar_x, bar_width, max_height = 44.0, 24.0, 18.0, 27.0
    continent_balance = balances[balances["continent"] == continent]
    outgoing = continent_balance[continent_balance["direction"] == "outgoing"]
    incoming = continent_balance[continent_balance["direction"] == "incoming"]
    positive_total = float(outgoing["volume_t"].sum())
    negative_total = abs(float(incoming["volume_t"].sum()))

    y = baseline
    for row in outgoing.itertuples(index=False):
        height = max_height * float(row.volume_t) / maximum_trade if maximum_trade else 0.0
        drawing.add_artist(
            Rectangle(
                (bar_x, y),
                bar_width,
                height,
                fc=commodity_colours[row.commodity],
                ec="white",
                lw=0.3,
                hatch=CARBON_ROUTE_HATCHES.get(row.carbon_route, ""),
            )
        )
        y += height
    y = baseline
    for row in incoming.itertuples(index=False):
        height = max_height * abs(float(row.volume_t)) / maximum_trade if maximum_trade else 0.0
        y -= height
        drawing.add_artist(
            Rectangle(
                (bar_x, y),
                bar_width,
                height,
                fc=commodity_colours[row.commodity],
                ec="white",
                lw=0.3,
                hatch=CARBON_ROUTE_HATCHES.get(row.carbon_route, ""),
            )
        )
    drawing.add_artist(Rectangle((bar_x - 1, baseline - 0.4), bar_width + 2, 0.8, fc="#222222"))
    drawing.add_artist(
        plt.Text(bar_x + bar_width / 2, 76, f"+{positive_total / 1e6:.1f}", ha="center", va="bottom", fontsize=6)
    )
    drawing.add_artist(
        plt.Text(bar_x + bar_width / 2, 10, f"−{negative_total / 1e6:.1f}", ha="center", va="top", fontsize=6)
    )

    continent_production = production[production["continent"] == continent]
    production_total = float(continent_production["volume_t"].sum())
    center, radius, angle = (88.0, 45.0), 17.0, 0.0
    if production_total:
        for row in continent_production.itertuples(index=False):
            next_angle = angle + 360.0 * float(row.volume_t) / production_total
            drawing.add_artist(
                Wedge(
                    center, radius, angle, next_angle, fc=production_colours[row.production_label], ec="white", lw=0.5
                )
            )
            angle = next_angle
    drawing.add_artist(Wedge(center, radius, 0, 360, fill=False, ec="#222222", lw=0.7))
    drawing.add_artist(plt.Text(62, 93, continent, ha="center", va="top", fontsize=7, weight="bold"))
    drawing.add_artist(plt.Text(88, 23, f"P {production_total / 1e6:.1f}", ha="center", va="top", fontsize=6))
    drawing.add_artist(plt.Text(8, 3, "Mt", ha="left", va="bottom", fontsize=5))
    axis.add_artist(
        AnnotationBbox(
            drawing,
            position,
            xycoords=ccrs.PlateCarree()._as_mpl_transform(axis),
            frameon=False,
            zorder=6,
        )
    )


def draw_trade_map(
    flows: pd.DataFrame,
    balances: pd.DataFrame,
    production: pd.DataFrame,
    group: str,
    scenario_title: str,
    year: int,
    output_path: Path,
) -> None:
    """Draw intercontinental arrows with signed trade bars and production pies."""
    projection = ccrs.Robinson()
    fig, axis = plt.subplots(figsize=(18, 10), dpi=180, subplot_kw={"projection": projection})
    axis.set_global()
    axis.add_feature(cfeature.LAND, facecolor="#F1F0EC")
    axis.add_feature(cfeature.OCEAN, facecolor="#EAF3F8")
    axis.add_feature(cfeature.COASTLINE, edgecolor="#999999", linewidth=0.35)
    axis.add_feature(cfeature.BORDERS, edgecolor="#BBBBBB", linewidth=0.2)
    colour_names = set(balances["commodity"].unique())
    if group != "steel":
        colour_names.update(production["production_label"].unique())
    commodity_colours = _fixed_colours(colour_names, COMMODITY_COLOURS)
    production_colours = (
        commodity_colours
        if group != "steel"
        else _fixed_colours(production["production_label"].unique(), TECHNOLOGY_COLOURS)
    )

    maximum_flow = float(flows["volume_t"].max()) if not flows.empty else 1.0
    for row in flows.itertuples(index=False):
        start = CONTINENT_POSITIONS[row.source_continent]
        end = CONTINENT_POSITIONS[row.destination_continent]
        width = 0.5 + 5.0 * math.sqrt(float(row.volume_t) / maximum_flow)
        arrow = FancyArrowPatch(
            start,
            end,
            transform=ccrs.PlateCarree()._as_mpl_transform(axis),
            arrowstyle="-|>",
            mutation_scale=7 + width * 1.5,
            linewidth=width,
            color="#485D70",
            linestyle={"green": "solid", "blue": "dashed", "grey": "dotted"}.get(row.carbon_route, "solid"),
            alpha=0.38,
            connectionstyle="arc3,rad=0.13",
            zorder=2,
        )
        axis.add_patch(arrow)

    maximum_trade = max(
        (float(grouped["volume_t"].abs().sum()) for _, grouped in balances.groupby(["continent", "direction"])),
        default=1.0,
    )
    for continent, position in CONTINENT_POSITIONS.items():
        _add_continent_glyph(
            axis, position, continent, balances, production, commodity_colours, production_colours, maximum_trade
        )

    handles = [Patch(facecolor=colour, label=name.replace("_", " ")) for name, colour in commodity_colours.items()]
    iron_routes = sorted(set(balances["carbon_route"]) - {"not_applicable"})
    handles += [
        Patch(
            facecolor="white",
            edgecolor="#555555",
            hatch=CARBON_ROUTE_HATCHES.get(route, ""),
            label=f"Iron route — {CARBON_ROUTE_LABELS.get(route, route)}",
        )
        for route in iron_routes
    ]
    if group == "steel":
        handles += [
            Patch(facecolor=colour, edgecolor="white", label=f"Production: {name}")
            for name, colour in production_colours.items()
        ]
    axis.legend(handles=handles, loc="lower left", ncol=3, fontsize=7, title="Stacked trade / production pie")
    axis.set_title(f"{scenario_title}: intercontinental {group} trade — {year}", fontsize=17, pad=14)
    axis.text(
        0.995,
        0.015,
        "Arrow direction and width: traded volume · bars: outgoing (+) / incoming (−), Mt · pie: allocated production",
        transform=axis.transAxes,
        ha="right",
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def calculate_production_metrics(run: ScenarioRun, years: Iterable[int]) -> pd.DataFrame:
    """Calculate regional/global production-weighted intensity and cost."""
    columns = [
        "year",
        "iso3",
        "furnace_group_id",
        "product",
        "production",
        "unit_production_cost",
        DIRECT_EMISSIONS,
        INDIRECT_EMISSIONS,
    ]
    plants = pd.read_csv(run.plants_csv, usecols=lambda column: column in columns, low_memory=False)
    missing = set(columns) - set(plants.columns)
    if missing:
        raise ValueError(f"{run.plants_csv} is missing columns: {', '.join(sorted(missing))}")
    plants = plants.drop_duplicates(["year", "furnace_group_id"]).copy()
    plants["year"] = pd.to_numeric(plants["year"], errors="coerce")
    plants["production"] = pd.to_numeric(plants["production"], errors="coerce").fillna(0.0)
    plants["unit_production_cost"] = pd.to_numeric(plants["unit_production_cost"], errors="coerce")
    plants[DIRECT_EMISSIONS] = pd.to_numeric(plants[DIRECT_EMISSIONS], errors="coerce").fillna(0.0)
    plants[INDIRECT_EMISSIONS] = pd.to_numeric(plants[INDIRECT_EMISSIONS], errors="coerce").fillna(0.0)
    plants["product"] = plants["product"].astype(str).str.casefold()
    plants["iso3"] = plants["iso3"].astype(str).str.upper()
    plants = plants[
        plants["year"].isin(set(years)) & plants["product"].isin(("iron", "steel")) & plants["production"].gt(0)
    ]

    records: list[dict[str, object]] = []
    for scope, scope_rows in (
        (EUROPE_ANALYSIS_LABEL, plants[plants["iso3"].isin(EUROPE_ANALYSIS_ISO3)]),
        ("Global", plants),
    ):
        for (year, product), group in scope_rows.groupby(["year", "product"]):
            production = float(group["production"].sum())
            emissions = float((group[DIRECT_EMISSIONS] + group[INDIRECT_EMISSIONS]).sum())
            valid_cost = group["unit_production_cost"].notna()
            cost = (
                float(
                    np.average(
                        group.loc[valid_cost, "unit_production_cost"], weights=group.loc[valid_cost, "production"]
                    )
                )
                if valid_cost.any()
                else math.nan
            )
            records.append(
                {
                    "scenario_index": run.index,
                    "scenario_title": run.title,
                    "scope": scope,
                    "year": int(year),
                    "product": product,
                    "production_mt": production / 1e6,
                    "carbon_intensity_tco2e_per_t": emissions / production,
                    "average_production_cost_usd_per_t": cost,
                }
            )
    return pd.DataFrame.from_records(records)


def draw_scenario_matrix(
    metrics: pd.DataFrame,
    scope: str,
    value_column: str,
    years: Iterable[int],
    output_path: Path,
) -> None:
    years = tuple(years)
    products = ("iron", "steel")
    selected = metrics[metrics["scope"] == scope]
    fig, axes = plt.subplots(
        len(years), len(products), figsize=(15, 10), dpi=180, squeeze=False, constrained_layout=True
    )
    for row_index, year in enumerate(years):
        for column_index, product in enumerate(products):
            axis = axes[row_index, column_index]
            matrix = np.full((4, 4), np.nan)
            subset = selected[(selected["year"] == year) & (selected["product"] == product)]
            for item in subset.itertuples(index=False):
                matrix_row, matrix_column = SCENARIO_GRID_POSITION[item.scenario_index]
                matrix[matrix_row, matrix_column] = float(getattr(item, value_column))
            finite = matrix[np.isfinite(matrix)]
            vmin = float(finite.min()) if finite.size else 0.0
            vmax = float(finite.max()) if finite.size else 1.0
            cmap = plt.get_cmap("RdYlGn_r").copy()
            image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=vmin, vmax=vmax)
            image.cmap.set_bad("#D9D9D9")
            axis.set_xticks(range(4), SCENARIO_COLUMN_LABELS)
            axis.set_yticks(range(4), SCENARIO_ROW_LABELS)
            axis.set_xlabel("European policy")
            axis.set_ylabel("World policy")
            axis.set_title(f"{product.title()} — {year}")
            for matrix_row in range(4):
                for matrix_column in range(4):
                    index = SCENARIO_GRID_ORDER[matrix_row * 4 + matrix_column]
                    value = matrix[matrix_row, matrix_column]
                    label = SCENARIO_TITLES[index]
                    text = f"{value:.2f}\n{label}" if not math.isnan(value) else f"N/A\n{label}"
                    axis.text(matrix_column, matrix_row, text, ha="center", va="center", fontsize=7)
            colorbar = fig.colorbar(image, ax=axis, fraction=0.047, pad=0.03)
            colorbar.ax.tick_params(labelsize=7)
    label = (
        "Average carbon intensity (tCO₂e/t; direct + indirect)"
        if value_column == "carbon_intensity_tco2e_per_t"
        else "Average production cost (USD/t)"
    )
    title = EUROPE_ANALYSIS_LABEL if scope == EUROPE_ANALYSIS_LABEL else "Global"
    fig.suptitle(f"{title} iron and steel {label.lower()}", fontsize=16)
    fig.supxlabel(label, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    results_dir: Path,
    *,
    output_dir: Path | None = None,
    trade_year: int = 2050,
    matrix_years: Iterable[int] = (2040, 2050),
    skip_runs: Iterable[int] = (11,),
) -> list[Path]:
    """Generate all requested CSV tables, trade maps, and scenario matrices."""
    results_dir = results_dir.resolve()
    destination = (output_dir or results_dir / "world_scenario_analysis").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    runs = discover_scenario_runs(results_dir, skip_runs)
    continents = _continent_lookup()
    created: list[Path] = []
    metric_frames: list[pd.DataFrame] = []
    label_rows: list[dict[str, object]] = []

    for run in runs:
        label_rows.append({"scenario_index": run.index, "scenario_title": run.title, "run_dir": str(run.run_dir)})
        metric_frames.append(calculate_production_metrics(run, matrix_years))
        trade_csv = run.run_dir / "TM" / f"steel_trade_allocations_{trade_year}.csv"
        if not trade_csv.is_file():
            continue
        scenario_dir = destination / "trade_maps" / run.title
        plant_columns = ["year", "furnace_group_id", "technology", "chosen_reductant"]
        plants = pd.read_csv(run.plants_csv, usecols=lambda column: column in plant_columns, low_memory=False)
        for group in TRADE_GROUPS:
            trade = prepare_trade(trade_csv, group, continents, plants, trade_year)
            flows, balances, production = aggregate_trade(trade, group)
            flow_csv = scenario_dir / f"{group}_{trade_year}_continent_flows.csv"
            balance_csv = scenario_dir / f"{group}_{trade_year}_continent_balances.csv"
            production_csv = scenario_dir / f"{group}_{trade_year}_continent_production.csv"
            map_path = scenario_dir / f"{group}_{trade_year}_world_map.png"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            flows.to_csv(flow_csv, index=False)
            balances.to_csv(balance_csv, index=False)
            production.to_csv(production_csv, index=False)
            draw_trade_map(flows, balances, production, group, run.title, trade_year, map_path)
            created.extend([map_path, flow_csv, balance_csv, production_csv])

    labels_path = destination / "scenario_labels.csv"
    pd.DataFrame(label_rows).sort_values("scenario_index").to_csv(labels_path, index=False)
    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(
        ["scenario_index", "scope", "year", "product"], ignore_index=True
    )
    metrics_path = destination / "production_carbon_intensity_and_cost.csv"
    metrics.to_csv(metrics_path, index=False)
    created.extend([labels_path, metrics_path])
    # Keep the historical ``eu27`` filename slug so existing reports and links
    # continue to resolve; the figure title states the expanded scope explicitly.
    for scope, slug in ((EUROPE_ANALYSIS_LABEL, "eu27"), ("Global", "global")):
        for value_column, metric_slug in (
            ("carbon_intensity_tco2e_per_t", "carbon_intensity"),
            ("average_production_cost_usd_per_t", "production_cost"),
        ):
            path = destination / f"{metric_slug}_{slug}_scenario_matrices.png"
            draw_scenario_matrix(metrics, scope, value_column, matrix_years, path)
            created.append(path)
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Directory containing master_input_0 ... master_input_15")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trade-year", type=int, default=2050)
    parser.add_argument("--matrix-years", type=int, nargs="+", default=[2040, 2050])
    parser.add_argument("--skip-runs", type=int, nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    created = run_analysis(
        args.results_dir,
        output_dir=args.output_dir,
        trade_year=args.trade_year,
        matrix_years=args.matrix_years,
        skip_runs=args.skip_runs,
    )
    print(
        f"Created {len(created)} files in {(args.output_dir or args.results_dir / 'world_scenario_analysis').resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
