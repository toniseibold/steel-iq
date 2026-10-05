"""Standalone Germany and Europe trade-flow analysis.

The routine reads the CSV artefacts of a completed simulation.  It deliberately
does not import or unpickle simulation state, which makes it suitable for old
runs and for analysis on a different machine.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import FancyArrowPatch, Patch, Wedge

from .route_classification import TECHNOLOGY_ROUTE_COLOURS, source_route_lookup


EUROPE = "Europe"
GERMANY = "DEU"
FINAL_DEMAND_TECH = "Final demand"
UNKNOWN_TECH = "Unknown"
METALLIC_STEEL_INPUTS = {
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

# Non-European countries are represented by one of these output regions.  The
# aliases cover both current ``region_for_outputs`` values and older runs.
REGION_ALIASES = {
    "middle east": "MENA",
    "north africa": "MENA",
    "middle east and north africa": "MENA",
    "sub-saharan africa": "Subsaharan Africa",
    "subsaharan africa": "Subsaharan Africa",
    "east asia": "Other Asia",
    "southeast asia": "Other Asia",
    "south asia": "Other Asia",
    "developed asia": "Developed Asia",
    "rest of world": "Rest of World",
}

GERMANY_MAP_REGION_POSITIONS = {
    "North America": (-31.0, 54.0),
    "Latin America": (-31.0, 30.0),
    "MENA": (31.0, 22.0),
    "Subsaharan Africa": (17.0, 17.5),
    "CIS": (54.0, 61.0),
    "China": (66.0, 47.0),
    "India": (61.0, 31.0),
    "Other Asia": (69.0, 23.0),
    "Developed Asia": (69.0, 38.0),
    "Oceania": (56.0, 18.0),
    "Rest of World": (-5.0, 17.0),
}

WORLD_REGION_POSITIONS = {
    "Europe": (10.0, 52.0),
    "North America": (-105.0, 47.0),
    "Latin America": (-62.0, -18.0),
    "MENA": (35.0, 26.0),
    "Subsaharan Africa": (22.0, -9.0),
    "CIS": (65.0, 56.0),
    "China": (104.0, 35.0),
    "India": (79.0, 21.0),
    "Other Asia": (105.0, 5.0),
    "Developed Asia": (137.0, 38.0),
    "Oceania": (135.0, -27.0),
    "Rest of World": (0.0, -42.0),
}

_LOCATION_FIELD = re.compile(
    r"(?P<key>lat|lon|country|region|iso3)=(?P<value>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[-+\d.eE]+)"
)


@dataclass(frozen=True)
class LocationRecord:
    iso3: str
    region: str
    lat: float
    lon: float
    country: str = ""


@dataclass(frozen=True)
class AnalysisInputs:
    run_dir: Path
    year: int
    trade_csv: Path
    plant_csv: Path | None
    shapefile: Path


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_location(value: object) -> LocationRecord:
    """Parse a model ``Location(...)`` string or a small JSON/dict value."""
    if isinstance(value, dict):
        fields = value
    else:
        text = _clean_text(value)
        if not text or text == "N/A":
            raise ValueError("missing location")
        fields: dict[str, object] = {}
        if text.startswith("{"):
            try:
                candidate = ast.literal_eval(text)
                if isinstance(candidate, dict):
                    fields = candidate
            except (SyntaxError, ValueError):
                pass
        if not fields:
            for match in _LOCATION_FIELD.finditer(text):
                raw = match.group("value")
                try:
                    fields[match.group("key")] = ast.literal_eval(raw)
                except (SyntaxError, ValueError):
                    fields[match.group("key")] = raw

    try:
        return LocationRecord(
            iso3=str(fields["iso3"]).upper(),
            region=str(fields.get("region", "")),
            lat=float(fields["lat"]),
            lon=float(fields["lon"]),
            country=str(fields.get("country", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"could not parse location: {value!r}") from exc


def _normalise_region(region: str) -> str:
    cleaned = region.strip()
    return REGION_ALIASES.get(cleaned.casefold(), cleaned or "Rest of World")


def _explicit_location(row: pd.Series, prefix: str) -> LocationRecord | None:
    required = [f"{prefix}_iso3", f"{prefix}_lat", f"{prefix}_lon"]
    if not all(column in row.index and not pd.isna(row[column]) for column in required):
        return None
    return LocationRecord(
        iso3=_clean_text(row[f"{prefix}_iso3"]).upper(),
        region=_clean_text(row.get(f"{prefix}_region", "")),
        lat=float(row[f"{prefix}_lat"]),
        lon=float(row[f"{prefix}_lon"]),
        country=_clean_text(row.get(f"{prefix}_country", "")),
    )


def _load_country_geography(run_dir: Path, plants: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """Load authoritative ISO3/region mappings, with plant output as fallback."""
    iso_to_region: dict[str, str] = {}
    country_to_iso: dict[str, str] = {}
    config_path = run_dir / "simulation_config.json"
    mapping_candidates: list[Path] = []
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}
        if data_dir := config.get("data_dir"):
            mapping_candidates.append(Path(data_dir) / "fixtures" / "country_mappings.json")

    for mapping_path in mapping_candidates:
        if not mapping_path.is_file():
            continue
        try:
            mappings = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for item in mappings:
            iso3 = _clean_text(item.get("ISO 3-letter code")).upper()
            region = _normalise_region(_clean_text(item.get("region_for_outputs")))
            country = _clean_text(item.get("Country"))
            if iso3:
                iso_to_region[iso3] = region
                country_to_iso[iso3.casefold()] = iso3
            if country and iso3:
                country_to_iso[country.casefold()] = iso3
        break

    if {"iso3", "region"}.issubset(plants.columns):
        columns = ["iso3", "region"] + (["country"] if "country" in plants else [])
        for row in plants[columns].drop_duplicates().itertuples(index=False):
            iso3 = _clean_text(row.iso3).upper()
            if not iso3:
                continue
            iso_to_region.setdefault(iso3, _normalise_region(_clean_text(row.region)))
            country_to_iso.setdefault(iso3.casefold(), iso3)
            country = _clean_text(getattr(row, "country", ""))
            if country:
                country_to_iso.setdefault(country.casefold(), iso3)
    return iso_to_region, country_to_iso


def _resolve_location_geography(
    location: LocationRecord,
    iso_to_region: dict[str, str],
    country_to_iso: dict[str, str],
) -> LocationRecord:
    iso3 = location.iso3.upper()
    if not iso3:
        iso3 = country_to_iso.get(location.country.casefold(), "")
    # Some supplier locations put an ISO3 code in ``country`` but leave
    # ``iso3`` blank (e.g. country='USA').
    if not iso3 and len(location.country.strip()) == 3:
        candidate = location.country.strip().upper()
        if candidate in iso_to_region:
            iso3 = candidate
    region = iso_to_region.get(iso3, _normalise_region(location.region))
    return LocationRecord(iso3=iso3, region=region, lat=location.lat, lon=location.lon, country=location.country)


def _locations_from_trade(
    trade: pd.DataFrame,
    iso_to_region: dict[str, str] | None = None,
    country_to_iso: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, LocationRecord]]:
    iso_to_region = iso_to_region or {}
    country_to_iso = country_to_iso or {}
    records: list[dict[str, object]] = []
    locations: dict[str, LocationRecord] = {}
    failures: list[int] = []
    for index, row in trade.iterrows():
        try:
            source = _explicit_location(row, "source") or parse_location(row["source_location"])
            destination = _explicit_location(row, "destination") or parse_location(row["destination_location"])
            source = _resolve_location_geography(source, iso_to_region, country_to_iso)
            destination = _resolve_location_geography(destination, iso_to_region, country_to_iso)
        except (KeyError, ValueError):
            failures.append(int(index))
            continue
        locations[source.iso3] = source
        locations[destination.iso3] = destination
        record = row.to_dict()
        record.update(
            source_iso3=source.iso3,
            source_region=_normalise_region(source.region),
            source_lat=source.lat,
            source_lon=source.lon,
            destination_iso3=destination.iso3,
            destination_region=_normalise_region(destination.region),
            destination_lat=destination.lat,
            destination_lon=destination.lon,
        )
        records.append(record)
    if failures:
        sample = ", ".join(map(str, failures[:5]))
        raise ValueError(f"Could not parse locations in {len(failures)} trade rows (first row indices: {sample})")
    return pd.DataFrame.from_records(records), locations


def _latest_matching(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def _year_from_trade_path(path: Path) -> int:
    match = re.search(r"steel_trade_allocations_(\d{4})\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot infer simulation year from {path.name}")
    return int(match.group(1))


def _resolve_shapefile(run_dir: Path, explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit)
    config_path = run_dir / "simulation_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}
        configured = config.get("countries_shapefile_dir")
        if configured:
            candidates.append(Path(configured))
        data_dir = config.get("data_dir")
        if data_dir:
            candidates.append(Path(data_dir) / "ne_110m_admin_0_countries")

    for candidate in candidates:
        if candidate.is_dir():
            candidate = candidate / "ne_110m_admin_0_countries.shp"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Natural Earth country shapefile not found. Pass --shapefile or keep "
        "countries_shapefile_dir/data_dir in simulation_config.json."
    )


def discover_inputs(
    run_dir: Path,
    year: int | None = None,
    trade_csv: Path | None = None,
    plant_csv: Path | None = None,
    shapefile: Path | None = None,
) -> AnalysisInputs:
    """Resolve the standard outputs of a completed run."""
    run_dir = run_dir.resolve()
    if trade_csv is None:
        if year is not None:
            trade_csv = run_dir / "TM" / f"steel_trade_allocations_{year}.csv"
        else:
            candidates = list((run_dir / "TM").glob("steel_trade_allocations_*.csv"))
            if not candidates:
                raise FileNotFoundError(f"No trade allocation CSVs found below {run_dir / 'TM'}")
            trade_csv = max(candidates, key=_year_from_trade_path)
    trade_csv = trade_csv.resolve()
    if not trade_csv.is_file():
        raise FileNotFoundError(trade_csv)
    resolved_year = year if year is not None else _year_from_trade_path(trade_csv)

    if plant_csv is None:
        plant_csv = _latest_matching(run_dir, "post_processed_*.csv")
    elif not plant_csv.is_file():
        raise FileNotFoundError(plant_csv)

    return AnalysisInputs(
        run_dir=run_dir,
        year=resolved_year,
        trade_csv=trade_csv,
        plant_csv=plant_csv.resolve() if plant_csv else None,
        shapefile=_resolve_shapefile(run_dir, shapefile),
    )


def _is_europe(region: object) -> bool:
    return _normalise_region(_clean_text(region)).casefold() == EUROPE.casefold()


def _technology_lookup(plants: pd.DataFrame, year: int) -> dict[str, str]:
    if plants.empty or "furnace_group_id" not in plants or "technology" not in plants:
        return {}
    routes = source_route_lookup(plants, year)
    return dict(zip(routes["furnace_group_id"], routes["technology_route"]))


def _prepare_trade(
    trade: pd.DataFrame,
    plants: pd.DataFrame,
    year: int,
    commodity: str,
    iso_to_region: dict[str, str] | None = None,
    country_to_iso: dict[str, str] | None = None,
) -> pd.DataFrame:
    required = {"commodity", "source_id", "destination_id", "destination_type", "allocated_volume"}
    missing = required - set(trade.columns)
    if missing:
        raise ValueError(f"Trade CSV is missing required columns: {', '.join(sorted(missing))}")
    trade, _ = _locations_from_trade(trade, iso_to_region, country_to_iso)
    trade["commodity"] = trade["commodity"].astype(str).str.casefold()
    trade = trade[trade["commodity"] == commodity.casefold()].copy()
    trade["allocated_volume"] = pd.to_numeric(trade["allocated_volume"], errors="coerce").fillna(0.0)
    trade = trade[trade["allocated_volume"] > 0]

    lookup = _technology_lookup(plants, year)
    if "source_tech" not in trade:
        trade["source_tech"] = ""
    source_fallback = trade["source_id"].map(lambda value: lookup.get(_clean_text(value), UNKNOWN_TECH))
    invalid_source = trade["source_tech"].map(_clean_text).isin({"", "N/A", "Unknown"})
    trade.loc[invalid_source, "source_tech"] = source_fallback[invalid_source]
    if "destination_tech" not in trade:
        trade["destination_tech"] = trade["destination_id"].map(
            lambda value: lookup.get(_clean_text(value), FINAL_DEMAND_TECH)
        )
    return trade


def _country_label(iso3: str, world: gpd.GeoDataFrame) -> str:
    iso_columns = [column for column in ("ISO_A3", "ADM0_A3", "SOV_A3", "iso_a3") if column in world]
    name_columns = [column for column in ("NAME_EN", "NAME", "ADMIN", "name") if column in world]
    if iso_columns and name_columns:
        match = world[world[iso_columns[0]].astype(str).str.upper() == iso3]
        if not match.empty:
            return _clean_text(match.iloc[0][name_columns[0]]) or iso3
    return iso3


def _node_for(row: pd.Series, prefix: str, scope: str) -> str:
    iso3 = row[f"{prefix}_iso3"]
    region = row[f"{prefix}_region"]
    if scope == "germany":
        return iso3 if _is_europe(region) else _normalise_region(region)
    return EUROPE if _is_europe(region) else _normalise_region(region)


def aggregate_flows(trade: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Filter to flows touching Germany/Europe and aggregate partner nodes."""
    if scope == "germany":
        relevant = (trade["source_iso3"] == GERMANY) | (trade["destination_iso3"] == GERMANY)
    elif scope == "europe":
        source_europe = trade["source_region"].map(_is_europe)
        destination_europe = trade["destination_region"].map(_is_europe)
        relevant = source_europe ^ destination_europe
    else:
        raise ValueError("scope must be 'germany' or 'europe'")
    selected = trade[relevant].copy()
    if selected.empty:
        return pd.DataFrame(columns=["source", "destination", "volume"])
    selected["source"] = selected.apply(_node_for, axis=1, args=("source", scope))
    selected["destination"] = selected.apply(_node_for, axis=1, args=("destination", scope))
    return (
        selected.groupby(["source", "destination"], as_index=False)["allocated_volume"]
        .sum()
        .rename(columns={"allocated_volume": "volume"})
        .sort_values("volume", ascending=False)
        .reset_index(drop=True)
    )


def _production_by_country(plants: pd.DataFrame, year: int, commodity: str) -> pd.DataFrame:
    required = {"iso3", "technology", "product", "production"}
    if plants.empty or not required.issubset(plants.columns):
        return pd.DataFrame(columns=["iso3", "region", "technology", "volume"])
    selected = plants.copy()
    if "year" in selected:
        selected = selected[pd.to_numeric(selected["year"], errors="coerce") == year]
    selected = selected[selected["product"].astype(str).str.casefold() == commodity.casefold()]
    # The post-processed table can contain one row per feedstock. Production is a
    # furnace-level value, so count it only once.
    if "furnace_group_id" in selected:
        selected = selected.drop_duplicates("furnace_group_id")
    selected["volume"] = pd.to_numeric(selected["production"], errors="coerce").fillna(0.0)
    if "region" not in selected:
        selected["region"] = ""
    if "chosen_reductant" not in selected:
        selected["chosen_reductant"] = ""
    routes = source_route_lookup(selected, year).set_index("furnace_group_id")["technology_route"]
    selected["technology"] = selected["furnace_group_id"].astype(str).map(routes).fillna(UNKNOWN_TECH)
    return selected[["iso3", "region", "technology", "volume"]]


def _node_metrics(
    trade: pd.DataFrame,
    plants: pd.DataFrame,
    year: int,
    commodity: str,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    production = _production_by_country(plants, year, commodity)
    if production.empty:
        production = trade[["source_iso3", "source_region", "source_tech", "allocated_volume"]].rename(
            columns={
                "source_iso3": "iso3",
                "source_region": "region",
                "source_tech": "technology",
                "allocated_volume": "volume",
            }
        )
    production["node"] = production.apply(
        lambda row: row["iso3"]
        if scope == "germany" and _is_europe(row["region"])
        else (EUROPE if scope == "europe" and _is_europe(row["region"]) else _normalise_region(row["region"])),
        axis=1,
    )
    prod = production.groupby(["node", "technology"], as_index=False)["volume"].sum()

    demand = trade.copy()
    demand["node"] = demand.apply(_node_for, axis=1, args=("destination", scope))
    dem = (
        demand.groupby("node", as_index=False)["allocated_volume"].sum().rename(columns={"allocated_volume": "demand"})
    )
    return prod, dem


def _representative_positions(
    trade: pd.DataFrame,
    world: gpd.GeoDataFrame,
    flows: pd.DataFrame,
    scope: str,
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    nodes = set(flows["source"]) | set(flows["destination"])
    positions: dict[str, tuple[float, float]] = {}
    labels: dict[str, str] = {}
    fixed = GERMANY_MAP_REGION_POSITIONS if scope == "germany" else WORLD_REGION_POSITIONS
    for node in nodes:
        if node in fixed:
            positions[node] = fixed[node]
            labels[node] = node
            continue
        rows = pd.concat(
            [
                trade.loc[trade["source_iso3"] == node, ["source_lon", "source_lat"]].rename(
                    columns={"source_lon": "lon", "source_lat": "lat"}
                ),
                trade.loc[trade["destination_iso3"] == node, ["destination_lon", "destination_lat"]].rename(
                    columns={"destination_lon": "lon", "destination_lat": "lat"}
                ),
            ],
            ignore_index=True,
        )
        if not rows.empty:
            positions[node] = (float(rows["lon"].mean()), float(rows["lat"].mean()))
        labels[node] = _country_label(node, world)
    return positions, labels


def _technology_colors(technologies: Iterable[str]) -> dict[str, tuple[float, float, float, float]]:
    names = sorted({_clean_text(name) or UNKNOWN_TECH for name in technologies})
    cmap = plt.get_cmap("tab20")
    return {name: TECHNOLOGY_ROUTE_COLOURS.get(name, cmap(index % 20)) for index, name in enumerate(names)}


def _add_half_pie(
    ax: plt.Axes,
    xy: tuple[float, float],
    production: dict[str, float],
    demand: float,
    colors: dict[str, object],
    max_total: float,
) -> None:
    total = max(sum(production.values()), demand)
    radius = 10.0 + 20.0 * math.sqrt(total / max_total) if max_total > 0 else 10.0
    drawing = DrawingArea(2 * radius, 2 * radius, clip=False)
    production_total = sum(production.values())
    angle = 0.0
    if production_total > 0:
        for technology, volume in sorted(production.items()):
            next_angle = angle + 180.0 * volume / production_total
            drawing.add_artist(Wedge((radius, radius), radius, angle, next_angle, fc=colors[technology], ec="white"))
            angle = next_angle
    else:
        drawing.add_artist(Wedge((radius, radius), radius, 0, 180, fc="#f7f7f7", ec="white"))
    demand_color = "#777777" if demand > 0 else "#f7f7f7"
    drawing.add_artist(Wedge((radius, radius), radius, 180, 360, fc=demand_color, ec="white"))
    drawing.add_artist(Wedge((radius, radius), radius, 0, 360, fill=False, ec="#222222", lw=0.8))
    ax.add_artist(AnnotationBbox(drawing, xy, frameon=False, box_alignment=(0.5, 0.5), zorder=6))


def _format_volume(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.1f} Mt"
    if value >= 1e3:
        return f"{value / 1e3:.0f} kt"
    return f"{value:.0f} t"


def _draw_map(
    world: gpd.GeoDataFrame,
    trade: pd.DataFrame,
    flows: pd.DataFrame,
    production: pd.DataFrame,
    demand: pd.DataFrame,
    scope: str,
    commodity: str,
    year: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    world.plot(ax=ax, color="#edf0f2", edgecolor="#aab2b8", linewidth=0.35, zorder=0)
    if scope == "germany":
        ax.set_xlim(-38, 73)
        ax.set_ylim(14, 75)
        focus = "Germany"
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-58, 82)
        focus = "Europe"
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    positions, labels = _representative_positions(trade, world, flows, scope)
    max_flow = float(flows["volume"].max()) if not flows.empty else 1.0
    for row in flows.itertuples(index=False):
        start = positions.get(row.source)
        end = positions.get(row.destination)
        if not start or not end or start == end:
            continue
        is_export = row.source in ({GERMANY} if scope == "germany" else {EUROPE})
        color = "#c44e52" if is_export else "#4c72b0"
        width = 0.7 + 4.5 * math.sqrt(float(row.volume) / max_flow)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8 + width * 1.5,
            linewidth=width,
            color=color,
            alpha=0.62,
            connectionstyle="arc3,rad=0.12",
            zorder=3,
        )
        ax.add_patch(arrow)
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.annotate(
            _format_volume(float(row.volume)),
            midpoint,
            fontsize=7,
            color="#333333",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.72},
            zorder=4,
        )

    technologies = production["technology"].unique() if not production.empty else []
    colors = _technology_colors(technologies)
    prod_lookup: dict[str, dict[str, float]] = defaultdict(dict)
    for row in production.itertuples(index=False):
        if row.volume > 0:
            prod_lookup[row.node][row.technology] = float(row.volume)
    demand_lookup = dict(zip(demand["node"], demand["demand"]))
    totals = [sum(prod_lookup[node].values()) for node in positions] + [
        float(demand_lookup.get(node, 0)) for node in positions
    ]
    max_total = max(totals, default=1.0)
    for node, position in positions.items():
        _add_half_pie(ax, position, prod_lookup[node], float(demand_lookup.get(node, 0)), colors, max_total)
        ax.annotate(
            labels.get(node, node),
            position,
            xytext=(0, -28),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            weight="bold" if node in {GERMANY, EUROPE} else "normal",
            zorder=7,
        )

    legend_handles = [Patch(facecolor=color, label=technology) for technology, color in colors.items()]
    legend_handles += [
        Patch(facecolor="#777777", label="Demand (lower half)"),
        Patch(facecolor="none", edgecolor="none", label="Node size scales with volume"),
        Patch(facecolor="#c44e52", label=f"Exports from {focus}"),
        Patch(facecolor="#4c72b0", label=f"Imports to {focus}"),
    ]
    ax.legend(
        handles=legend_handles,
        title="Production technology / map encoding",
        loc="lower left",
        frameon=True,
        framealpha=0.94,
        fontsize=8,
        title_fontsize=9,
        ncol=2,
    )
    ax.set_title(f"{commodity.replace('_', ' ').title()} trade flows to and from {focus} — {year}", fontsize=17, pad=12)
    ax.text(
        0.995,
        0.01,
        "Upper pie: production by technology · Lower pie: allocated demand · Line width: trade volume",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _node_table(production: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    if production.empty:
        prod_wide = pd.DataFrame(columns=["node", "production_total"])
    else:
        prod_wide = production.pivot_table(
            index="node", columns="technology", values="volume", aggfunc="sum", fill_value=0
        )
        prod_wide.columns = [f"production_{column}" for column in prod_wide.columns]
        prod_wide["production_total"] = prod_wide.sum(axis=1)
        prod_wide = prod_wide.reset_index()
    return prod_wide.merge(demand, on="node", how="outer").fillna(0)


def _domestic_steel_origin(plants: pd.DataFrame) -> pd.DataFrame:
    """Split furnace-level steel output into scrap and primary material origin."""
    required = {"year", "iso3", "region", "furnace_group_id", "product", "production"}
    missing = required - set(plants.columns)
    if missing:
        raise ValueError(f"Post-processed CSV is missing required columns: {', '.join(sorted(missing))}")

    steel = plants[plants["product"].astype(str).str.casefold() == "steel"].copy()
    steel["production"] = pd.to_numeric(steel["production"], errors="coerce").fillna(0.0)
    if "feedstock" not in steel or "demand" not in steel:
        steel["feedstock"] = ""
        steel["demand"] = 0.0
    steel["feedstock"] = steel["feedstock"].fillna("").astype(str).str.casefold()
    steel["demand"] = pd.to_numeric(steel["demand"], errors="coerce").fillna(0.0)

    records: list[dict[str, object]] = []
    group_columns = ["year", "furnace_group_id"]
    for (year, furnace_group_id), group in steel.groupby(group_columns, sort=True):
        metadata = group.iloc[0]
        metallic = group[group["feedstock"].isin(METALLIC_STEEL_INPUTS)]
        metallic_demand = float(metallic["demand"].clip(lower=0).sum())
        scrap_demand = float(metallic.loc[metallic["feedstock"] == "scrap", "demand"].clip(lower=0).sum())
        secondary_share = min(1.0, scrap_demand / metallic_demand) if metallic_demand > 0 else 0.0
        production = float(group["production"].max())
        records.append(
            {
                "year": int(year),
                "furnace_group_id": furnace_group_id,
                "iso3": _clean_text(metadata["iso3"]).upper(),
                "region": _normalise_region(_clean_text(metadata["region"])),
                "primary_production": production * (1.0 - secondary_share),
                "secondary_production": production * secondary_share,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "year",
            "furnace_group_id",
            "iso3",
            "region",
            "primary_production",
            "secondary_production",
        ],
    )


def _steel_trade_by_year(
    run_dir: Path,
    plants: pd.DataFrame,
    iso_to_region: dict[str, str],
    country_to_iso: dict[str, str],
) -> dict[int, pd.DataFrame]:
    yearly_trade: dict[int, pd.DataFrame] = {}
    for path in sorted((run_dir / "TM").glob("steel_trade_allocations_*.csv")):
        year = _year_from_trade_path(path)
        raw = pd.read_csv(path)
        prepared = _prepare_trade(
            raw,
            plants,
            year,
            "steel",
            iso_to_region=iso_to_region,
            country_to_iso=country_to_iso,
        )
        yearly_trade[year] = prepared
    return yearly_trade


def calculate_steel_production_origin_transition(
    plants: pd.DataFrame,
    yearly_trade: dict[int, pd.DataFrame],
    scope: str,
) -> pd.DataFrame:
    """Return annual primary, secondary and signed net-trade steel supply.

    Only furnace groups whose final product is ``steel`` and allocations whose
    commodity is exactly ``steel`` are included. Intermediate iron products are
    therefore excluded by construction.
    """
    domestic = _domestic_steel_origin(plants)
    years = sorted(set(domestic["year"]) | set(yearly_trade))
    records: list[dict[str, object]] = []
    for year in years:
        production = domestic[domestic["year"] == year]
        trade = yearly_trade.get(year, pd.DataFrame())
        if scope == "germany":
            production = production[production["iso3"] == GERMANY]
            if trade.empty:
                imports = exports = 0.0
            else:
                imports = float(
                    trade.loc[
                        (trade["destination_iso3"] == GERMANY) & (trade["source_iso3"] != GERMANY),
                        "allocated_volume",
                    ].sum()
                )
                exports = float(
                    trade.loc[
                        (trade["source_iso3"] == GERMANY) & (trade["destination_iso3"] != GERMANY),
                        "allocated_volume",
                    ].sum()
                )
        elif scope == "europe":
            production = production[production["region"].map(_is_europe)]
            if trade.empty:
                imports = exports = 0.0
            else:
                source_europe = trade["source_region"].map(_is_europe)
                destination_europe = trade["destination_region"].map(_is_europe)
                imports = float(trade.loc[~source_europe & destination_europe, "allocated_volume"].sum())
                exports = float(trade.loc[source_europe & ~destination_europe, "allocated_volume"].sum())
        else:
            raise ValueError("scope must be 'germany' or 'europe'")

        net_trade = imports - exports
        records.append(
            {
                "year": year,
                "primary_production": float(production["primary_production"].sum()),
                "secondary_production": float(production["secondary_production"].sum()),
                "steel_imports": imports,
                "steel_exports": exports,
                "net_trade": net_trade,
                "trade_direction": "net importer"
                if net_trade > 0
                else ("net exporter" if net_trade < 0 else "balanced"),
            }
        )
    return pd.DataFrame.from_records(records)


def _draw_steel_origin_transition(data: pd.DataFrame, scope: str, output_path: Path) -> None:
    focus = "Germany" if scope == "germany" else "Europe"
    values = data.copy()
    scale = 1e6
    x = list(range(len(values)))
    primary = values["primary_production"] / scale
    secondary = values["secondary_production"] / scale
    net_imports = values["net_trade"].clip(lower=0) / scale
    net_exports = values["net_trade"].clip(upper=0) / scale

    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    ax.bar(x, primary, color="#4c72b0", label="Primary steel production")
    ax.bar(x, secondary, bottom=primary, color="#55a868", label="Secondary production from scrap")
    ax.bar(x, net_imports, bottom=primary + secondary, color="#dd8452", label="Net steel imports")
    ax.bar(x, net_exports, color="#c44e52", label="Net steel exports")
    ax.axhline(0, color="#333333", linewidth=0.8)
    positive_max = float((primary + secondary + net_imports).max()) if len(values) else 0.0
    negative_min = float(net_exports.min()) if len(values) else 0.0
    value_range = max(positive_max - negative_min, 1.0)
    ax.set_ylim(negative_min - 0.06 * value_range, positive_max + 0.06 * value_range)
    ax.set_xticks(x, values["year"].astype(int))
    ax.set_ylabel("Million tonnes of steel")
    ax.set_xlabel("Simulation year")
    ax.set_title(f"Steel production origin and net trade — {focus}")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True)
    ax.text(
        0.995,
        0.01,
        "Secondary share = scrap / all metallic charges per steel furnace · Trade = imports − exports",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _create_steel_transition_outputs(
    inputs: AnalysisInputs,
    plants: pd.DataFrame,
    iso_to_region: dict[str, str],
    country_to_iso: dict[str, str],
    destination: Path,
) -> list[Path]:
    if plants.empty:
        warnings.warn("No post-processed furnace CSV found; skipping steel transition charts.", stacklevel=2)
        return []
    yearly_trade = _steel_trade_by_year(inputs.run_dir, plants, iso_to_region, country_to_iso)
    created: list[Path] = []
    for scope in ("germany", "europe"):
        transition = calculate_steel_production_origin_transition(plants, yearly_trade, scope)
        csv_path = destination / f"{scope}_steel_production_origin_transition.csv"
        image_path = destination / f"{scope}_steel_production_origin_transition.png"
        destination.mkdir(parents=True, exist_ok=True)
        transition.to_csv(csv_path, index=False)
        _draw_steel_origin_transition(transition, scope, image_path)
        created.extend([image_path, csv_path])
    return created


def run_regional_trade_analysis(
    run_dir: Path,
    *,
    year: int | None = None,
    commodity: str = "steel",
    output_dir: Path | None = None,
    trade_csv: Path | None = None,
    plant_csv: Path | None = None,
    shapefile: Path | None = None,
) -> list[Path]:
    """Generate Germany and Europe maps plus their underlying aggregate CSVs."""
    inputs = discover_inputs(run_dir, year, trade_csv, plant_csv, shapefile)
    raw_trade = pd.read_csv(inputs.trade_csv)
    plants = pd.read_csv(inputs.plant_csv) if inputs.plant_csv else pd.DataFrame()
    iso_to_region, country_to_iso = _load_country_geography(inputs.run_dir, plants)
    trade = _prepare_trade(
        raw_trade,
        plants,
        inputs.year,
        commodity,
        iso_to_region=iso_to_region,
        country_to_iso=country_to_iso,
    )
    if trade.empty:
        raise ValueError(f"No positive allocations found for commodity {commodity!r} in {inputs.trade_csv}")
    world = gpd.read_file(inputs.shapefile)
    if world.crs is not None and not world.crs.is_geographic:
        world = world.to_crs(4326)

    destination = (output_dir or inputs.run_dir / "regional_trade_analysis" / str(inputs.year)).resolve()
    created: list[Path] = []
    map_created = False
    for scope in ("germany", "europe"):
        flows = aggregate_flows(trade, scope)
        if flows.empty:
            warnings.warn(
                f"No {commodity} trade flows to or from {scope.title()} were found; skipping that map.",
                stacklevel=2,
            )
            continue
        production, demand = _node_metrics(trade, plants, inputs.year, commodity, scope)
        image_path = destination / f"{scope}_trade_{commodity}.png"
        flow_path = destination / f"{scope}_trade_{commodity}_flows.csv"
        node_path = destination / f"{scope}_trade_{commodity}_nodes.csv"
        _draw_map(world, trade, flows, production, demand, scope, commodity, inputs.year, image_path)
        destination.mkdir(parents=True, exist_ok=True)
        flows.to_csv(flow_path, index=False)
        _node_table(production, demand).to_csv(node_path, index=False)
        created.extend([image_path, flow_path, node_path])
        map_created = True
    if not map_created:
        raise ValueError(f"No {commodity} trade flows to or from Germany or Europe were found")
    created.extend(_create_steel_transition_outputs(inputs, plants, iso_to_region, country_to_iso, destination))
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create standalone Germany and Europe trade-flow maps from a completed simulation run."
    )
    parser.add_argument("run_dir", type=Path, help="Simulation output directory (contains TM/ and post_processed CSV)")
    parser.add_argument("--year", type=int, help="Year to analyse; defaults to the latest trade-allocation year")
    parser.add_argument("--commodity", default="steel", help="Commodity in the allocation CSV (default: steel)")
    parser.add_argument("--output-dir", type=Path, help="Destination; defaults below the simulation run directory")
    parser.add_argument("--trade-csv", type=Path, help="Explicit trade-allocation CSV")
    parser.add_argument("--plant-csv", type=Path, help="Explicit post-processed furnace CSV")
    parser.add_argument("--shapefile", type=Path, help="Natural Earth country .shp file or its containing directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    created = run_regional_trade_analysis(
        args.run_dir,
        year=args.year,
        commodity=args.commodity,
        output_dir=args.output_dir,
        trade_csv=args.trade_csv,
        plant_csv=args.plant_csv,
        shapefile=args.shapefile,
    )
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
