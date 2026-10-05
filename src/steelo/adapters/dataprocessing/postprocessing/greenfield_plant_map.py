"""Map greenfield capacity for one run or every scenario below a results directory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Iterable
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Package-qualified imports also work when this file is loaded directly with
# ``%run`` or imported from its directory in an interactive Python session.
from steelo.adapters.dataprocessing.postprocessing.regional_trade_analysis import _resolve_shapefile
from steelo.adapters.dataprocessing.postprocessing.route_classification import (
    TECHNOLOGY_ROUTE_COLOURS,
    technology_route_label,
)
from steelo.adapters.dataprocessing.postprocessing.world_scenario_analysis import (
    SCENARIO_GRID_ORDER,
    SCENARIO_TITLES,
    TECHNOLOGY_COLOURS,
)


PRODUCT_MARKERS = {"iron": "o", "steel": "s"}
LOCATION_PATTERN = re.compile(r"lat=([-+0-9.eE]+),\s*lon=([-+0-9.eE]+)")
OUTPUT_COLUMNS = [
    "operational_year",
    "plant_group_id",
    "plant_id",
    "furnace_group_ids",
    "iso3",
    "country",
    "region",
    "product",
    "technology",
    "chosen_reductant",
    "technology_route",
    "capacity_tpa",
    "capacity_mtpa",
    "latitude",
    "longitude",
]


@dataclass(frozen=True)
class GreenfieldRun:
    """A resolved simulation run and its human-readable scenario title."""

    index: int | None
    title: str
    run_dir: Path


def _completed_run(scenario_dir: Path) -> Path | None:
    """Return the newest run containing a fleet decision file."""
    candidates = [scenario_dir, scenario_dir / "latest", *sorted(scenario_dir.glob("sim_*"), reverse=True)]
    for candidate in candidates:
        if (candidate / "plant_agent_fleet_decisions.csv").is_file():
            return candidate.resolve()
    return None


def discover_greenfield_runs(input_dir: Path, skip_runs: Iterable[int] = (11,)) -> list[GreenfieldRun]:
    """Resolve either one run or all available scenario runs below ``input_dir``."""
    input_dir = input_dir.resolve()
    if (input_dir / "plant_agent_fleet_decisions.csv").is_file():
        return [GreenfieldRun(None, input_dir.name, input_dir)]

    skipped = set(skip_runs)
    runs: list[GreenfieldRun] = []
    children = input_dir.iterdir() if input_dir.is_dir() else []
    for scenario_dir in sorted(children):
        if not scenario_dir.is_dir():
            continue
        match = re.fullmatch(r"master_input_(\d+)", scenario_dir.name)
        index = int(match.group(1)) if match else None
        if index is not None and index in skipped:
            continue
        run_dir = _completed_run(scenario_dir)
        if run_dir is None:
            continue
        title = SCENARIO_TITLES[index] if index is not None and index < len(SCENARIO_TITLES) else scenario_dir.name
        runs.append(GreenfieldRun(index, title, run_dir))

    if not runs:
        raise FileNotFoundError(f"No completed scenario runs found below {input_dir}")
    return sorted(runs, key=lambda run: (run.index is None, run.index if run.index is not None else run.title))


def _fleet_path(run_dir: Path, fleet_csv: Path | None = None) -> Path:
    path = fleet_csv or run_dir / "plant_agent_fleet_decisions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parse_location(value: object) -> tuple[float, float] | None:
    match = LOCATION_PATTERN.search(str(value))
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _trade_location_lookup(run_dir: Path, furnace_ids: set[str]) -> dict[str, tuple[float, float]]:
    """Recover source coordinates from legacy runs whose fleet CSV lacks coordinates."""
    locations: dict[str, tuple[float, float]] = {}
    for path in sorted((run_dir / "TM").glob("steel_trade_allocations_*.csv")):
        try:
            rows = pd.read_csv(path, usecols=["source_id", "source_location"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        rows["source_id"] = rows["source_id"].astype(str)
        rows = rows[rows["source_id"].isin(furnace_ids)].drop_duplicates("source_id")
        for row in rows.itertuples(index=False):
            parsed = _parse_location(row.source_location)
            if parsed is not None:
                locations[str(row.source_id)] = parsed
    return locations


def _plant_route_table(run_dir: Path) -> pd.DataFrame:
    """Load reductants from the post-processed plant table for legacy fleet CSVs."""
    paths = sorted(run_dir.glob("post_processed_*.csv"), key=lambda path: path.stat().st_mtime)
    columns = ["year", "furnace_group_id", "chosen_reductant"]
    if not paths:
        return pd.DataFrame(columns=columns)
    routes = pd.read_csv(paths[-1], usecols=lambda column: column in columns, low_memory=False)
    if not set(columns).issubset(routes.columns):
        return pd.DataFrame(columns=columns)
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce")
    routes["furnace_group_id"] = routes["furnace_group_id"].astype(str)
    return routes.drop_duplicates(["year", "furnace_group_id"])[columns]


def summarise_greenfield_capacity(
    run_dir: Path,
    *,
    fleet_csv: Path | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> pd.DataFrame:
    """Return one record per greenfield plant, product, and technology at first operation.

    Runtime-born plants are registered in ``indi_<ISO3>`` plant groups. Only
    furnaces present in the plant's first active year are counted, so a later
    expansion at a greenfield site is not misclassified.
    """
    run_dir = run_dir.resolve()
    fleet = pd.read_csv(_fleet_path(run_dir, fleet_csv), low_memory=False)
    required = {
        "year",
        "plant_group_id",
        "plant_id",
        "furnace_group_id",
        "product",
        "technology",
        "capacity",
        "iso3",
    }
    if missing := required - set(fleet.columns):
        raise ValueError(f"Fleet CSV is missing columns: {', '.join(sorted(missing))}")

    fleet["year"] = pd.to_numeric(fleet["year"], errors="coerce")
    fleet["capacity"] = pd.to_numeric(fleet["capacity"], errors="coerce")
    fleet["product"] = fleet["product"].astype(str).str.casefold()
    fleet = fleet[
        fleet["plant_group_id"].astype(str).str.startswith("indi_")
        & fleet["product"].isin(PRODUCT_MARKERS)
        & fleet["capacity"].gt(0)
    ].copy()
    if fleet.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    routes = _plant_route_table(run_dir)
    fleet["furnace_group_id"] = fleet["furnace_group_id"].astype(str)
    fleet = fleet.merge(routes, on=["year", "furnace_group_id"], how="left")
    fleet["chosen_reductant"] = fleet["chosen_reductant"].fillna("").astype(str)
    fleet["technology_route"] = [
        technology_route_label(technology, reductant)
        for technology, reductant in zip(fleet["technology"], fleet["chosen_reductant"])
    ]

    fleet["operational_year"] = fleet.groupby("plant_id")["year"].transform("min")
    fleet = fleet[fleet["year"] == fleet["operational_year"]]
    fleet = fleet.sort_values(["year", "furnace_group_id"]).drop_duplicates("furnace_group_id", keep="first")
    fleet = fleet.rename(columns={"capacity": "capacity_tpa"})
    if year_from is not None:
        fleet = fleet[fleet["operational_year"] >= year_from]
    if year_to is not None:
        fleet = fleet[fleet["operational_year"] <= year_to]
    if fleet.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in ("latitude", "longitude"):
        if column not in fleet:
            fleet[column] = np.nan
        fleet[column] = pd.to_numeric(fleet[column], errors="coerce")

    missing_location = fleet["latitude"].isna() | fleet["longitude"].isna()
    if missing_location.any():
        lookup = _trade_location_lookup(run_dir, set(fleet.loc[missing_location, "furnace_group_id"].astype(str)))
        for index in fleet.index[missing_location]:
            location = lookup.get(str(fleet.at[index, "furnace_group_id"]))
            if location is not None:
                fleet.at[index, "latitude"], fleet.at[index, "longitude"] = location

    for column in ("country", "region"):
        if column not in fleet:
            fleet[column] = pd.NA
    fleet["technology"] = fleet["technology"].fillna("Unknown").astype(str)
    group_columns = [
        "operational_year",
        "plant_group_id",
        "plant_id",
        "iso3",
        "product",
        "technology",
        "chosen_reductant",
        "technology_route",
    ]
    fleet = fleet.groupby(group_columns, dropna=False, as_index=False).agg(
        country=("country", "first"),
        region=("region", "first"),
        furnace_group_ids=("furnace_group_id", lambda values: ";".join(sorted(set(map(str, values))))),
        capacity_tpa=("capacity_tpa", "sum"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )
    fleet["capacity_mtpa"] = fleet["capacity_tpa"] / 1e6
    return fleet[OUTPUT_COLUMNS].sort_values(
        ["operational_year", "product", "technology", "capacity_tpa"],
        ascending=[True, True, True, False],
        ignore_index=True,
    )


def _technology_colours(technologies: Iterable[str]) -> dict[str, object]:
    """Return stable named colours, with deterministic fallbacks for new technologies."""
    names = sorted(set(map(str, technologies)))
    fallback = plt.get_cmap("tab20")
    colours: dict[str, object] = {**TECHNOLOGY_COLOURS, **TECHNOLOGY_ROUTE_COLOURS}
    unknown = [name for name in names if name not in colours]
    colours.update({name: fallback(index % fallback.N) for index, name in enumerate(unknown)})
    return colours


def _greenfield_figure(
    summary: pd.DataFrame,
    shapefile: Path | None,
    *,
    title: str,
    legend_technologies: Iterable[str] | None = None,
    figsize: tuple[float, float] = (16, 9),
    dpi: int = 180,
) -> plt.Figure:
    """Build a global map with technology colour and product marker shape."""
    located = summary.dropna(subset=["latitude", "longitude"]).copy()
    route_column = "technology_route" if "technology_route" in located else "technology"
    if shapefile is not None:
        world = gpd.read_file(shapefile)
        if world.crs is not None and not world.crs.is_geographic:
            world = world.to_crs(4326)
        fig, axis = plt.subplots(figsize=figsize, dpi=dpi)
        world.plot(ax=axis, color="#F2F1ED", edgecolor="#A9A9A9", linewidth=0.35)
        point_transform = None
    else:
        fig, axis = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={"projection": ccrs.PlateCarree()})
        axis.add_feature(cfeature.LAND, facecolor="#F2F1ED")
        axis.add_feature(cfeature.COASTLINE, edgecolor="#A9A9A9", linewidth=0.35)
        point_transform = ccrs.PlateCarree()

    size_scale = 65.0
    technologies = list(legend_technologies if legend_technologies is not None else located[route_column].unique())
    colours = _technology_colours(technologies)
    for technology in technologies:
        for product, marker in PRODUCT_MARKERS.items():
            rows = located[(located[route_column] == technology) & (located["product"] == product)]
            if rows.empty:
                continue
            axis.scatter(
                rows["longitude"],
                rows["latitude"],
                s=rows["capacity_mtpa"] * size_scale,
                color=colours[technology],
                marker=marker,
                alpha=0.68,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
                transform=point_transform,
            )

    capacity_max = float(summary["capacity_mtpa"].max()) if not summary.empty else 0.0
    examples = [value for value in (1.0, 2.5, 5.0, 10.0) if value <= capacity_max]
    if capacity_max and not examples:
        examples = [round(capacity_max, 2)]
    technology_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=colours[name],
            markeredgecolor="white",
            label=name,
        )
        for name in technologies
    ]
    if technology_handles:
        technology_legend = axis.legend(handles=technology_handles, title="Technology", loc="lower left", ncol=2)
        axis.add_artist(technology_legend)
    product_handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            markersize=8,
            markerfacecolor="#777777",
            markeredgecolor="white",
            label=product.title(),
        )
        for product, marker in PRODUCT_MARKERS.items()
    ]
    product_legend = axis.legend(handles=product_handles, title="Product", loc="upper left")
    axis.add_artist(product_legend)
    if examples:
        capacity_handles = [
            axis.scatter([], [], s=value * size_scale, facecolor="#777777", alpha=0.45, edgecolor="white")
            for value in examples
        ]
        axis.legend(capacity_handles, [f"{value:g} Mtpa" for value in examples], title="Capacity", loc="lower right")

    if located.empty:
        axis.text(
            0.5,
            0.5,
            "No greenfield capacity operating yet",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color="#555555",
        )
    axis.set_title(title, fontsize=15)
    if shapefile is not None:
        axis.set_xlim(-180, 180)
        axis.set_ylim(-60, 90)
    else:
        axis.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(color="#D5D5D5", linewidth=0.35, alpha=0.5)
    fig.tight_layout()
    return fig


def draw_greenfield_map(
    summary: pd.DataFrame,
    shapefile: Path | None,
    output_path: Path,
    *,
    title: str = "New greenfield iron and steel capacity",
) -> None:
    """Draw the final all-years global bubble map."""
    fig = _greenfield_figure(summary, shapefile, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_greenfield_evolution_gif(
    summary: pd.DataFrame,
    shapefile: Path | None,
    output_path: Path,
    *,
    scenario_title: str,
    year_from: int = 2025,
    year_to: int = 2060,
    frame_duration_ms: int = 350,
) -> None:
    """Write a cumulative annual animation of newly operating capacity."""
    route_column = "technology_route" if "technology_route" in summary else "technology"
    technologies = sorted(summary[route_column].unique()) if not summary.empty else []
    frames: list[Image.Image] = []
    for year in range(year_from, year_to + 1):
        cumulative = summary[summary["operational_year"] <= year]
        fig = _greenfield_figure(
            cumulative,
            shapefile,
            title=f"{scenario_title} — cumulative greenfield capacity in {year}",
            legend_technologies=technologies,
            figsize=(12, 6.75),
            dpi=100,
        )
        buffer = BytesIO()
        fig.savefig(buffer, format="png", facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        with Image.open(buffer) as image:
            frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=frame_duration_ms, loop=0, disposal=2)


def draw_greenfield_map_matrix(
    maps_by_index: dict[int, Path],
    output_path: Path,
    *,
    columns: int = 4,
    rows: int = 4,
    cell_size: tuple[int, int] = (1000, 450),
) -> None:
    """Combine final scenario maps in their scenario-index positions."""
    cell_width, cell_height = cell_size
    matrix = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()

    scenario_order = SCENARIO_GRID_ORDER if columns * rows == len(SCENARIO_GRID_ORDER) else range(columns * rows)
    for position, index in enumerate(scenario_order):
        left = (position % columns) * cell_width
        top = (position // columns) * cell_height
        path = maps_by_index.get(index)
        if path is not None and path.is_file():
            with Image.open(path) as source:
                cell = ImageOps.contain(source.convert("RGB"), cell_size, Image.Resampling.LANCZOS)
            offset = (left + (cell_width - cell.width) // 2, top + (cell_height - cell.height) // 2)
            matrix.paste(cell, offset)
            continue

        label = SCENARIO_TITLES[index] if index < len(SCENARIO_TITLES) else f"Scenario {index}"
        message = f"{label} — unavailable"
        draw = ImageDraw.Draw(matrix)
        bounds = draw.textbbox((0, 0), message, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        draw.text(
            (left + (cell_width - text_width) / 2, top + (cell_height - text_height) / 2),
            message,
            fill="#666666",
            font=font,
        )

    draw = ImageDraw.Draw(matrix)
    for column in range(1, columns):
        x = column * cell_width
        draw.line((x, 0, x, matrix.height), fill="#dddddd", width=2)
    for row in range(1, rows):
        y = row * cell_height
        draw.line((0, y, matrix.width, y), fill="#dddddd", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.save(output_path, format="PNG", optimize=True)


def run_greenfield_map(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    fleet_csv: Path | None = None,
    shapefile: Path | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    scenario_title: str | None = None,
    frame_duration_ms: int = 350,
) -> list[Path]:
    """Generate the table, final map, and animation for one simulation run."""
    run_dir = run_dir.resolve()
    summary = summarise_greenfield_capacity(run_dir, fleet_csv=fleet_csv, year_from=year_from, year_to=year_to)
    destination = (output_dir or run_dir / "greenfield_plant_map").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    first_year = year_from or 2025
    last_year = year_to or 2060
    csv_path = destination / "greenfield_plant_capacity.csv"
    map_path = destination / "greenfield_plant_capacity_map.png"
    gif_path = destination / f"greenfield_capacity_evolution_{first_year}_{last_year}.gif"
    summary.to_csv(csv_path, index=False)
    missing = int(summary[["latitude", "longitude"]].isna().any(axis=1).sum())
    if missing:
        warnings.warn(
            f"{missing} of {len(summary)} greenfield plant/product/technology records have no recoverable "
            "coordinates and are listed in the CSV but omitted from the map.",
            stacklevel=2,
        )
    try:
        resolved_shapefile = _resolve_shapefile(run_dir, shapefile)
    except FileNotFoundError:
        if shapefile is not None:
            raise
        resolved_shapefile = None
    label = scenario_title or run_dir.name
    draw_greenfield_map(
        summary,
        resolved_shapefile,
        map_path,
        title=f"{label} — all greenfield capacity ({first_year}–{last_year})",
    )
    draw_greenfield_evolution_gif(
        summary,
        resolved_shapefile,
        gif_path,
        scenario_title=label,
        year_from=first_year,
        year_to=last_year,
        frame_duration_ms=frame_duration_ms,
    )
    return [csv_path, map_path, gif_path]


def run_greenfield_maps(
    input_dir: Path,
    *,
    output_dir: Path | None = None,
    fleet_csv: Path | None = None,
    shapefile: Path | None = None,
    year_from: int = 2025,
    year_to: int = 2060,
    skip_runs: Iterable[int] = (11,),
    frame_duration_ms: int = 350,
) -> list[Path]:
    """Generate outputs for one run or every discovered scenario."""
    input_dir = input_dir.resolve()
    runs = discover_greenfield_runs(input_dir, skip_runs=skip_runs)
    batch = len(runs) > 1 or not (input_dir / "plant_agent_fleet_decisions.csv").is_file()
    if batch and fleet_csv is not None:
        raise ValueError("--fleet-csv can only be used when input_dir is one simulation run")
    root = (
        output_dir or (input_dir / "greenfield_plant_maps" if batch else input_dir / "greenfield_plant_map")
    ).resolve()
    outputs: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    maps_by_index: dict[int, Path] = {}
    for run in runs:
        destination = root / run.title if batch else root
        try:
            generated = run_greenfield_map(
                run.run_dir,
                output_dir=destination,
                fleet_csv=fleet_csv,
                shapefile=shapefile,
                year_from=year_from,
                year_to=year_to,
                scenario_title=run.title,
                frame_duration_ms=frame_duration_ms,
            )
        except (FileNotFoundError, ValueError) as error:
            warnings.warn(f"Skipping {run.title}: {error}", stacklevel=2)
            manifest_rows.append(
                {
                    "scenario_index": run.index,
                    "scenario": run.title,
                    "run_dir": run.run_dir,
                    "status": "skipped",
                    "message": str(error),
                }
            )
            continue
        outputs.extend(generated)
        if run.index is not None:
            maps_by_index[run.index] = generated[1]
        manifest_rows.append(
            {
                "scenario_index": run.index,
                "scenario": run.title,
                "run_dir": run.run_dir,
                "status": "created",
                "message": "",
            }
        )
    if batch:
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "scenario_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
        matrix_path = input_dir / "greenfield_plant_capacity_maps_matrix.png"
        draw_greenfield_map_matrix(maps_by_index, matrix_path)
        outputs.extend([manifest, matrix_path])
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="One completed run or a directory containing scenario folders")
    parser.add_argument("--output-dir", type=Path, help="Destination directory")
    parser.add_argument("--fleet-csv", type=Path, help="Explicit plant_agent_fleet_decisions.csv (single run only)")
    parser.add_argument("--shapefile", type=Path, help="Natural Earth country shapefile or containing directory")
    parser.add_argument("--year-from", type=int, default=2025, help="First animation year (default: 2025)")
    parser.add_argument("--year-to", type=int, default=2060, help="Last animation year (default: 2060)")
    parser.add_argument("--skip-runs", type=int, nargs="*", default=[], help="Scenario indices to skip (default: none)")
    parser.add_argument("--frame-duration-ms", type=int, default=350, help="GIF frame duration in milliseconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.year_from > args.year_to:
        raise ValueError("--year-from must not be later than --year-to")
    for path in run_greenfield_maps(
        args.input_dir,
        output_dir=args.output_dir,
        fleet_csv=args.fleet_csv,
        shapefile=args.shapefile,
        year_from=args.year_from,
        year_to=args.year_to,
        skip_runs=args.skip_runs,
        frame_duration_ms=args.frame_duration_ms,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
