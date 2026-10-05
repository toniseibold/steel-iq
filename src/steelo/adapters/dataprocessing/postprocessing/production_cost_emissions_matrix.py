"""Compare 2050 iron and primary/secondary steel cost and emissions across scenarios."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from .analysis_scope import EUROPE_ANALYSIS_DESCRIPTION, EUROPE_ANALYSIS_LABEL
from .europe_scenario_post_analysis import (
    METALLIC_INPUTS,
    ScenarioInput,
    _europe_iso3,
    discover_scenario_inputs,
)
from .world_scenario_analysis import (
    DIRECT_EMISSIONS,
    INDIRECT_EMISSIONS,
    SCENARIO_GRID_ORDER,
    SCENARIO_TITLES,
)


PRODUCT_STAGES = ("iron", "primary_steel", "secondary_steel")
STAGE_LABELS = ("Iron", "Primary steel", "Secondary steel")
GEOGRAPHIES = ("Global", "EU")
MATERIAL_COST = "total_cost - material and allocation"
REQUIRED_COLUMNS = {
    "year",
    "iso3",
    "furnace_group_id",
    "product",
    "production",
    "unit_production_cost",
    "feedstock",
    "demand",
    MATERIAL_COST,
    DIRECT_EMISSIONS,
    INDIRECT_EMISSIONS,
}


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def _prepare_plants(path: Path, year: int) -> pd.DataFrame:
    plants = pd.read_csv(path, usecols=lambda column: column in REQUIRED_COLUMNS, low_memory=False)
    if missing := REQUIRED_COLUMNS - set(plants.columns):
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    plants["year"] = pd.to_numeric(plants["year"], errors="coerce")
    plants = plants[plants["year"].eq(year)].copy()
    plants["product"] = plants["product"].astype(str).str.casefold()
    plants["feedstock"] = plants["feedstock"].fillna("").astype(str).str.casefold()
    for column in (
        "production",
        "unit_production_cost",
        "demand",
        MATERIAL_COST,
        DIRECT_EMISSIONS,
        INDIRECT_EMISSIONS,
    ):
        plants[column] = pd.to_numeric(plants[column], errors="coerce")
    plants["demand"] = plants["demand"].fillna(0.0).clip(lower=0.0)
    plants[MATERIAL_COST] = plants[MATERIAL_COST].fillna(0.0)
    plants[DIRECT_EMISSIONS] = plants[DIRECT_EMISSIONS].fillna(0.0)
    plants[INDIRECT_EMISSIONS] = plants[INDIRECT_EMISSIONS].fillna(0.0)
    return plants


def _scope_metrics(plants: pd.DataFrame, iso3: set[str] | None) -> list[dict[str, object]]:
    scoped = plants if iso3 is None else plants[plants["iso3"].astype(str).str.upper().isin(iso3)]
    scoped = scoped[scoped["production"].gt(0)].copy()

    iron_rows = scoped[scoped["product"].eq("iron")]
    iron = iron_rows.drop_duplicates("furnace_group_id")
    iron_production = float(iron["production"].sum())
    valid_iron_cost = iron["unit_production_cost"].notna()
    iron_cost_total = float(
        (iron.loc[valid_iron_cost, "production"] * iron.loc[valid_iron_cost, "unit_production_cost"]).sum()
    )
    iron_direct = float(iron[DIRECT_EMISSIONS].sum())
    iron_indirect = float(iron[INDIRECT_EMISSIONS].sum())
    iron_cost = _divide(iron_cost_total, float(iron.loc[valid_iron_cost, "production"].sum()))
    iron_direct_intensity = _divide(iron_direct, iron_production)
    iron_total_intensity = _divide(iron_direct + iron_indirect, iron_production)

    records: list[dict[str, object]] = [
        {
            "product_stage": "iron",
            "production_t": iron_production,
            "metallic_input_t": math.nan,
            "conversion_cost_total_usd": math.nan,
            "metallic_cost_total_usd": math.nan,
            "avg_cost_usd_per_t": iron_cost,
            "direct_emissions_tco2e_per_t": iron_direct_intensity,
            "direct_indirect_emissions_tco2e_per_t": iron_total_intensity,
        }
    ]

    steel_rows = scoped[scoped["product"].eq("steel")].copy()
    if steel_rows.empty:
        for stage in PRODUCT_STAGES[1:]:
            records.append(
                {
                    "product_stage": stage,
                    "production_t": 0.0,
                    "metallic_input_t": 0.0,
                    "conversion_cost_total_usd": 0.0,
                    "metallic_cost_total_usd": 0.0,
                    "avg_cost_usd_per_t": math.nan,
                    "direct_emissions_tco2e_per_t": math.nan,
                    "direct_indirect_emissions_tco2e_per_t": math.nan,
                }
            )
        return records

    metadata = steel_rows.drop_duplicates("furnace_group_id").set_index("furnace_group_id")
    metallic = steel_rows[steel_rows["feedstock"].isin(METALLIC_INPUTS)].copy()
    metallic["scrap_demand"] = metallic["demand"].where(metallic["feedstock"].eq("scrap"), 0.0)
    metallic["primary_demand"] = metallic["demand"].where(~metallic["feedstock"].eq("scrap"), 0.0)
    metallic["scrap_cost"] = metallic[MATERIAL_COST].where(metallic["feedstock"].eq("scrap"), 0.0)
    metallic["primary_cost"] = metallic[MATERIAL_COST].where(~metallic["feedstock"].eq("scrap"), 0.0)
    charges = metallic.groupby("furnace_group_id").agg(
        metallic_demand=("demand", "sum"),
        scrap_demand=("scrap_demand", "sum"),
        primary_demand=("primary_demand", "sum"),
        metallic_cost=(MATERIAL_COST, "sum"),
        scrap_cost=("scrap_cost", "sum"),
        primary_cost=("primary_cost", "sum"),
    )
    steel = metadata.join(charges, how="left").fillna(
        {
            "metallic_demand": 0.0,
            "scrap_demand": 0.0,
            "primary_demand": 0.0,
            "metallic_cost": 0.0,
            "scrap_cost": 0.0,
            "primary_cost": 0.0,
        }
    )
    steel["secondary_share"] = np.where(
        steel["metallic_demand"].gt(0), steel["scrap_demand"] / steel["metallic_demand"], 0.0
    )
    steel["primary_share"] = 1.0 - steel["secondary_share"]
    steel["primary_output"] = steel["production"] * steel["primary_share"]
    steel["secondary_output"] = steel["production"] * steel["secondary_share"]
    steel["conversion_unit_cost"] = steel["unit_production_cost"] - np.where(
        steel["production"].gt(0), steel["metallic_cost"] / steel["production"], 0.0
    )
    steel["conversion_cost_primary"] = steel["conversion_unit_cost"] * steel["primary_output"]
    steel["conversion_cost_secondary"] = steel["conversion_unit_cost"] * steel["secondary_output"]
    steel["direct_primary"] = steel[DIRECT_EMISSIONS] * steel["primary_share"]
    steel["direct_secondary"] = steel[DIRECT_EMISSIONS] * steel["secondary_share"]
    steel["total_primary"] = (steel[DIRECT_EMISSIONS] + steel[INDIRECT_EMISSIONS]) * steel["primary_share"]
    steel["total_secondary"] = (steel[DIRECT_EMISSIONS] + steel[INDIRECT_EMISSIONS]) * steel["secondary_share"]

    primary_output = float(steel["primary_output"].sum())
    primary_charge = float(steel["primary_demand"].sum())
    primary_conversion_cost = float(steel["conversion_cost_primary"].sum())
    primary_upstream_cost = iron_cost * primary_charge
    primary_direct = float(steel["direct_primary"].sum()) + iron_direct_intensity * primary_charge
    primary_total = float(steel["total_primary"].sum()) + iron_total_intensity * primary_charge
    records.append(
        {
            "product_stage": "primary_steel",
            "production_t": primary_output,
            "metallic_input_t": primary_charge,
            "conversion_cost_total_usd": primary_conversion_cost,
            "metallic_cost_total_usd": primary_upstream_cost,
            "avg_cost_usd_per_t": _divide(primary_conversion_cost + primary_upstream_cost, primary_output),
            "direct_emissions_tco2e_per_t": _divide(primary_direct, primary_output),
            "direct_indirect_emissions_tco2e_per_t": _divide(primary_total, primary_output),
        }
    )

    secondary_output = float(steel["secondary_output"].sum())
    secondary_charge = float(steel["scrap_demand"].sum())
    secondary_conversion_cost = float(steel["conversion_cost_secondary"].sum())
    secondary_scrap_cost = float(steel["scrap_cost"].sum())
    records.append(
        {
            "product_stage": "secondary_steel",
            "production_t": secondary_output,
            "metallic_input_t": secondary_charge,
            "conversion_cost_total_usd": secondary_conversion_cost,
            "metallic_cost_total_usd": secondary_scrap_cost,
            "avg_cost_usd_per_t": _divide(secondary_conversion_cost + secondary_scrap_cost, secondary_output),
            "direct_emissions_tco2e_per_t": _divide(float(steel["direct_secondary"].sum()), secondary_output),
            "direct_indirect_emissions_tco2e_per_t": _divide(float(steel["total_secondary"].sum()), secondary_output),
        }
    )
    return records


def summarise_scenario(
    scenario: ScenarioInput,
    *,
    year: int = 2050,
    include_ukraine: bool = False,
) -> pd.DataFrame:
    """Return the 2×3 global/EU matrix values for one scenario."""
    plants = _prepare_plants(scenario.plants_csv, year)
    europe = _europe_iso3(scenario, include_ukraine)
    frames: list[pd.DataFrame] = []
    for geography, countries in (("Global", None), ("EU", europe)):
        frame = pd.DataFrame.from_records(_scope_metrics(plants, countries))
        frame.insert(0, "geography", geography)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "year", year)
    result.insert(0, "scenario_title", scenario.title)
    result.insert(0, "scenario_index", scenario.index)
    return result


def build_summary(
    scenarios: Iterable[ScenarioInput],
    *,
    year: int = 2050,
    include_ukraine: bool = False,
) -> pd.DataFrame:
    frames = [summarise_scenario(item, year=year, include_ukraine=include_ukraine) for item in scenarios]
    if not frames:
        raise ValueError("No completed scenarios were available")
    return pd.concat(frames, ignore_index=True)


def plot_metric_matrix(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    decimals: int,
    geographies: Iterable[str] = GEOGRAPHIES,
) -> None:
    """Plot an annotated geography × product heatmap in each 4×4 scenario cell."""
    geographies = tuple(geographies)
    values = pd.to_numeric(summary[metric], errors="coerce")
    finite = values[np.isfinite(values)]
    vmax = max(float(finite.max()) if not finite.empty else 1.0, 1e-9)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#D9D9D9")
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), dpi=180, constrained_layout=True)
    image = None
    for index, axis in zip(SCENARIO_GRID_ORDER, axes.flat):
        selected = summary[summary["scenario_index"].eq(index)]
        if selected.empty:
            axis.axis("off")
            axis.set_title(f"{SCENARIO_TITLES[index]} — unavailable", fontsize=10)
            continue
        matrix = selected.pivot(index="geography", columns="product_stage", values=metric).reindex(
            index=geographies, columns=PRODUCT_STAGES
        )
        image = axis.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap=cmap, norm=norm, aspect="auto")
        axis.set_xticks(range(3), STAGE_LABELS, rotation=25, ha="right", fontsize=8)
        axis.set_yticks(range(len(geographies)), geographies, fontsize=8)
        axis.set_title(SCENARIO_TITLES[index], fontsize=11)
        for row in range(len(geographies)):
            for column in range(3):
                value = matrix.iloc[row, column]
                label = f"{value:.{decimals}f}" if pd.notna(value) else "N/A"
                text_color = "white" if pd.notna(value) and norm(float(value)) > 0.55 else "black"
                axis.text(column, row, label, ha="center", va="center", fontsize=8, color=text_color)
    fig.suptitle(title, fontsize=17)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, location="bottom", shrink=0.55, pad=0.035, aspect=40)
        colorbar.set_label(colorbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_methodology(path: Path, *, include_ukraine: bool) -> None:
    europe_note = "included" if include_ukraine else "excluded"
    path.write_text(
        "\n".join(
            [
                "2050 production cost and emissions methodology",
                "",
                "- All averages are production-weighted.",
                f"- EU scope means {EUROPE_ANALYSIS_DESCRIPTION}.",
                f"- Ukraine is {europe_note}; Turkey follows its model region (MENA) and is outside EU.",
                "- Steel output is split by scrap versus non-scrap metallic charge.",
                "- Steel conversion cost equals furnace unit production cost less purchased metallic inputs.",
                "- Primary steel cost adds average iron production cost multiplied by primary metallic tonnes used.",
                "- Secondary steel cost adds actual scrap input cost to the steel conversion step.",
                "- Primary steel emissions add average iron intensity multiplied by primary metallic tonnes used.",
                "- Secondary steel emissions cover the steelmaking step; recycled scrap carries no upstream burden.",
                "- Direct+indirect uses RS-inspired direct and indirect GHG columns.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_analysis(
    results_dir: Path,
    *,
    output_dir: Path | None = None,
    year: int = 2050,
    skip_runs: Iterable[int] = (),
    include_ukraine: bool = False,
) -> list[Path]:
    results_dir = results_dir.resolve()
    destination = (output_dir or results_dir / f"production_cost_emissions_{year}").resolve()
    scenarios = discover_scenario_inputs(results_dir, skip_runs=skip_runs)
    summary = build_summary(scenarios, year=year, include_ukraine=include_ukraine)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"production_cost_emissions_{year}.csv"
    eu_csv_path = destination / f"eu_production_cost_emissions_{year}.csv"
    cost_path = destination / f"average_production_cost_{year}_scenario_matrix.png"
    direct_path = destination / f"direct_emissions_intensity_{year}_scenario_matrix.png"
    total_path = destination / f"direct_indirect_emissions_intensity_{year}_scenario_matrix.png"
    eu_cost_path = destination / f"eu_average_production_cost_{year}_scenario_matrix.png"
    eu_direct_path = destination / f"eu_direct_emissions_intensity_{year}_scenario_matrix.png"
    eu_total_path = destination / f"eu_direct_indirect_emissions_intensity_{year}_scenario_matrix.png"
    methodology_path = destination / "methodology.txt"
    summary.to_csv(csv_path, index=False)
    eu_summary = summary[summary["geography"].eq("EU")].copy()
    eu_summary.to_csv(eu_csv_path, index=False)
    plot_metric_matrix(
        summary,
        "avg_cost_usd_per_t",
        cost_path,
        title=f"Average production cost — {year}",
        colorbar_label="Production cost [USD/t]",
        decimals=1,
    )
    plot_metric_matrix(
        summary,
        "direct_emissions_tco2e_per_t",
        direct_path,
        title=f"Direct emissions intensity — {year}",
        colorbar_label="Direct emissions [tCO₂e/t]",
        decimals=2,
    )
    plot_metric_matrix(
        summary,
        "direct_indirect_emissions_tco2e_per_t",
        total_path,
        title=f"Direct + indirect emissions intensity — {year}",
        colorbar_label="Direct + indirect emissions [tCO₂e/t]",
        decimals=2,
    )
    plot_metric_matrix(
        eu_summary,
        "avg_cost_usd_per_t",
        eu_cost_path,
        title=f"{EUROPE_ANALYSIS_LABEL}-made iron and steel — average production cost — {year}",
        colorbar_label="Production cost [USD/t]",
        decimals=1,
        geographies=("EU",),
    )
    plot_metric_matrix(
        eu_summary,
        "direct_emissions_tco2e_per_t",
        eu_direct_path,
        title=f"{EUROPE_ANALYSIS_LABEL}-made iron and steel — direct emissions intensity — {year}",
        colorbar_label="Direct emissions [tCO₂e/t]",
        decimals=2,
        geographies=("EU",),
    )
    plot_metric_matrix(
        eu_summary,
        "direct_indirect_emissions_tco2e_per_t",
        eu_total_path,
        title=f"{EUROPE_ANALYSIS_LABEL}-made iron and steel — direct + indirect emissions intensity — {year}",
        colorbar_label="Direct + indirect emissions [tCO₂e/t]",
        decimals=2,
        geographies=("EU",),
    )
    write_methodology(methodology_path, include_ukraine=include_ukraine)
    return [
        csv_path,
        eu_csv_path,
        cost_path,
        direct_path,
        total_path,
        eu_cost_path,
        eu_direct_path,
        eu_total_path,
        methodology_path,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Directory containing master_input_* scenarios")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--year", type=int, default=2050)
    parser.add_argument("--skip-runs", type=int, nargs="*", default=[])
    parser.add_argument("--include-ukraine", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in run_analysis(
        args.results_dir,
        output_dir=args.output_dir,
        year=args.year,
        skip_runs=args.skip_runs,
        include_ukraine=args.include_ukraine,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
