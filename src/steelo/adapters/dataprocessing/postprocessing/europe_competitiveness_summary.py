"""Scenario-matrix summary for European steel and iron competitiveness.

The script scans completed run directories below ``cluster/results`` and writes
Europe-focused summary tables plus a small set of matrix charts. The intent is
to compare how scenario variants affect European production, trade exposure,
costs, subsidies, and emissions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .regional_trade_analysis import _is_europe, _load_country_geography, _prepare_trade


DEFAULT_YEARS = (2030, 2035, 2040)
STEEL = "steel"
IRON = "iron"


@dataclass(frozen=True)
class RunArtifacts:
    scenario_name: str
    master_input_name: str
    run_dir: Path
    post_processed_csv: Path
    simulation_config: Path | None


def _latest_post_processed_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("post_processed_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No post_processed_*.csv found in {run_dir}")
    return candidates[-1]


def discover_runs(results_dir: Path) -> list[RunArtifacts]:
    runs: list[RunArtifacts] = []
    for candidate in sorted(results_dir.glob("*master_input_*")):
        if not candidate.is_dir():
            continue
        latest = candidate / "latest"
        run_candidates = ([latest] if latest.is_dir() else []) + sorted(
            (path for path in candidate.glob("sim_*") if path.is_dir()), reverse=True
        )
        completed = [(run_dir, sorted(run_dir.glob("post_processed_*.csv"))) for run_dir in run_candidates]
        completed = [(run_dir, outputs) for run_dir, outputs in completed if outputs]
        if not completed:
            continue
        run_dir, post_processed = completed[0]
        config_path = run_dir / "simulation_config.json"
        master_input_name = candidate.name.split("master_input_", 1)[1]
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                config = {}
            master_excel = Path(str(config.get("master_excel_path", ""))).stem
            if master_excel:
                master_input_name = master_excel
        runs.append(
            RunArtifacts(
                scenario_name=candidate.name,
                master_input_name=master_input_name,
                run_dir=run_dir,
                post_processed_csv=post_processed[-1],
                simulation_config=config_path if config_path.is_file() else None,
            )
        )
    if not runs:
        raise FileNotFoundError(f"No run directories found below {results_dir}")
    return runs


def _safe_weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = numeric_values.notna() & numeric_weights.gt(0)
    if not valid.any():
        return math.nan
    return float(np.average(numeric_values[valid], weights=numeric_weights[valid]))


def _filter_product_rows(plants: pd.DataFrame, year: int, product: str) -> pd.DataFrame:
    selected = plants.copy()
    if "year" in selected.columns:
        selected = selected[pd.to_numeric(selected["year"], errors="coerce") == year]
    selected = selected[selected["product"].astype(str).str.casefold() == product.casefold()].copy()
    if "furnace_group_id" in selected.columns:
        selected = selected.drop_duplicates("furnace_group_id")
    selected["production"] = pd.to_numeric(selected["production"], errors="coerce").fillna(0.0)
    return selected[selected["production"] > 0].copy()


def _source_metrics_for_year(plants: pd.DataFrame, year: int, product: str) -> pd.DataFrame:
    selected = _filter_product_rows(plants, year, product)
    columns = [
        "furnace_group_id",
        "region",
        "technology",
        "production",
        "unit_production_cost",
        "unit_subsidy_total",
        "unit_carbon_cost",
        "cost_breakdown - material cost (incl. transport and tariffs)",
        "emissions_rs-inspired_direct_ghg",
        "emissions_rs-inspired_indirect_ghg",
    ]
    for column in columns:
        if column not in selected.columns:
            selected[column] = math.nan
    selected = selected[columns].copy()
    selected = selected.rename(columns={"furnace_group_id": "source_id"})
    selected["source_id"] = selected["source_id"].astype(str)
    total_emissions = pd.to_numeric(selected["emissions_rs-inspired_direct_ghg"], errors="coerce").fillna(
        0.0
    ) + pd.to_numeric(selected["emissions_rs-inspired_indirect_ghg"], errors="coerce").fillna(0.0)
    selected["total_emissions_intensity"] = total_emissions / selected["production"]
    return selected


def _load_market_prices(run_dir: Path) -> pd.DataFrame:
    candidates = sorted((run_dir / "data").glob("market_prices_*.csv"))
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(candidates[-1])


def _load_international_iron_trade(run_dir: Path) -> pd.DataFrame:
    candidates = sorted((run_dir / "data").glob("international_iron_trade_*.csv"))
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(candidates[-1])


def summarise_run_year_steel(run: RunArtifacts, year: int) -> dict[str, object] | None:
    trade_csv = run.run_dir / "TM" / f"steel_trade_allocations_{year}.csv"
    if not trade_csv.is_file():
        return None

    plants = pd.read_csv(run.post_processed_csv)
    iso_to_region, country_to_iso = _load_country_geography(run.run_dir, plants)
    trade = _prepare_trade(
        pd.read_csv(trade_csv),
        plants,
        year,
        "steel",
        iso_to_region=iso_to_region,
        country_to_iso=country_to_iso,
    )
    if trade.empty:
        return None

    source_metrics = _source_metrics_for_year(plants, year, STEEL)
    europe_plants = source_metrics[source_metrics["region"].map(_is_europe)].copy()

    source_europe = trade["source_region"].map(_is_europe)
    destination_europe = trade["destination_region"].map(_is_europe)
    imports = trade.loc[~source_europe & destination_europe].copy()
    exports = trade.loc[source_europe & ~destination_europe].copy()
    european_deliveries = trade.loc[destination_europe].copy()
    intra_europe = trade.loc[source_europe & destination_europe].copy()

    europe_production_mt = float(europe_plants["production"].sum() / 1e6)
    europe_demand_mt = float(european_deliveries["allocated_volume"].sum() / 1e6)
    imports_mt = float(imports["allocated_volume"].sum() / 1e6)
    exports_mt = float(exports["allocated_volume"].sum() / 1e6)
    intra_europe_mt = float(intra_europe["allocated_volume"].sum() / 1e6)
    net_imports_mt = imports_mt - exports_mt
    self_sufficiency_ratio = europe_production_mt / europe_demand_mt if europe_demand_mt else math.nan
    import_share = imports_mt / europe_demand_mt if europe_demand_mt else math.nan
    export_share_of_production = exports_mt / europe_production_mt if europe_production_mt else math.nan

    eu_avg_cost = _safe_weighted_average(europe_plants["unit_production_cost"], europe_plants["production"])
    eu_avg_subsidy = _safe_weighted_average(europe_plants["unit_subsidy_total"], europe_plants["production"])
    eu_avg_carbon_cost = _safe_weighted_average(europe_plants["unit_carbon_cost"], europe_plants["production"])
    eu_avg_material_transport = _safe_weighted_average(
        europe_plants["cost_breakdown - material cost (incl. transport and tariffs)"],
        europe_plants["production"],
    )
    eu_avg_emissions = _safe_weighted_average(europe_plants["total_emissions_intensity"], europe_plants["production"])

    imported_sources = imports.merge(source_metrics, on="source_id", how="left", suffixes=("", "_source"))
    imported_sources["matched_volume"] = imported_sources["allocated_volume"].where(
        imported_sources["unit_production_cost"].notna(),
        0.0,
    )
    matched_import_volume_mt = float(imported_sources["matched_volume"].sum() / 1e6)
    imported_cost_coverage = matched_import_volume_mt / imports_mt if imports_mt else math.nan
    imported_avg_source_cost = _safe_weighted_average(
        imported_sources["unit_production_cost"],
        imported_sources["allocated_volume"],
    )
    imported_avg_source_emissions = _safe_weighted_average(
        imported_sources["total_emissions_intensity"],
        imported_sources["allocated_volume"],
    )
    imported_avg_allocation_cost = _safe_weighted_average(
        imported_sources["allocation_cost"], imported_sources["allocated_volume"]
    )
    eu_vs_import_cost_gap = (
        eu_avg_cost - imported_avg_source_cost
        if not math.isnan(eu_avg_cost) and not math.isnan(imported_avg_source_cost)
        else math.nan
    )
    eu_vs_import_emissions_gap = (
        eu_avg_emissions - imported_avg_source_emissions
        if not math.isnan(eu_avg_emissions) and not math.isnan(imported_avg_source_emissions)
        else math.nan
    )

    return {
        "scenario_name": run.scenario_name,
        "master_input_name": run.master_input_name,
        "year": year,
        "europe_production_mt": europe_production_mt,
        "europe_demand_mt": europe_demand_mt,
        "imports_mt": imports_mt,
        "exports_mt": exports_mt,
        "intra_europe_mt": intra_europe_mt,
        "net_imports_mt": net_imports_mt,
        "self_sufficiency_ratio": self_sufficiency_ratio,
        "import_share_of_demand": import_share,
        "export_share_of_production": export_share_of_production,
        "eu_avg_unit_production_cost": eu_avg_cost,
        "eu_avg_unit_subsidy": eu_avg_subsidy,
        "eu_avg_unit_carbon_cost": eu_avg_carbon_cost,
        "eu_avg_material_transport_tariff_cost": eu_avg_material_transport,
        "eu_avg_emissions_intensity": eu_avg_emissions,
        "imported_avg_source_cost": imported_avg_source_cost,
        "imported_avg_source_emissions_intensity": imported_avg_source_emissions,
        "imported_avg_allocation_cost": imported_avg_allocation_cost,
        "imported_source_cost_coverage": imported_cost_coverage,
        "eu_vs_import_cost_gap": eu_vs_import_cost_gap,
        "eu_vs_import_emissions_gap": eu_vs_import_emissions_gap,
    }


def summarise_run_year_iron(run: RunArtifacts, year: int) -> dict[str, object] | None:
    plants = pd.read_csv(run.post_processed_csv)
    source_metrics = _source_metrics_for_year(plants, year, IRON)
    if source_metrics.empty:
        return None

    europe_plants = source_metrics[source_metrics["region"].map(_is_europe)].copy()
    if europe_plants.empty:
        return None

    market_prices = _load_market_prices(run.run_dir)
    market_row = market_prices[pd.to_numeric(market_prices.get("year"), errors="coerce") == year]
    market = market_row.iloc[0] if not market_row.empty else None

    iron_trade = _load_international_iron_trade(run.run_dir)
    iron_trade_year = iron_trade[pd.to_numeric(iron_trade.get("year"), errors="coerce") == year]

    europe_production_mt = float(europe_plants["production"].sum() / 1e6)
    eu_avg_cost = _safe_weighted_average(europe_plants["unit_production_cost"], europe_plants["production"])
    eu_avg_subsidy = _safe_weighted_average(europe_plants["unit_subsidy_total"], europe_plants["production"])
    eu_avg_carbon_cost = _safe_weighted_average(europe_plants["unit_carbon_cost"], europe_plants["production"])
    eu_avg_material_transport = _safe_weighted_average(
        europe_plants["cost_breakdown - material cost (incl. transport and tariffs)"],
        europe_plants["production"],
    )
    eu_avg_emissions = _safe_weighted_average(europe_plants["total_emissions_intensity"], europe_plants["production"])

    global_international_trade_mt = (
        float(pd.to_numeric(iron_trade_year.get("volume_tonnes"), errors="coerce").fillna(0.0).sum() / 1e6)
        if not iron_trade_year.empty
        else math.nan
    )
    global_trade_to_eu_production_ratio = (
        global_international_trade_mt / europe_production_mt
        if europe_production_mt and not math.isnan(global_international_trade_mt)
        else math.nan
    )
    iron_market_price = float(market["iron_price_usd_per_t"]) if market is not None else math.nan
    global_iron_weighted_avg_cost = (
        float(market["iron_weighted_avg_cost_usd_per_t"]) if market is not None else math.nan
    )
    eu_vs_market_price_margin = (
        iron_market_price - eu_avg_cost
        if not math.isnan(iron_market_price) and not math.isnan(eu_avg_cost)
        else math.nan
    )
    eu_cost_to_market_price_ratio = (
        eu_avg_cost / iron_market_price if iron_market_price and not math.isnan(eu_avg_cost) else math.nan
    )
    eu_vs_global_iron_cost_gap = (
        eu_avg_cost - global_iron_weighted_avg_cost
        if not math.isnan(eu_avg_cost) and not math.isnan(global_iron_weighted_avg_cost)
        else math.nan
    )

    return {
        "scenario_name": run.scenario_name,
        "master_input_name": run.master_input_name,
        "year": year,
        "europe_production_mt": europe_production_mt,
        "eu_avg_unit_production_cost": eu_avg_cost,
        "eu_avg_unit_subsidy": eu_avg_subsidy,
        "eu_avg_unit_carbon_cost": eu_avg_carbon_cost,
        "eu_avg_material_transport_tariff_cost": eu_avg_material_transport,
        "eu_avg_emissions_intensity": eu_avg_emissions,
        "iron_market_price_usd_per_t": iron_market_price,
        "global_iron_weighted_avg_cost_usd_per_t": global_iron_weighted_avg_cost,
        "eu_vs_market_price_margin": eu_vs_market_price_margin,
        "eu_cost_to_market_price_ratio": eu_cost_to_market_price_ratio,
        "eu_vs_global_iron_cost_gap": eu_vs_global_iron_cost_gap,
        "global_international_trade_mt": global_international_trade_mt,
        "global_trade_to_eu_production_ratio": global_trade_to_eu_production_ratio,
    }


def summarise_run_year(run: RunArtifacts, year: int, commodity: str) -> dict[str, object] | None:
    if commodity == STEEL:
        return summarise_run_year_steel(run, year)
    if commodity == IRON:
        return summarise_run_year_iron(run, year)
    raise ValueError(f"Unsupported commodity: {commodity}")


def build_summary(results_dir: Path, years: Iterable[int], commodity: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for run in discover_runs(results_dir):
        for year in years:
            record = summarise_run_year(run, year, commodity)
            if record is not None:
                records.append(record)
    if not records:
        raise ValueError(f"No competitiveness summary records could be created for {commodity}")
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["master_input_name", "year", "scenario_name"])
        .reset_index(drop=True)
    )


def _save_heatmap(data: pd.DataFrame, value_column: str, title: str, output_path: Path, fmt: str = ".2f") -> None:
    pivot = data.pivot(index="master_input_name", columns="year", values=value_column).sort_index()
    fig_width = max(8, 1.6 * max(len(pivot.columns), 1))
    fig_height = max(4.5, 0.6 * max(len(pivot.index), 1) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(column) for column in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    ax.set_title(title)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if math.isnan(value):
                continue
            ax.text(column_index, row_index, format(value, fmt), ha="center", va="center", color="white", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.set_ylabel(value_column.replace("_", " "))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_trade_lines(data: pd.DataFrame, output_path: Path, commodity: str) -> None:
    scenarios = list(dict.fromkeys(data["master_input_name"].tolist()))
    columns = 2
    rows = math.ceil(len(scenarios) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, max(4.5 * rows, 4.5)), dpi=180, squeeze=False)
    for axis, scenario in zip(axes.flat, scenarios):
        subset = data[data["master_input_name"] == scenario].sort_values("year")
        if commodity == STEEL:
            axis.plot(subset["year"], subset["imports_mt"], marker="o", label="Imports")
            axis.plot(subset["year"], subset["exports_mt"], marker="o", label="Exports")
            axis.plot(subset["year"], subset["net_imports_mt"], marker="o", label="Net imports")
            axis.set_ylabel("Mt steel")
        else:
            axis.plot(
                subset["year"],
                subset["global_international_trade_mt"],
                marker="o",
                label="Global international iron trade",
            )
            axis.plot(subset["year"], subset["europe_production_mt"], marker="o", label="EU iron production")
            axis.axhline(0.0, color="#666666", linewidth=0.8)
            axis.set_ylabel("Mt iron")
        axis.set_title(scenario)
        axis.set_xlabel("Year")
        axis.grid(alpha=0.3)
    for axis in axes.flat[len(scenarios) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1), frameon=False)
    fig.suptitle(
        "Europe steel trade position by scenario"
        if commodity == STEEL
        else "Europe iron production and global trade by scenario",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_scatter(data: pd.DataFrame, output_path: Path, commodity: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    year_colors = {2030: "#4c72b0", 2035: "#55a868", 2040: "#c44e52"}
    x_column = "import_share_of_demand" if commodity == STEEL else "eu_vs_market_price_margin"
    x_label = "Import share of European demand" if commodity == STEEL else "EU iron market margin (price - cost)"
    for year, subset in data.groupby("year"):
        ax.scatter(
            subset[x_column],
            subset["eu_avg_emissions_intensity"],
            s=(subset["eu_avg_unit_subsidy"].fillna(0.0) + 5.0) * 8.0,
            alpha=0.8,
            color=year_colors.get(int(year), "#8172b2"),
            label=str(year),
        )
        for row in subset.itertuples(index=False):
            ax.annotate(row.master_input_name, (getattr(row, x_column), row.eu_avg_emissions_intensity), fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("EU emissions intensity")
    ax.set_title("Competitiveness and decarbonisation frontier")
    ax.grid(alpha=0.3)
    ax.legend(title="Year")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_summary_markdown(
    summary: pd.DataFrame, selected_years: pd.DataFrame, output_path: Path, commodity: str
) -> None:
    if commodity == STEEL:
        lines = [
            "# Europe Steel Competitiveness Summary",
            "",
            "This folder compares the scenario runs in `cluster/results` from a European steel competitiveness perspective.",
            "",
            "Key metrics:",
            "- `self_sufficiency_ratio`: European steel production divided by steel delivered to European destinations.",
            "- `import_share_of_demand`: Share of European steel demand covered by imports from outside Europe.",
            "- `eu_avg_unit_production_cost`: Production-weighted European steel cost from the post-processed furnace table.",
            "- `eu_vs_import_cost_gap`: European weighted production cost minus imported source cost where the source could be joined back to a steel furnace row.",
            "- `imported_source_cost_coverage`: Share of import volume covered by that source-cost join.",
            "",
        ]
    else:
        lines = [
            "# Europe Iron Competitiveness Summary",
            "",
            "This folder compares the scenario runs in `cluster/results` from a European iron competitiveness perspective.",
            "",
            "Caveat:",
            "- The current result exports do not include Europe-specific iron import/export flow tables comparable to steel.",
            "- This iron view therefore uses European iron production economics plus two global proxies: iron market prices and total international iron trade volume.",
            "",
            "Key metrics:",
            "- `eu_vs_market_price_margin`: Iron market price minus Europe's weighted iron production cost; higher is better for competitiveness.",
            "- `eu_vs_global_iron_cost_gap`: Europe's weighted iron production cost minus the run's global weighted-average iron cost.",
            "- `global_trade_to_eu_production_ratio`: Total international iron trade volume divided by European iron production.",
            "",
        ]
    for year in sorted(selected_years["year"].unique()):
        subset = selected_years[selected_years["year"] == year].copy()
        if subset.empty:
            continue
        lowest_cost = subset.sort_values("eu_avg_unit_production_cost").iloc[0]
        if commodity == STEEL:
            lowest_import = subset.sort_values("import_share_of_demand").iloc[0]
            highest_self = subset.sort_values("self_sufficiency_ratio", ascending=False).iloc[0]
            lines.extend(
                [
                    f"## {year}",
                    "",
                    f"- Lowest import share: **{lowest_import['master_input_name']}** at {lowest_import['import_share_of_demand']:.1%}",
                    f"- Highest self-sufficiency: **{highest_self['master_input_name']}** at {highest_self['self_sufficiency_ratio']:.2f}",
                    f"- Lowest EU production cost: **{lowest_cost['master_input_name']}** at {lowest_cost['eu_avg_unit_production_cost']:.1f}",
                    "",
                ]
            )
        else:
            highest_margin = subset.sort_values("eu_vs_market_price_margin", ascending=False).iloc[0]
            lowest_gap = subset.sort_values("eu_vs_global_iron_cost_gap").iloc[0]
            lines.extend(
                [
                    f"## {year}",
                    "",
                    f"- Highest iron market margin: **{highest_margin['master_input_name']}** at {highest_margin['eu_vs_market_price_margin']:.1f}",
                    f"- Lowest EU-vs-global cost gap: **{lowest_gap['master_input_name']}** at {lowest_gap['eu_vs_global_iron_cost_gap']:.1f}",
                    f"- Lowest EU production cost: **{lowest_cost['master_input_name']}** at {lowest_cost['eu_avg_unit_production_cost']:.1f}",
                    "",
                ]
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(summary: pd.DataFrame, output_dir: Path, selected_years: Iterable[int], commodity: str) -> list[Path]:
    selected_years = tuple(selected_years)
    selected = summary[summary["year"].isin(selected_years)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"europe_{commodity}_competitiveness"
    yearly_csv = output_dir / f"{prefix}_yearly_summary.csv"
    selected_csv = output_dir / f"{prefix}_selected_years.csv"
    summary_md = output_dir / "README.md"
    trade_lines_png = output_dir / (
        "europe_trade_balance_by_scenario.png" if commodity == STEEL else "iron_trade_and_production_by_scenario.png"
    )
    primary_heatmap_png = output_dir / (
        "import_share_heatmap.png" if commodity == STEEL else "iron_market_margin_heatmap.png"
    )
    secondary_heatmap_png = output_dir / (
        "self_sufficiency_heatmap.png" if commodity == STEEL else "global_trade_to_eu_production_heatmap.png"
    )
    cost_heatmap_png = output_dir / "eu_production_cost_heatmap.png"
    cost_gap_png = output_dir / (
        "eu_vs_import_cost_gap_heatmap.png" if commodity == STEEL else "eu_vs_global_iron_cost_gap_heatmap.png"
    )
    emissions_scatter_png = output_dir / "competitiveness_frontier_scatter.png"

    summary.to_csv(yearly_csv, index=False)
    selected.to_csv(selected_csv, index=False)
    _write_summary_markdown(summary, selected, summary_md, commodity)
    _save_trade_lines(summary, trade_lines_png, commodity)
    if commodity == STEEL:
        _save_heatmap(
            selected, "import_share_of_demand", "Import share of European demand", primary_heatmap_png, fmt=".1%"
        )
        _save_heatmap(
            selected, "self_sufficiency_ratio", "European self-sufficiency ratio", secondary_heatmap_png, fmt=".2f"
        )
        _save_heatmap(
            selected, "eu_avg_unit_production_cost", "EU weighted steel production cost", cost_heatmap_png, fmt=".1f"
        )
        _save_heatmap(selected, "eu_vs_import_cost_gap", "EU cost gap vs imported source cost", cost_gap_png, fmt=".1f")
    else:
        _save_heatmap(
            selected,
            "eu_vs_market_price_margin",
            "EU iron market margin (price - cost)",
            primary_heatmap_png,
            fmt=".1f",
        )
        _save_heatmap(
            selected,
            "global_trade_to_eu_production_ratio",
            "Global international iron trade / EU iron production",
            secondary_heatmap_png,
            fmt=".2f",
        )
        _save_heatmap(
            selected, "eu_avg_unit_production_cost", "EU weighted iron production cost", cost_heatmap_png, fmt=".1f"
        )
        _save_heatmap(
            selected, "eu_vs_global_iron_cost_gap", "EU cost gap vs global weighted iron cost", cost_gap_png, fmt=".1f"
        )
    _save_scatter(selected, emissions_scatter_png, commodity)

    return [
        yearly_csv,
        selected_csv,
        summary_md,
        trade_lines_png,
        primary_heatmap_png,
        secondary_heatmap_png,
        cost_heatmap_png,
        cost_gap_png,
        emissions_scatter_png,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Europe competitiveness scenario summary from cluster results."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path("cluster/results"),
        help="Directory containing *_master_input_* run folders (default: cluster/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cluster/results/europe_competitiveness_summary"),
        help="Directory where summary outputs are written",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="Years to include in the summary (default: 2030 2035 2040)",
    )
    parser.add_argument(
        "--commodity",
        choices=[STEEL, IRON],
        default=STEEL,
        help="Commodity summary to generate (default: steel)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    if args.commodity != STEEL and output_dir.name == "europe_competitiveness_summary":
        output_dir = output_dir.with_name(f"europe_{args.commodity}_competitiveness_summary")
    summary = build_summary(args.results_dir.resolve(), args.years, args.commodity)
    created = write_outputs(summary, output_dir, args.years, args.commodity)
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
