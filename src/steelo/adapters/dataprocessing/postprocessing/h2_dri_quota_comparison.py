"""Compare forced European H2-DRI quota runs with the vanilla merit order."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
import geopandas as gpd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

from steelo.adapters.dataprocessing.postprocessing.regional_trade_analysis import _resolve_shapefile


DEFAULT_YEARS = tuple(range(2030, 2061, 5))


def _country_name(iso3: object) -> str:
    code = str(iso3).upper()
    country = pycountry.countries.get(alpha_3=code)
    return country.name if country is not None else code


def _capital_recovery_factor(rate: float, lifetime: int) -> float:
    if not math.isfinite(rate) or rate <= 0.0:
        return 1.0 / lifetime
    return rate * (1.0 + rate) ** lifetime / ((1.0 + rate) ** lifetime - 1.0)


def _latest_simulation_dir(container: Path) -> Path:
    if (container / "simulation_config.json").is_file():
        return container
    candidates = sorted(
        (path for path in container.glob("sim_*") if (path / "simulation_config.json").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No completed simulation directory below {container}")
    return candidates[0]


def _find_vanilla_run(results_dir: Path) -> Path:
    candidates = [path for path in results_dir.iterdir() if path.is_dir() and not path.name.startswith("h2dri_")]
    merit_candidates = [path for path in candidates if (path / "data" / "h2_dri_merit_order.csv").is_file()]
    if len(merit_candidates) != 1:
        raise ValueError(
            f"Expected exactly one vanilla run containing data/h2_dri_merit_order.csv; found {len(merit_candidates)}"
        )
    return merit_candidates[0]


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_ex_post_curve(
    vanilla_run: Path,
    years: tuple[int, ...],
    candidate_scope: str = "eu",
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = _load_json(vanilla_run / "simulation_config.json")
    lifetime = int(config["plant_lifetime"])
    iron_per_steel = float(config["h2_dri_quota"]["iron_per_steel"])
    merit = pd.read_csv(vanilla_run / "data" / "h2_dri_merit_order.csv")
    scopes = {
        "eu": {"EU+EFTA+UK"},
        "global": {"EU+EFTA+UK", "Non-EU+EFTA+UK"},
    }
    if candidate_scope not in scopes:
        raise ValueError(f"Unknown candidate scope {candidate_scope!r}; expected 'eu' or 'global'")
    merit = merit[merit["target_year"].isin(years) & merit["scope"].isin(scopes[candidate_scope])].copy()
    if merit.empty:
        raise ValueError(f"No {candidate_scope} merit-order rows for the requested target years")

    numeric = [
        "capacity_counted_t",
        "utilization_rate",
        "cost_of_equity",
        "indicative_policy_support_npv_usd",
        "steel_demand_t",
    ]
    merit[numeric] = merit[numeric].apply(pd.to_numeric, errors="coerce")
    merit["country"] = merit["iso3"].map(_country_name)
    merit["annual_iron_output_t"] = merit["capacity_counted_t"] * merit["utilization_rate"]
    merit["annual_steel_equivalent_t"] = merit["annual_iron_output_t"] / iron_per_steel
    merit["crf"] = merit["cost_of_equity"].map(lambda rate: _capital_recovery_factor(float(rate), lifetime))
    merit["annualized_support_usd"] = merit["indicative_policy_support_npv_usd"] * merit["crf"]
    merit["levelized_support_usd_per_t_steel"] = np.divide(
        merit["annualized_support_usd"],
        merit["annual_steel_equivalent_t"],
        out=np.zeros(len(merit), dtype=float),
        where=merit["annual_steel_equivalent_t"].gt(0),
    )
    market_files = sorted((vanilla_run / "data").glob("market_prices_*.csv"))
    if not market_files:
        raise FileNotFoundError(f"No market_prices_*.csv found in {vanilla_run / 'data'}")
    market = pd.read_csv(market_files[-1]).set_index("year")
    steel_price = pd.to_numeric(market["steel_price_usd_per_t"], errors="coerce").to_dict()
    iron_price = pd.to_numeric(market["iron_price_usd_per_t"], errors="coerce").to_dict()
    merit["reference_steel_price_usd_per_t"] = merit["target_year"].map(steel_price)
    merit["reference_iron_price_usd_per_t"] = merit["target_year"].map(iron_price)
    merit["levelized_support_usd_per_t_iron"] = np.divide(
        merit["annualized_support_usd"],
        merit["annual_iron_output_t"],
        out=np.zeros(len(merit), dtype=float),
        where=merit["annual_iron_output_t"].gt(0),
    )
    merit["required_iron_price_usd_per_t"] = (
        merit["reference_iron_price_usd_per_t"] + merit["levelized_support_usd_per_t_iron"]
    )
    merit["iron_price_increase_pct"] = (
        merit["levelized_support_usd_per_t_iron"] / merit["reference_iron_price_usd_per_t"] * 100.0
    )
    merit["required_steel_price_usd_per_t"] = (
        merit["reference_steel_price_usd_per_t"] + merit["levelized_support_usd_per_t_steel"]
    )
    merit["steel_price_increase_pct"] = (
        merit["levelized_support_usd_per_t_steel"] / merit["reference_steel_price_usd_per_t"] * 100.0
    )
    merit["npv_rank"] = merit["rank"]
    # Raw NPV rank can differ after annualizing with site-specific discount
    # rates and output. Sort by the metric displayed on the marginal curve.
    merit = merit.sort_values(["target_year", "levelized_support_usd_per_t_steel", "iso3"]).reset_index(drop=True)
    merit["merit_order_rank"] = merit.groupby("target_year").cumcount() + 1
    merit["cumulative_steel_equivalent_t"] = merit.groupby("target_year")["annual_steel_equivalent_t"].cumsum()
    merit["cumulative_iron_output_t"] = merit.groupby("target_year")["annual_iron_output_t"].cumsum()
    merit["cumulative_annualized_support_usd"] = merit.groupby("target_year")["annualized_support_usd"].cumsum()
    merit["average_support_usd_per_t_steel"] = (
        merit["cumulative_annualized_support_usd"] / merit["cumulative_steel_equivalent_t"]
    )
    merit["average_required_steel_price_usd_per_t"] = (
        merit["reference_steel_price_usd_per_t"] + merit["average_support_usd_per_t_steel"]
    )
    merit["average_steel_price_increase_pct"] = (
        merit["average_support_usd_per_t_steel"] / merit["reference_steel_price_usd_per_t"] * 100.0
    )
    merit["average_support_usd_per_t_iron"] = (
        merit["cumulative_annualized_support_usd"] / merit["cumulative_iron_output_t"]
    )
    merit["average_required_iron_price_usd_per_t"] = (
        merit["reference_iron_price_usd_per_t"] + merit["average_support_usd_per_t_iron"]
    )
    merit["average_iron_price_increase_pct"] = (
        merit["average_support_usd_per_t_iron"] / merit["reference_iron_price_usd_per_t"] * 100.0
    )
    merit["cumulative_steel_demand_share"] = merit["cumulative_steel_equivalent_t"] / merit["steel_demand_t"]
    return merit, config


def load_forced_runs(
    results_dir: Path,
    years: tuple[int, ...],
    vanilla_merit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    merit_lookup = vanilla_merit.set_index(["target_year", "iso3"]).sort_index()
    reference_demand = vanilla_merit.groupby("target_year")["steel_demand_t"].first().to_dict()
    for container in sorted(results_dir.glob("h2dri_*")):
        run_dir = _latest_simulation_dir(container)
        config = _load_json(run_dir / "simulation_config.json")
        quota = config["h2_dri_quota"]
        candidate_scope = str(quota.get("candidate_scope", "eu"))
        share = float(quota["demand_share"])
        lifetime = int(config["plant_lifetime"])
        utilization = float(config["capacity_limit"])
        iron_per_steel = float(quota["iron_per_steel"])
        project_path = run_dir / "data" / "h2_dri_quota" / "forced_projects.csv"
        summary_path = run_dir / "data" / "h2_dri_quota" / "eu_h2_dri_quota_summary.csv"
        if not project_path.is_file() or not summary_path.is_file():
            continue
        projects = pd.read_csv(project_path)
        annual = pd.read_csv(summary_path).set_index("year")
        invalid_years = annual.index[pd.to_numeric(annual["eu_steel_demand_t"], errors="coerce").le(0)]
        first_invalid_trade_year = int(min(invalid_years)) if len(invalid_years) else math.nan
        projects["scenario"] = container.name
        projects["candidate_scope"] = candidate_scope
        projects["quota_share"] = share
        projects["country"] = projects["iso3"].map(_country_name)
        projects["annual_iron_output_t"] = projects["capacity_t"] * utilization
        projects["annual_steel_equivalent_t"] = projects["capacity_t"] * utilization / iron_per_steel

        matched_support: list[float] = []
        matched_rate: list[float] = []
        for row in projects.itertuples(index=False):
            key = (int(row.quota_year), str(row.iso3))
            if key not in merit_lookup.index:
                matched_support.append(math.nan)
                matched_rate.append(math.nan)
                continue
            match = merit_lookup.loc[key]
            if isinstance(match, pd.DataFrame):
                match = match.iloc[0]
            matched_support.append(float(match["indicative_policy_support_npv_usd"]))
            matched_rate.append(float(match["cost_of_equity"]))
        projects["vanilla_h2_support_npv_usd"] = matched_support
        projects["discount_rate"] = matched_rate
        fallback_rate = pd.to_numeric(vanilla_merit["cost_of_equity"], errors="coerce").median()
        projects["discount_rate"] = projects["discount_rate"].fillna(fallback_rate)
        projects["crf"] = projects["discount_rate"].map(lambda rate: _capital_recovery_factor(float(rate), lifetime))
        projects["recorded_annualized_support_usd"] = projects["indicative_policy_support_npv_usd"] * projects["crf"]
        projects["vanilla_h2_annualized_support_usd"] = projects["vanilla_h2_support_npv_usd"] * projects["crf"]
        project_frames.append(projects)

        for year in years:
            active = projects[projects["quota_year"].le(year) & projects["quota_year"].add(lifetime).gt(year)].copy()
            reported_demand = float(annual.loc[year, "eu_steel_demand_t"]) if year in annual.index else math.nan
            actual_h2_dri_cost = (
                float(annual.loc[year, "h2_dri_unit_production_cost_usd_t"]) if year in annual.index else math.nan
            )
            actual_h2_dri_production = math.nan
            actual_h2_dri_delivered = math.nan
            if year in annual.index:
                production_column = (
                    "quota_scope_h2_dri_production_t"
                    if "quota_scope_h2_dri_production_t" in annual.columns
                    else "eu_h2_dri_production_t"
                )
                delivered_column = (
                    "quota_h2_dri_delivered_to_eu_eaf_t"
                    if "quota_h2_dri_delivered_to_eu_eaf_t" in annual.columns
                    else "eu_h2_dri_delivered_to_eu_steel_t"
                )
                actual_h2_dri_production = float(annual.loc[year, production_column])
                actual_h2_dri_delivered = float(annual.loc[year, delivered_column])
            actual_iron_market_price = (
                float(annual.loc[year, "market_iron_price_usd_t"]) if year in annual.index else math.nan
            )
            steel_demand = float(reference_demand.get(year, math.nan))
            run_valid = math.isfinite(reported_demand) and reported_demand > 0.0
            capacity = float(active["annual_steel_equivalent_t"].sum())
            iron_output = float(active["annual_iron_output_t"].sum())
            recorded_annual = float(active["recorded_annualized_support_usd"].sum())
            revalued_annual = float(active["vanilla_h2_annualized_support_usd"].sum())
            current_mix_annual = 0.0
            current_mix_complete = True
            for project in active.itertuples(index=False):
                current_key = (year, str(project.iso3))
                if current_key not in merit_lookup.index:
                    current_mix_complete = False
                    continue
                current_match = merit_lookup.loc[current_key]
                if isinstance(current_match, pd.DataFrame):
                    current_match = current_match.iloc[0]
                scale = float(project.capacity_t) / float(current_match["capacity_t"])
                current_mix_annual += float(current_match["annualized_support_usd"]) * scale
            if not current_mix_complete:
                current_mix_annual = math.nan
            summaries.append(
                {
                    "year": year,
                    "scenario": container.name,
                    "candidate_scope": candidate_scope,
                    "quota_share": share,
                    "reported_eu_steel_demand_t": reported_demand,
                    "reference_eu_steel_demand_t": steel_demand,
                    "actual_h2_dri_unit_production_cost_usd_per_t": actual_h2_dri_cost,
                    "actual_h2_dri_production_t": actual_h2_dri_production,
                    "actual_h2_dri_delivered_to_eu_steel_t": actual_h2_dri_delivered,
                    "actual_iron_market_price_usd_per_t": actual_iron_market_price,
                    "actual_downstream_steel_unit_production_cost_usd_per_t": (
                        float(annual.loc[year, "h2_dri_downstream_steel_unit_production_cost_usd_t"])
                        if year in annual.index
                        else math.nan
                    ),
                    "actual_eu_steel_unit_production_cost_usd_per_t": (
                        float(annual.loc[year, "eu_steel_unit_production_cost_usd_t"])
                        if year in annual.index
                        else math.nan
                    ),
                    "actual_world_steel_unit_production_cost_usd_per_t": (
                        float(annual.loc[year, "world_steel_unit_production_cost_usd_t"])
                        if year in annual.index
                        else math.nan
                    ),
                    "actual_steel_market_price_usd_per_t": (
                        float(annual.loc[year, "market_steel_price_usd_t"]) if year in annual.index else math.nan
                    ),
                    "forced_run_valid": run_valid,
                    "first_invalid_trade_year": first_invalid_trade_year,
                    "active_forced_projects": len(active),
                    "active_steel_equivalent_t": capacity,
                    "active_iron_output_t": iron_output,
                    "active_share_of_demand": capacity / steel_demand if steel_demand > 0 else math.nan,
                    "recorded_support_npv_usd": float(active["indicative_policy_support_npv_usd"].sum()),
                    "recorded_annualized_support_usd": recorded_annual,
                    "recorded_support_usd_per_t_steel": recorded_annual / capacity if capacity > 0 else math.nan,
                    "recorded_support_usd_per_t_iron": (recorded_annual / iron_output if iron_output > 0 else math.nan),
                    "vanilla_h2_revalued_support_npv_usd": float(active["vanilla_h2_support_npv_usd"].sum()),
                    "vanilla_h2_revalued_annualized_support_usd": revalued_annual,
                    "vanilla_h2_revalued_support_usd_per_t_steel": (
                        revalued_annual / capacity if capacity > 0 else math.nan
                    ),
                    "vanilla_h2_revalued_support_usd_per_t_iron": (
                        revalued_annual / iron_output if iron_output > 0 else math.nan
                    ),
                    "current_year_h2_forced_mix_annualized_support_usd": current_mix_annual,
                    "current_year_h2_forced_mix_support_usd_per_t_steel": (
                        current_mix_annual / capacity if capacity > 0 else math.nan
                    ),
                    "current_year_h2_forced_mix_support_usd_per_t_iron": (
                        current_mix_annual / iron_output if iron_output > 0 else math.nan
                    ),
                    "countries": ", ".join(active["iso3"].astype(str)),
                }
            )
    if not project_frames:
        raise FileNotFoundError(f"No forced-project quota outputs found below {results_dir}")
    return pd.concat(project_frames, ignore_index=True), pd.DataFrame(summaries)


def load_forced_project_locations(results_dir: Path, years: tuple[int, ...]) -> pd.DataFrame:
    """Load realised annual location and cost data for projects selected by the quota."""

    frames: list[pd.DataFrame] = []
    columns = [
        "year",
        "furnace_group_id",
        "iso3",
        "latitude",
        "longitude",
        "production",
        "unit_production_cost",
        "technology",
        "chosen_reductant",
    ]
    for container in sorted(results_dir.glob("h2dri_*")):
        run_dir = _latest_simulation_dir(container)
        project_path = run_dir / "data" / "h2_dri_quota" / "forced_projects.csv"
        post_processed = sorted(run_dir.glob("post_processed_*.csv"), key=lambda path: path.stat().st_mtime)
        if not project_path.is_file() or not post_processed:
            continue
        config = _load_json(run_dir / "simulation_config.json")
        share = float(config["h2_dri_quota"]["demand_share"])
        project_ids = set(pd.read_csv(project_path)["furnace_group_id"].astype(str))
        annual = pd.read_csv(post_processed[-1], usecols=lambda column: column in columns, low_memory=False)
        if missing := set(columns) - set(annual.columns):
            raise ValueError(
                f"{post_processed[-1]} is missing forced-project map columns: {', '.join(sorted(missing))}"
            )
        annual["furnace_group_id"] = annual["furnace_group_id"].astype(str)
        annual["year"] = pd.to_numeric(annual["year"], errors="coerce")
        annual["production"] = pd.to_numeric(annual["production"], errors="coerce")
        annual["unit_production_cost"] = pd.to_numeric(annual["unit_production_cost"], errors="coerce")
        annual["latitude"] = pd.to_numeric(annual["latitude"], errors="coerce")
        annual["longitude"] = pd.to_numeric(annual["longitude"], errors="coerce")
        annual = annual[
            annual["year"].isin(years)
            & annual["furnace_group_id"].isin(project_ids)
            & annual["technology"].astype(str).str.casefold().eq("dri")
            & annual["chosen_reductant"].astype(str).str.casefold().eq("hydrogen")
            & annual["production"].gt(0)
        ].drop_duplicates(["year", "furnace_group_id"])
        annual["scenario"] = container.name
        annual["quota_share"] = share
        annual["method"] = "Forced"
        annual = annual.rename(
            columns={
                "production": "annual_output_t",
                "unit_production_cost": "cost_usd_per_t",
            }
        )
        frames.append(annual)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_location_comparison(
    merit: pd.DataFrame,
    forced_locations: pd.DataFrame,
    comparison: pd.DataFrame,
    years: tuple[int, ...],
) -> pd.DataFrame:
    """Match each valid forced fleet with an ex-post portfolio at the same output."""

    records: list[pd.DataFrame] = []
    for point in comparison[comparison["year"].isin(years) & comparison["forced_run_valid"].astype(bool)].itertuples(
        index=False
    ):
        forced = forced_locations[
            forced_locations["scenario"].eq(point.scenario) & forced_locations["year"].eq(point.year)
        ].copy()
        if forced.empty:
            continue
        forced["represented_output_t"] = forced["annual_output_t"]
        forced["required_price_usd_per_t"] = np.nan
        records.append(forced)

        remaining = float(forced["annual_output_t"].sum())
        selected_rows: list[dict[str, object]] = []
        year_merit = merit[merit["target_year"].eq(point.year)].sort_values(
            ["levelized_support_usd_per_t_iron", "iso3"]
        )
        for row in year_merit.itertuples(index=False):
            if remaining <= 0.0:
                break
            available = float(row.annual_iron_output_t)
            represented = min(remaining, available)
            selected_rows.append(
                {
                    "year": int(point.year),
                    "furnace_group_id": "",
                    "iso3": row.iso3,
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "annual_output_t": available,
                    "cost_usd_per_t": np.nan,
                    "technology": "DRI",
                    "chosen_reductant": "hydrogen",
                    "scenario": point.scenario,
                    "quota_share": point.quota_share,
                    "method": "Ex-post",
                    "represented_output_t": represented,
                    "required_price_usd_per_t": row.required_iron_price_usd_per_t,
                }
            )
            remaining -= represented
        if selected_rows:
            records.append(pd.DataFrame(selected_rows))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def plot_cost_curves(merit: pd.DataFrame, forced: pd.DataFrame, years: tuple[int, ...], output: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(18, 25), sharey=False)
    axes_flat = axes.flatten()
    for ax, year in zip(axes_flat, years, strict=False):
        year_merit = merit[merit["target_year"].eq(year)].copy()
        if year_merit.empty:
            ax.axis("off")
            continue
        left = 0.0
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        for row in year_merit.itertuples(index=False):
            width = float(row.annual_steel_equivalent_t) / 1e6
            height = float(row.levelized_support_usd_per_t_steel)
            ax.bar(left, height, width=width, align="edge", color="#9ecae1", edgecolor="white", linewidth=0.7)
            tick_positions.append(left + width / 2)
            tick_labels.append(f"{row.country}\n{width:.1f} Mt")
            left += width
        average_x = np.r_[0.0, year_merit["cumulative_steel_equivalent_t"].to_numpy() / 1e6]
        average_y = np.r_[
            year_merit.iloc[0]["average_support_usd_per_t_steel"],
            year_merit["average_support_usd_per_t_steel"].to_numpy(),
        ]
        ax.plot(
            average_x,
            average_y,
            color="#08519c",
            linewidth=2.0,
            label="Ex-post cumulative average",
        )

        year_forced = forced[forced["year"].eq(year) & forced["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        ax.scatter(
            year_forced["active_steel_equivalent_t"] / 1e6,
            year_forced["recorded_support_usd_per_t_steel"],
            color="#e6550d",
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label="Forced runs: recorded NPV gap",
        )
        for point in year_forced.itertuples(index=False):
            ax.annotate(
                f"{point.quota_share:.0%}",
                (point.active_steel_equivalent_t / 1e6, point.recorded_support_usd_per_t_steel),
                xytext=(3, -10),
                textcoords="offset points",
                fontsize=7,
            )
        invalid = forced[forced["year"].eq(year) & ~forced["forced_run_valid"].astype(bool)].sort_values("quota_share")
        ax.scatter(
            invalid["active_steel_equivalent_t"] / 1e6,
            np.zeros(len(invalid)),
            color="#b2182b",
            marker="X",
            s=48,
            zorder=6,
            label="Forced run invalid/empty trade output",
        )
        for point in invalid.itertuples(index=False):
            ax.annotate(
                f"{point.quota_share:.0%} invalid",
                (point.active_steel_equivalent_t / 1e6, 0.0),
                xytext=(3, 5),
                textcoords="offset points",
                fontsize=7,
                color="#8c0d1a",
                rotation=45,
            )
        ax.set_title(str(year))
        ax.set_xlabel("Annual steel-equivalent H₂-DRI output [Mt steel/a]")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=6)
        ax.set_ylabel("Levelized NPV gap [USD/t steel]")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
    for ax in axes_flat[len(years) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "EU+EFTA+UK H₂-DRI support merit order: vanilla ex-post curve vs forced runs\n"
        "Country blocks use hydrogen-locked vanilla NPVs; orange points use NPVs recorded by quota runs",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=4.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_iron_cost_curves(
    merit: pd.DataFrame,
    forced: pd.DataFrame,
    years: tuple[int, ...],
    output: Path,
) -> None:
    """Plot the support merit order per physical tonne of H2-DRI output."""

    fig, axes = plt.subplots(4, 2, figsize=(18, 25), sharey=False)
    axes_flat = axes.flatten()
    for ax, year in zip(axes_flat, years, strict=False):
        year_merit = merit[merit["target_year"].eq(year)].copy()
        if year_merit.empty:
            ax.axis("off")
            continue
        left = 0.0
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        for row in year_merit.itertuples(index=False):
            width = float(row.annual_iron_output_t) / 1e6
            height = float(row.levelized_support_usd_per_t_iron)
            ax.bar(left, height, width=width, align="edge", color="#9ecae1", edgecolor="white", linewidth=0.7)
            tick_positions.append(left + width / 2)
            tick_labels.append(f"{row.country}\n{width:.1f} Mt")
            left += width
        ax.plot(
            np.r_[0.0, year_merit["cumulative_iron_output_t"].to_numpy() / 1e6],
            np.r_[
                year_merit.iloc[0]["average_support_usd_per_t_iron"],
                year_merit["average_support_usd_per_t_iron"].to_numpy(),
            ],
            color="#08519c",
            linewidth=2.0,
            label="Ex-post cumulative average",
        )
        valid = forced[forced["year"].eq(year) & forced["forced_run_valid"].astype(bool)].sort_values("quota_share")
        ax.scatter(
            valid["active_iron_output_t"] / 1e6,
            valid["recorded_support_usd_per_t_iron"],
            color="#e6550d",
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label="Forced runs: recorded NPV gap",
        )
        for point in valid.itertuples(index=False):
            ax.annotate(
                f"{point.quota_share:.0%}",
                (point.active_iron_output_t / 1e6, point.recorded_support_usd_per_t_iron),
                xytext=(3, -10),
                textcoords="offset points",
                fontsize=7,
            )
        invalid = forced[forced["year"].eq(year) & ~forced["forced_run_valid"].astype(bool)].sort_values("quota_share")
        ax.scatter(
            invalid["active_iron_output_t"] / 1e6,
            np.zeros(len(invalid)),
            color="#b2182b",
            marker="X",
            s=48,
            zorder=6,
            label="Forced run invalid/empty trade output",
        )
        for point in invalid.itertuples(index=False):
            ax.annotate(
                f"{point.quota_share:.0%} invalid",
                (point.active_iron_output_t / 1e6, 0.0),
                xytext=(3, 5),
                textcoords="offset points",
                fontsize=7,
                color="#8c0d1a",
                rotation=45,
            )
        ax.set_title(str(year))
        ax.set_xlabel("Annual H₂-DRI output [Mt iron/a]")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=6)
        ax.set_ylabel("Levelized NPV gap [USD/t H₂-DRI]")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
    for ax in list(axes_flat)[len(years) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "EU+EFTA+UK H₂-DRI support merit order: physical iron output\n"
        "Country blocks use hydrogen-locked vanilla NPVs; orange points use NPVs recorded by quota runs",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=4.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _support_at_capacity(
    year_merit: pd.DataFrame,
    capacity_t: float,
    volume_column: str = "annual_steel_equivalent_t",
) -> tuple[float, float, float]:
    remaining = capacity_t
    npv = 0.0
    annual = 0.0
    for row in year_merit.itertuples(index=False):
        if remaining <= 0:
            break
        available = float(getattr(row, volume_column))
        take = min(remaining, available)
        fraction = take / available
        npv += fraction * float(row.indicative_policy_support_npv_usd)
        annual += fraction * float(row.annualized_support_usd)
        remaining -= take
    covered = capacity_t - max(0.0, remaining)
    return npv, annual, covered


def build_comparison_table(merit: pd.DataFrame, forced: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for point in forced.itertuples(index=False):
        year_merit = merit[merit["target_year"].eq(point.year)]
        ex_npv, ex_annual, covered = _support_at_capacity(year_merit, float(point.active_steel_equivalent_t))
        iron_ex_npv, iron_ex_annual, iron_covered = _support_at_capacity(
            year_merit,
            float(point.active_iron_output_t),
            "annual_iron_output_t",
        )
        actual_iron_npv, actual_iron_annual, actual_iron_covered = _support_at_capacity(
            year_merit,
            float(point.actual_h2_dri_production_t),
            "annual_iron_output_t",
        )
        reference_price = (
            float(year_merit.iloc[0]["reference_steel_price_usd_per_t"]) if not year_merit.empty else math.nan
        )
        recorded_support = float(point.recorded_support_usd_per_t_steel)
        ex_post_support = ex_annual / covered if covered > 0 else math.nan
        recorded_iron_support = float(point.recorded_support_usd_per_t_iron)
        ex_post_iron_support = iron_ex_annual / iron_covered if iron_covered > 0 else math.nan
        revalued_iron_support = float(point.vanilla_h2_revalued_support_usd_per_t_iron)
        current_mix_iron_support = float(point.current_year_h2_forced_mix_support_usd_per_t_iron)
        revalued_steel_support = float(point.vanilla_h2_revalued_support_usd_per_t_steel)
        current_mix_steel_support = float(point.current_year_h2_forced_mix_support_usd_per_t_steel)
        reference_iron_price = (
            float(year_merit.iloc[0]["reference_iron_price_usd_per_t"]) if not year_merit.empty else math.nan
        )
        actual_ex_post_support = actual_iron_annual / actual_iron_covered if actual_iron_covered > 0 else math.nan
        actual_ex_post_break_even_price = reference_iron_price + actual_ex_post_support
        actual_h2_dri_cost = float(point.actual_h2_dri_unit_production_cost_usd_per_t)
        actual_iron_market_price = float(point.actual_iron_market_price_usd_per_t)
        actual_cost_support_gap = (
            max(0.0, actual_h2_dri_cost - actual_iron_market_price)
            if math.isfinite(actual_h2_dri_cost) and math.isfinite(actual_iron_market_price)
            else math.nan
        )
        rows.append(
            {
                **point._asdict(),
                "ex_post_support_npv_usd_at_forced_capacity": ex_npv,
                "ex_post_annualized_support_usd_at_forced_capacity": ex_annual,
                "ex_post_covered_steel_equivalent_t": covered,
                "ex_post_average_support_usd_per_t_steel": ex_post_support,
                "reference_steel_price_usd_per_t": reference_price,
                "recorded_required_steel_price_usd_per_t": reference_price + recorded_support,
                "recorded_steel_price_increase_pct": recorded_support / reference_price * 100.0,
                "ex_post_required_steel_price_usd_per_t": reference_price + ex_post_support,
                "ex_post_steel_price_increase_pct": ex_post_support / reference_price * 100.0,
                "ex_post_iron_support_npv_usd_at_forced_capacity": iron_ex_npv,
                "ex_post_iron_annualized_support_usd_at_forced_capacity": iron_ex_annual,
                "ex_post_covered_iron_output_t": iron_covered,
                "ex_post_average_support_usd_per_t_iron": ex_post_iron_support,
                "reference_iron_price_usd_per_t": reference_iron_price,
                "recorded_required_iron_price_usd_per_t": reference_iron_price + recorded_iron_support,
                "recorded_iron_price_increase_pct": recorded_iron_support / reference_iron_price * 100.0,
                "ex_post_required_iron_price_usd_per_t": reference_iron_price + ex_post_iron_support,
                "ex_post_iron_price_increase_pct": ex_post_iron_support / reference_iron_price * 100.0,
                "recorded_minus_ex_post_usd_per_t_steel": (
                    recorded_support - ex_post_support if covered > 0 and math.isfinite(recorded_support) else math.nan
                ),
                "recording_route_effect_usd_per_t_steel": recorded_support - revalued_steel_support,
                "vintage_market_effect_usd_per_t_steel": (revalued_steel_support - current_mix_steel_support),
                "selection_effect_usd_per_t_steel": current_mix_steel_support - ex_post_support,
                "recorded_minus_ex_post_usd_per_t_iron": (
                    recorded_iron_support - ex_post_iron_support
                    if iron_covered > 0 and math.isfinite(recorded_iron_support)
                    else math.nan
                ),
                "recording_route_effect_usd_per_t_iron": recorded_iron_support - revalued_iron_support,
                "vintage_market_effect_usd_per_t_iron": (revalued_iron_support - current_mix_iron_support),
                "selection_effect_usd_per_t_iron": current_mix_iron_support - ex_post_iron_support,
                "ex_post_support_npv_usd_at_actual_h2_dri_production": actual_iron_npv,
                "ex_post_annualized_support_usd_at_actual_h2_dri_production": actual_iron_annual,
                "ex_post_covered_actual_h2_dri_production_t": actual_iron_covered,
                "ex_post_average_support_at_actual_h2_dri_production_usd_per_t": actual_ex_post_support,
                "ex_post_break_even_price_at_actual_h2_dri_production_usd_per_t": (actual_ex_post_break_even_price),
                "actual_cost_minus_ex_post_break_even_usd_per_t": (
                    actual_h2_dri_cost - actual_ex_post_break_even_price
                    if math.isfinite(actual_h2_dri_cost) and actual_iron_covered > 0
                    else math.nan
                ),
                "actual_cost_support_gap_usd_per_t": actual_cost_support_gap,
                "actual_cost_support_total_production_usd_per_year": (
                    actual_cost_support_gap * float(point.actual_h2_dri_production_t)
                    if math.isfinite(actual_cost_support_gap)
                    else math.nan
                ),
                "actual_cost_support_total_delivered_usd_per_year": (
                    actual_cost_support_gap * float(point.actual_h2_dri_delivered_to_eu_steel_t)
                    if math.isfinite(actual_cost_support_gap)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_method_gap_decomposition(
    comparison: pd.DataFrame,
    years: tuple[int, ...],
    output: Path,
) -> None:
    """Show why recorded forced-run support differs from the ex-post merit order."""

    panels = (
        ("recorded_minus_ex_post_usd_per_t_iron", "Total: forced − ex-post"),
        ("recording_route_effect_usd_per_t_iron", "Stored valuation / route-lock effect"),
        ("vintage_market_effect_usd_per_t_iron", "Investment-vintage / market effect"),
        ("selection_effect_usd_per_t_iron", "Country-selection effect"),
    )
    valid = comparison[comparison["forced_run_valid"].astype(bool)].copy()
    shares = sorted(comparison["quota_share"].dropna().unique(), reverse=True)
    finite_values = np.concatenate([valid[column].dropna().to_numpy(dtype=float) for column, _ in panels])
    limit = max(1.0, float(np.nanmax(np.abs(finite_values))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    image = None
    for ax, (column, title) in zip(axes.flat, panels, strict=True):
        matrix = valid.pivot_table(index="quota_share", columns="year", values=column, aggfunc="first").reindex(
            index=shares, columns=years
        )
        image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(years)), labels=years)
        ax.set_yticks(range(len(shares)), labels=[f"{share:.0%}" for share in shares])
        ax.set_xlabel("Analysis year")
        ax.set_ylabel("Configured quota")
        for row_index, share in enumerate(shares):
            for column_index, year in enumerate(years):
                value = matrix.loc[share, year]
                if pd.notna(value):
                    text_colour = "white" if abs(float(value)) > limit * 0.55 else "black"
                    ax.text(
                        column_index,
                        row_index,
                        f"{float(value):+.0f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_colour,
                    )
    if image is not None:
        colourbar = fig.colorbar(image, ax=axes, shrink=0.86)
        colourbar.set_label("Difference [USD/t H₂-DRI]")
    fig.suptitle(
        "Recorded forced-run support versus contemporaneous ex-post H₂-DRI portfolio\n"
        "Positive (red) = forced is more expensive; negative (blue) = forced is cheaper",
        fontsize=15,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_support_totals(comparison: pd.DataFrame, years: tuple[int, ...], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(years)))
    for colour, year in zip(colours, years, strict=True):
        data = comparison[comparison["year"].eq(year) & comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        axes[0].plot(
            data["quota_share"] * 100,
            data["recorded_annualized_support_usd"] / 1e9,
            marker="o",
            color=colour,
            label=str(year),
        )
        axes[0].plot(
            data["quota_share"] * 100,
            data["ex_post_annualized_support_usd_at_forced_capacity"] / 1e9,
            linestyle="--",
            color=colour,
            alpha=0.9,
        )
        axes[1].plot(
            data["quota_share"] * 100,
            data["recorded_support_usd_per_t_steel"],
            marker="o",
            color=colour,
        )
        axes[1].plot(
            data["quota_share"] * 100,
            data["ex_post_average_support_usd_per_t_steel"],
            linestyle="--",
            color=colour,
        )
    axes[0].set_ylabel("Annualized support [billion USD/a]")
    axes[1].set_ylabel("Average support [USD/t steel]")
    axes[1].set_xlabel("Configured H₂-DRI quota [% of EU+EFTA+UK steel demand]")
    for ax in axes:
        ax.grid(linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
    axes[0].legend(title="Year", ncol=4, frameon=False)
    fig.suptitle(
        "Support needed for forced H₂-DRI capacity\n"
        "Solid: NPV gaps recorded in forced runs; dashed: vanilla hydrogen-specific ex-post valuation",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_iron_support_totals(comparison: pd.DataFrame, years: tuple[int, ...], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(years)))
    for colour, year in zip(colours, years, strict=True):
        data = comparison[comparison["year"].eq(year) & comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        axes[0].plot(
            data["quota_share"] * 100,
            data["recorded_annualized_support_usd"] / 1e9,
            marker="o",
            color=colour,
            label=str(year),
        )
        axes[0].plot(
            data["quota_share"] * 100,
            data["ex_post_iron_annualized_support_usd_at_forced_capacity"] / 1e9,
            linestyle="--",
            color=colour,
            alpha=0.9,
        )
        axes[1].plot(
            data["quota_share"] * 100,
            data["recorded_support_usd_per_t_iron"],
            marker="o",
            color=colour,
        )
        axes[1].plot(
            data["quota_share"] * 100,
            data["ex_post_average_support_usd_per_t_iron"],
            linestyle="--",
            color=colour,
        )
    axes[0].set_ylabel("Annualized support [billion USD/a]")
    axes[1].set_ylabel("Average support [USD/t H₂-DRI]")
    axes[1].set_xlabel("Configured H₂-DRI quota [% of EU+EFTA+UK steel demand]")
    for ax in axes:
        ax.grid(linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
    axes[0].legend(title="Year", ncol=4, frameon=False)
    fig.suptitle(
        "Support needed for forced H₂-DRI production\n"
        "Solid: NPV gaps recorded in forced runs; dashed: vanilla hydrogen-specific ex-post valuation",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_actual_cost_support_totals(
    comparison: pd.DataFrame,
    years: tuple[int, ...],
    output: Path,
) -> None:
    """Plot annual cash support implied by realised H2-DRI production costs."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(years)))
    for colour, year in zip(colours, years, strict=True):
        data = comparison[comparison["year"].eq(year) & comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        axes[0].plot(
            data["quota_share"] * 100,
            data["actual_cost_support_total_production_usd_per_year"] / 1e9,
            marker="o",
            color=colour,
            linewidth=2,
            label=str(year),
        )
        axes[0].plot(
            data["quota_share"] * 100,
            data["actual_cost_support_total_delivered_usd_per_year"] / 1e9,
            marker="x",
            color=colour,
            linewidth=1.5,
            linestyle="--",
        )
        axes[1].plot(
            data["quota_share"] * 100,
            data["actual_cost_support_gap_usd_per_t"],
            marker="o",
            color=colour,
            linewidth=2,
        )

    axes[0].set_ylabel("Required cash support [billion USD/a]")
    axes[1].set_ylabel("Realised cost shortfall [USD/t H₂-DRI]")
    axes[1].set_xlabel("Configured H₂-DRI quota [% of EU+EFTA+UK steel demand]")
    for ax in axes:
        ax.grid(linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
    year_legend = axes[0].legend(title="Year", ncol=4, frameon=False, loc="upper left")
    axes[0].add_artist(year_legend)
    basis_handles = [
        Line2D([], [], color="#555555", marker="o", linewidth=2, label="All realised H₂-DRI production"),
        Line2D(
            [],
            [],
            color="#555555",
            marker="x",
            linewidth=1.5,
            linestyle="--",
            label="Volume delivered to EU EAFs",
        ),
    ]
    axes[0].legend(handles=basis_handles, loc="upper right", frameon=False)
    fig.suptitle(
        "Annual support implied by realised forced-run H₂-DRI costs\n"
        "max(0, realised unit production cost − forced-run iron market price) × volume",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_total_steel_price_curves(
    merit: pd.DataFrame,
    comparison: pd.DataFrame,
    years: tuple[int, ...],
    output: Path,
) -> None:
    """Plot the market steel price plus the support needed to close the NPV gap."""

    fig, axes = plt.subplots(4, 2, figsize=(18, 25), sharey=False)
    axes_flat = axes.flat
    for ax, year in zip(axes_flat, years, strict=False):
        year_merit = merit[merit["target_year"].eq(year)].copy()
        if year_merit.empty:
            ax.axis("off")
            continue
        reference = float(year_merit.iloc[0]["reference_steel_price_usd_per_t"])
        left = 0.0
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        for row in year_merit.itertuples(index=False):
            width = float(row.annual_steel_equivalent_t) / 1e6
            price = float(row.required_steel_price_usd_per_t)
            ax.bar(left, price, width=width, align="edge", color="#a1d99b", edgecolor="white", linewidth=0.7)
            ax.text(
                left + width / 2,
                price + 8,
                f"+{row.steel_price_increase_pct:.0f}%",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
            tick_positions.append(left + width / 2)
            tick_labels.append(f"{row.country}\n{width:.1f} Mt")
            left += width
        ax.axhline(
            reference,
            color="#252525",
            linewidth=1.5,
            linestyle="--",
            label=f"Vanilla steel market price (${reference:.0f}/t)",
        )

        valid = comparison[comparison["year"].eq(year) & comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        ax.scatter(
            valid["active_steel_equivalent_t"] / 1e6,
            valid["recorded_required_steel_price_usd_per_t"],
            color="#e6550d",
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label="Forced runs: recorded break-even price",
        )
        for point in valid.itertuples(index=False):
            ax.annotate(
                f"{point.quota_share:.0%}\n+{point.recorded_steel_price_increase_pct:.0f}%",
                (point.active_steel_equivalent_t / 1e6, point.recorded_required_steel_price_usd_per_t),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        invalid = comparison[comparison["year"].eq(year) & ~comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        ax.scatter(
            invalid["active_steel_equivalent_t"] / 1e6,
            np.full(len(invalid), reference),
            color="#b2182b",
            marker="X",
            s=48,
            zorder=6,
            label="Forced run invalid/empty trade output",
        )
        ax.set_title(str(year))
        ax.set_xlabel("Annual steel-equivalent H₂-DRI output [Mt steel/a]")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=6)
        ax.set_ylabel("Indicative break-even steel price [USD/t]")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
    for ax in list(axes_flat)[len(years) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Indicative H₂-DRI steel break-even price\n"
        "Vanilla steel market price + annualized support required to bring project NPV to zero",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=4.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_total_iron_price_curves(
    merit: pd.DataFrame,
    comparison: pd.DataFrame,
    years: tuple[int, ...],
    output: Path,
) -> None:
    """Plot the market iron price plus the support needed to close the NPV gap."""

    fig, axes = plt.subplots(4, 2, figsize=(18, 25), sharey=False)
    axes_flat = axes.flat
    for ax, year in zip(axes_flat, years, strict=False):
        year_merit = merit[merit["target_year"].eq(year)].copy()
        if year_merit.empty:
            ax.axis("off")
            continue
        reference = float(year_merit.iloc[0]["reference_iron_price_usd_per_t"])
        left = 0.0
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        for row in year_merit.itertuples(index=False):
            width = float(row.annual_iron_output_t) / 1e6
            price = float(row.required_iron_price_usd_per_t)
            ax.bar(left, price, width=width, align="edge", color="#fdae6b", edgecolor="white", linewidth=0.7)
            ax.text(
                left + width / 2,
                price + 8,
                f"+{row.iron_price_increase_pct:.0f}%",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
            tick_positions.append(left + width / 2)
            tick_labels.append(f"{row.country}\n{width:.1f} Mt")
            left += width
        ax.axhline(
            reference,
            color="#252525",
            linewidth=1.5,
            linestyle="--",
            label=f"Vanilla iron market price (${reference:.0f}/t)",
        )
        valid = comparison[comparison["year"].eq(year) & comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        ax.scatter(
            valid["active_iron_output_t"] / 1e6,
            valid["recorded_required_iron_price_usd_per_t"],
            color="#e6550d",
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
            label="Forced runs: recorded break-even price",
        )
        for point in valid.itertuples(index=False):
            ax.annotate(
                f"{point.quota_share:.0%}\n+{point.recorded_iron_price_increase_pct:.0f}%",
                (point.active_iron_output_t / 1e6, point.recorded_required_iron_price_usd_per_t),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        invalid = comparison[comparison["year"].eq(year) & ~comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        ax.scatter(
            invalid["active_iron_output_t"] / 1e6,
            np.full(len(invalid), reference),
            color="#b2182b",
            marker="X",
            s=48,
            zorder=6,
            label="Forced run invalid/empty trade output",
        )
        ax.set_title(str(year))
        ax.set_xlabel("Annual H₂-DRI output [Mt iron/a]")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=6)
        ax.set_ylabel("Indicative break-even H₂-DRI price [USD/t]")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
    for ax in list(axes_flat)[len(years) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Indicative H₂-DRI break-even price\n"
        "Vanilla iron market price + annualized support required to bring project NPV to zero",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=4.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_actual_h2_dri_cost_curves(
    merit: pd.DataFrame,
    comparison: pd.DataFrame,
    years: tuple[int, ...],
    output: Path,
) -> None:
    """Compare realised forced-run H2-DRI costs with the vanilla ex-post curve."""

    fig, axes = plt.subplots(4, 2, figsize=(18, 25), sharey=False)
    axes_flat = axes.flatten()
    for ax, year in zip(axes_flat, years, strict=False):
        year_merit = merit[merit["target_year"].eq(year)].copy()
        if year_merit.empty:
            ax.axis("off")
            continue
        reference = float(year_merit.iloc[0]["reference_iron_price_usd_per_t"])
        left = 0.0
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        for row in year_merit.itertuples(index=False):
            width = float(row.annual_iron_output_t) / 1e6
            ax.bar(
                left,
                float(row.required_iron_price_usd_per_t),
                width=width,
                align="edge",
                color="#9ecae1",
                edgecolor="white",
                linewidth=0.7,
            )
            tick_positions.append(left + width / 2)
            tick_labels.append(f"{row.country}\n{width:.1f} Mt")
            left += width

        average_x = np.r_[0.0, year_merit["cumulative_iron_output_t"].to_numpy() / 1e6]
        average_y = np.r_[
            year_merit.iloc[0]["average_required_iron_price_usd_per_t"],
            year_merit["average_required_iron_price_usd_per_t"].to_numpy(),
        ]
        ax.plot(
            average_x,
            average_y,
            color="#08519c",
            linewidth=2.2,
            label="Ex-post portfolio-average break-even price",
        )
        ax.axhline(
            reference,
            color="#636363",
            linewidth=1.2,
            linestyle="--",
            label=f"Vanilla iron market price (${reference:.0f}/t)",
        )

        actual = comparison[comparison["year"].eq(year) & comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        ax.scatter(
            actual["actual_h2_dri_production_t"] / 1e6,
            actual["actual_h2_dri_unit_production_cost_usd_per_t"],
            color="#e6550d",
            edgecolor="black",
            linewidth=0.5,
            s=45,
            zorder=5,
            label="Forced run: realised H₂-DRI production cost",
        )
        for point in actual.itertuples(index=False):
            ax.annotate(
                f"{point.quota_share:.0%}",
                (
                    point.actual_h2_dri_production_t / 1e6,
                    point.actual_h2_dri_unit_production_cost_usd_per_t,
                ),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_title(str(year))
        ax.set_xlabel("Realised or prospective H₂-DRI output [Mt iron/a]")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=6)
        ax.set_ylabel("Cost or break-even price [USD/t H₂-DRI]")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
    for ax in axes_flat[len(years) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Realised forced-run H₂-DRI production cost versus vanilla ex-post investment curve\n"
        "Orange is realised annual cost; blue is the price required for a hypothetical new project to reach NPV = 0",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=4.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_actual_h2_dri_cost_timeseries(
    comparison: pd.DataFrame,
    years: tuple[int, ...],
    output: Path,
) -> None:
    """Show realised and matched ex-post H2-DRI values over time by quota."""

    shares = sorted(comparison["quota_share"].dropna().unique())
    fig, axes = plt.subplots(4, 2, figsize=(15, 19), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for ax, share in zip(axes_flat, shares, strict=False):
        all_share = comparison[comparison["quota_share"].eq(share)].sort_values("year")
        data = all_share[all_share["forced_run_valid"].astype(bool)]
        ax.plot(
            data["year"],
            data["actual_h2_dri_unit_production_cost_usd_per_t"],
            color="#e6550d",
            marker="o",
            linewidth=2,
            label="Realised forced-run production cost",
        )
        ax.plot(
            data["year"],
            data["ex_post_break_even_price_at_actual_h2_dri_production_usd_per_t"],
            color="#08519c",
            marker="o",
            linewidth=2,
            label="Ex-post break-even price at same output",
        )
        ax.plot(
            data["year"],
            data["reference_iron_price_usd_per_t"],
            color="#636363",
            linestyle="--",
            linewidth=1.2,
            label="Vanilla iron market price",
        )
        invalid_year = all_share["first_invalid_trade_year"].dropna()
        if not invalid_year.empty:
            first_invalid = int(invalid_year.iloc[0])
            ax.axvline(first_invalid, color="#b2182b", linestyle=":", linewidth=1.5)
            ax.text(
                first_invalid,
                0.98,
                "LP infeasible",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="right",
                va="top",
                fontsize=7,
                color="#b2182b",
            )
        ax.set_title(f"{share:.0%} quota")
        ax.set_ylabel("USD/t H₂-DRI")
        ax.set_xticks(years)
        ax.grid(linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    for ax in axes_flat[len(shares) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Realised H₂-DRI production cost and matched vanilla ex-post break-even price\n"
        "Only years with a feasible forced-run trade solution are shown",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_h2_dri_location_maps(
    location_data: pd.DataFrame,
    comparison: pd.DataFrame,
    shapefile: Path,
    years: tuple[int, ...],
    output_dir: Path,
) -> list[Path]:
    """Draw one quota-scenario map matrix for every requested decade."""

    world = gpd.read_file(shapefile)
    if world.crs is not None and not world.crs.is_geographic:
        world = world.to_crs(4326)
    shares = sorted(comparison["quota_share"].dropna().unique())
    forced_costs = location_data.loc[location_data["method"].eq("Forced"), "cost_usd_per_t"].dropna()
    cost_min = float(forced_costs.min()) if not forced_costs.empty else 0.0
    cost_max = float(forced_costs.max()) if not forced_costs.empty else 1.0
    if math.isclose(cost_min, cost_max):
        cost_min -= 1.0
        cost_max += 1.0
    cost_norm = Normalize(vmin=cost_min, vmax=cost_max)
    cmap = plt.get_cmap("Blues")
    outputs: list[Path] = []

    for year in years:
        fig, axes = plt.subplots(4, 2, figsize=(20, 15), constrained_layout=True)
        axes_flat = axes.flatten()
        for ax, share in zip(axes_flat, shares, strict=False):
            world.plot(ax=ax, color="#F2F1ED", edgecolor="#A9A9A9", linewidth=0.3)
            panel = location_data[location_data["year"].eq(year) & np.isclose(location_data["quota_share"], share)]
            forced = panel[panel["method"].eq("Forced")].dropna(subset=["latitude", "longitude"])
            ex_post = panel[panel["method"].eq("Ex-post")].dropna(subset=["latitude", "longitude"])
            status = comparison[comparison["year"].eq(year) & np.isclose(comparison["quota_share"], share)]
            valid = not status.empty and bool(status.iloc[0]["forced_run_valid"])

            if valid and not ex_post.empty:
                ax.scatter(
                    ex_post["longitude"],
                    ex_post["latitude"],
                    s=np.maximum(25.0, ex_post["represented_output_t"] / 1e6 * 55.0),
                    marker="D",
                    color="#f28e2b",
                    alpha=0.72,
                    edgecolor="#8c510a",
                    linewidth=0.6,
                    zorder=3,
                )
            if valid and not forced.empty:
                ax.scatter(
                    forced["longitude"],
                    forced["latitude"],
                    s=np.maximum(25.0, forced["represented_output_t"] / 1e6 * 55.0),
                    marker="o",
                    c=forced["cost_usd_per_t"],
                    cmap=cmap,
                    norm=cost_norm,
                    alpha=0.88,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
            if not valid:
                invalid_year = (
                    status["first_invalid_trade_year"].dropna() if not status.empty else pd.Series(dtype=float)
                )
                suffix = f" from {int(invalid_year.iloc[0])}" if not invalid_year.empty else ""
                ax.text(
                    0.5,
                    0.5,
                    f"Trade LP infeasible{suffix}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#b2182b",
                    fontsize=12,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "#b2182b", "alpha": 0.85},
                    zorder=5,
                )
            elif forced.empty:
                ax.text(0.5, 0.5, "No realised forced-project output", transform=ax.transAxes, ha="center")

            forced_volume = float(forced["represented_output_t"].sum()) / 1e6
            ex_post_volume = float(ex_post["represented_output_t"].sum()) / 1e6
            ax.set_title(f"{share:.0%} quota — forced {forced_volume:.1f} Mt/a, ex-post {ex_post_volume:.1f} Mt/a")
            # Both portfolios are restricted to EU27+EFTA+UK; a European
            # extent keeps nearby site coordinates distinguishable.
            ax.set_xlim(-25, 40)
            ax.set_ylim(32, 72)
            ax.set_xticks([-20, 0, 20, 40])
            ax.set_yticks([35, 45, 55, 65])
            ax.grid(color="#D5D5D5", linewidth=0.3, alpha=0.45)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        for ax in axes_flat[len(shares) :]:
            ax.axis("off")

        method_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=9,
                markerfacecolor=cmap(0.7),
                markeredgecolor="white",
                label="Forced project (shade = realised cost)",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markersize=8,
                markerfacecolor="#f28e2b",
                markeredgecolor="#8c510a",
                label="Ex-post H₂-DRI opportunity",
            ),
        ]
        fig.legend(handles=method_handles, loc="lower center", ncol=2, frameon=False)
        colourbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=cost_norm, cmap=cmap),
            ax=axes,
            location="right",
            shrink=0.75,
            pad=0.015,
        )
        colourbar.set_label("Realised forced-project production cost [USD/t H₂-DRI]")
        fig.suptitle(
            f"EU+EFTA+UK H₂-DRI locations — {year}\n"
            "Circle/diamond area is annual output; ex-post portfolio is matched to forced-project output",
            fontsize=17,
        )
        output = output_dir / f"h2_dri_location_comparison_{year}.png"
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    return outputs


def _fmt(value: float, scale: float = 1.0, digits: int = 1) -> str:
    return "n/a" if not math.isfinite(value) else f"{value / scale:.{digits}f}"


def write_findings(
    path: Path,
    comparison: pd.DataFrame,
    projects: pd.DataFrame,
    merit: pd.DataFrame,
    years: tuple[int, ...],
) -> None:
    lines = [
        "# H₂-DRI quota run comparison",
        "",
        "## Reading the figures",
        "",
        "The blue country blocks are the hydrogen-locked ex-post opportunities from the vanilla run. "
        "Their height is the negative NPV converted into an equivalent annual support payment and divided "
        "by annual steel-equivalent output. They are sorted by that displayed levelized support metric. "
        "Orange points use the NPVs written by valid forced quota runs; red crosses identify runs with empty "
        "trade output.",
        "",
        "`NPV support` is the one-off present value needed to bring a project NPV to zero. `Annualized support` "
        "uses each site's cost of equity and the configured 20-year plant life. USD/t steel divides that annual "
        "payment by H₂-DRI output converted with the configured iron-to-steel factor.",
        "The indicative total steel price is the vanilla global steel market-clearing price plus this support "
        "gap. It is a break-even willingness-to-pay indicator, not a separately simulated EAF selling price.",
        "",
        "## Main findings",
        "",
        "The like-for-like comparison is the forced-run **portfolio-average** support against the blue "
        "ex-post **cumulative-average** curve at the same volume. Comparing an orange point with the height "
        "of one country block mixes an average with a marginal value.",
        "",
        "A positive gap below means the recorded forced portfolio is more expensive; a negative gap means "
        "it is cheaper than the contemporaneous ex-post portfolio.",
        "",
        "| Year | Valid quotas | Forced minus ex-post [USD/t H₂-DRI] |",
        "|---:|:---|---:|",
    ]
    for year in years:
        data = comparison[comparison["year"].eq(year) & comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        gaps = data["recorded_minus_ex_post_usd_per_t_iron"].dropna()
        if gaps.empty:
            continue
        quota_text = ", ".join(f"{share:.0%}" for share in data["quota_share"])
        gap_text = f"{gaps.iloc[0]:+.1f}" if len(gaps) == 1 else f"{gaps.min():+.1f} to {gaps.max():+.1f}"
        lines.append(f"| {year} | {quota_text} | {gap_text} |")
    lines.extend(
        [
            "",
            "The year-by-year sign and magnitude should be read from the table rather than assumed to be "
            "monotonic. Forced projects retain investment-vintage economics while active, whereas the ex-post "
            "curve is recalculated for every target year. Cohort additions and retirements can therefore move "
            "the two measures in different directions.",
            "",
            "The decomposition figure uses four comparable portfolio averages. Let A be recorded forced "
            "support; B the same forced countries revalued as hydrogen DRI at their quota vintage; C those "
            "countries revalued under the current analysis year's conditions; and D the cheapest current "
            "ex-post portfolio at the same volume. Then **A − D = (A − B) + (B − C) + (C − D)**. These terms "
            "separate differences in stored project valuation, changing market conditions between vintages, "
            "and country selection. The forced-run NPV values and realised annual production costs remain "
            "different concepts; use the realised-cost figures for observed operating-cost comparisons.",
            "",
            "## Realised H₂-DRI production cost",
            "",
            "The forced-run quota report also contains a production-weighted realised H₂-DRI unit cost. "
            "It includes operating expenditure, carbon cost, the current year's debt repayment and "
            "by-product adjustments. This is different from the recorded NPV-support proxy.",
            "",
            "The new comparison places that realised annual production cost against the vanilla ex-post "
            "portfolio-average break-even price at the same realised H₂-DRI volume. The latter is the "
            "product price that would bring the hypothetical new-build portfolio's lifetime NPV to zero; "
            "it is not an accounting production cost. Their difference therefore measures an observed "
            "annual cost versus a counterfactual investment threshold, rather than two identical cost concepts.",
            "",
            "| Year | Valid quotas | Realised cost range | Ex-post break-even range | Realised minus ex-post |",
            "|---:|:---|---:|---:|---:|",
        ]
    )
    for year in years:
        data = comparison[comparison["year"].eq(year) & comparison["forced_run_valid"].astype(bool)].sort_values(
            "quota_share"
        )
        actual = data["actual_h2_dri_unit_production_cost_usd_per_t"].dropna()
        ex_post = data["ex_post_break_even_price_at_actual_h2_dri_production_usd_per_t"].dropna()
        gap = data["actual_cost_minus_ex_post_break_even_usd_per_t"].dropna()
        if actual.empty or ex_post.empty or gap.empty:
            continue
        quota_text = ", ".join(f"{share:.0%}" for share in data["quota_share"])
        lines.append(
            f"| {year} | {quota_text} | {actual.min():.1f}–{actual.max():.1f} | "
            f"{ex_post.min():.1f}–{ex_post.max():.1f} | {gap.min():+.1f} to {gap.max():+.1f} |"
        )
    lines.append("")
    lines.extend(
        [
            "### Cash support implied by realised costs",
            "",
            "This is calculated without NPVs or capital-recovery factors as "
            "`max(0, realised H₂-DRI cost − forced-run iron market price) × volume`. "
            "The production basis covers every tonne produced; the delivered basis covers only H₂-DRI "
            "actually routed into European EAFs.",
            "",
            "| Year | Unit shortfall [USD/t] | Production-basis support [bn USD/a] | "
            "Delivered-basis support [bn USD/a] |",
            "|---:|---:|---:|---:|",
        ]
    )
    for year in years:
        data = comparison[comparison["year"].eq(year) & comparison["forced_run_valid"].astype(bool)]
        gap = data["actual_cost_support_gap_usd_per_t"].dropna()
        production_total = data["actual_cost_support_total_production_usd_per_year"].dropna() / 1e9
        delivered_total = data["actual_cost_support_total_delivered_usd_per_year"].dropna() / 1e9
        if gap.empty or production_total.empty or delivered_total.empty:
            continue
        lines.append(
            f"| {year} | {gap.min():.1f}–{gap.max():.1f} | "
            f"{production_total.min():.2f}–{production_total.max():.2f} | "
            f"{delivered_total.min():.2f}–{delivered_total.max():.2f} |"
        )
    lines.append("")

    lines.extend(["## H₂-DRI iron-price results", ""])
    for year in (2030, 2050, 2060):
        if year not in years:
            continue
        data = comparison[comparison["year"].eq(year)].sort_values("quota_share")
        if data.empty:
            continue
        lines.extend(
            [
                f"### {year}",
                "",
                "| Quota | Recorded support USD/t iron | Ex-post support USD/t iron | Reference iron price | "
                "Recorded total (+%) | Ex-post total (+%) |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in data.itertuples(index=False):
            if not row.forced_run_valid:
                lines.append(
                    f"| {row.quota_share:.0%} | invalid | invalid | "
                    f"{_fmt(row.reference_iron_price_usd_per_t)} | invalid | invalid |"
                )
                continue
            lines.append(
                f"| {row.quota_share:.0%} | {_fmt(row.recorded_support_usd_per_t_iron)} | "
                f"{_fmt(row.ex_post_average_support_usd_per_t_iron)} | "
                f"{_fmt(row.reference_iron_price_usd_per_t)} | "
                f"{_fmt(row.recorded_required_iron_price_usd_per_t)} "
                f"(+{_fmt(row.recorded_iron_price_increase_pct)}%) | "
                f"{_fmt(row.ex_post_required_iron_price_usd_per_t)} "
                f"(+{_fmt(row.ex_post_iron_price_increase_pct)}%) |"
            )
        lines.append("")
        lines.append(
            "| Quota | Active projects | Actual capacity share | Total NPV support | Annualized support | "
            "Recorded USD/t steel | Ex-post USD/t steel | Reference steel price | Recorded total (+%) | "
            "Ex-post total (+%) |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in data.itertuples(index=False):
            if not row.forced_run_valid:
                lines.append(
                    f"| {row.quota_share:.0%} | {row.active_forced_projects} | invalid | "
                    "invalid (empty trade allocation) | invalid | invalid | invalid | "
                    f"{_fmt(row.reference_steel_price_usd_per_t)} | invalid | invalid |"
                )
                continue
            lines.append(
                f"| {row.quota_share:.0%} | {row.active_forced_projects} | {row.active_share_of_demand:.1%} | "
                f"{_fmt(row.recorded_support_npv_usd, 1e9)} bn USD NPV | "
                f"{_fmt(row.recorded_annualized_support_usd, 1e9)} bn USD/a | "
                f"{_fmt(row.recorded_support_usd_per_t_steel)} | "
                f"{_fmt(row.ex_post_average_support_usd_per_t_steel)} | "
                f"{_fmt(row.reference_steel_price_usd_per_t)} | "
                f"{_fmt(row.recorded_required_steel_price_usd_per_t)} (+{_fmt(row.recorded_steel_price_increase_pct)}%) | "
                f"{_fmt(row.ex_post_required_steel_price_usd_per_t)} (+{_fmt(row.ex_post_steel_price_increase_pct)}%) |"
            )
        lines.append("")

    rank_year = 2030
    ex_order = merit[merit["target_year"].eq(rank_year)].sort_values("merit_order_rank")["iso3"].head(10).tolist()
    largest = projects[projects["quota_year"].eq(rank_year)].sort_values("quota_share")
    forced_10 = largest[np.isclose(largest["quota_share"], 0.10)]["iso3"].tolist()
    invalid_onsets = (
        comparison.dropna(subset=["first_invalid_trade_year"])
        .groupby("scenario")["first_invalid_trade_year"]
        .first()
        .astype(int)
        .sort_values()
    )
    lines.extend(
        [
            "## Selection comparison",
            "",
            f"The vanilla hydrogen-specific 2030 merit order starts **{' → '.join(ex_order)}**. "
            f"The 10% forced run selected **{' → '.join(forced_10)}**.",
            "",
            "The forced scenarios are endogenous: extra H₂-DRI changes later prices, capacity and subsequent "
            "investment decisions. A forced-versus-ex-post gap therefore combines portfolio selection, investment "
            "vintage and genuine scenario feedback. It should not be interpreted as a clean causal premium.",
            "",
            "## Outputs",
            "",
            "- `h2_dri_marginal_cost_curves.png`: country-level blocks and forced-run points for every five years.",
            "- `h2_dri_support_totals.png`: annualized total support and average support per tonne.",
            "- `h2_dri_total_steel_price_curves.png`: total break-even steel price and percentage premium.",
            "- `h2_dri_iron_marginal_cost_curves.png`: the same merit order in physical H₂-DRI tonnes.",
            "- `h2_dri_iron_support_totals.png`: total support and support per tonne H₂-DRI.",
            "- `h2_dri_total_iron_price_curves.png`: total H₂-DRI break-even price and percentage premium.",
            "- `h2_dri_method_gap_decomposition.png`: total forced/ex-post gap split into valuation, vintage, "
            "and selection effects.",
            "- `h2_dri_actual_cost_vs_ex_post_curves.png`: realised H₂-DRI cost points over the ex-post "
            "country and cumulative-average curves.",
            "- `h2_dri_actual_cost_vs_ex_post_timeseries.png`: realised and matched ex-post values through "
            "time for every quota run.",
            "- `h2_dri_actual_cost_support_totals.png`: annual cash support based on realised unit costs, "
            "shown for total production and EU-EAF-delivered volume.",
            "- `h2_dri_actual_cost_support_totals.csv`: numerical actual-cost support totals.",
            "- `h2_dri_location_comparison_2030.png` through `..._2060.png`: exact forced and matched "
            "ex-post locations every ten years; blue shade records realised forced-project cost.",
            "- `h2_dri_location_comparison.csv`: coordinates, represented output and costs behind the maps.",
            "- `ex_post_h2_dri_merit_curve.csv`: every country block and all calculations.",
            "- `forced_h2_dri_projects.csv`: projects actually forced in each run.",
            "- `forced_h2_dri_support_summary.csv`: active forced fleet by year and quota share.",
            "- `h2_dri_quota_comparison.csv`: forced and ex-post values at matching capacity volumes.",
        ]
    )
    if invalid_onsets.empty:
        lines.extend(
            [
                "",
                "All tested quota runs retain non-empty trade results through the requested analysis horizon.",
            ]
        )
    else:
        invalid_text = ", ".join(f"{scenario}: {year}" for scenario, year in invalid_onsets.items())
        lines.extend(
            [
                "",
                "Runs with empty later trade-allocation output are marked invalid and omitted from forced-run "
                f"curves: **{invalid_text}**.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    results_dir: Path,
    output_dir: Path,
    years: tuple[int, ...],
    *,
    vanilla_run: Path | None = None,
    candidate_scope: str = "eu",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vanilla_run = vanilla_run or _find_vanilla_run(results_dir)
    vanilla_run = _latest_simulation_dir(vanilla_run)
    merit, _ = load_ex_post_curve(vanilla_run, years, candidate_scope)
    projects, forced = load_forced_runs(results_dir, years, merit)
    comparison = build_comparison_table(merit, forced)
    map_years = tuple(year for year in years if year % 10 == 0)
    forced_locations = load_forced_project_locations(results_dir, map_years)
    location_comparison = build_location_comparison(merit, forced_locations, comparison, map_years)

    merit.to_csv(output_dir / "ex_post_h2_dri_merit_curve.csv", index=False)
    projects.to_csv(output_dir / "forced_h2_dri_projects.csv", index=False)
    forced.to_csv(output_dir / "forced_h2_dri_support_summary.csv", index=False)
    comparison.to_csv(output_dir / "h2_dri_quota_comparison.csv", index=False)
    support_columns = [
        "year",
        "scenario",
        "quota_share",
        "forced_run_valid",
        "actual_h2_dri_unit_production_cost_usd_per_t",
        "actual_iron_market_price_usd_per_t",
        "actual_cost_support_gap_usd_per_t",
        "actual_h2_dri_production_t",
        "actual_h2_dri_delivered_to_eu_steel_t",
        "actual_cost_support_total_production_usd_per_year",
        "actual_cost_support_total_delivered_usd_per_year",
    ]
    comparison[support_columns].to_csv(output_dir / "h2_dri_actual_cost_support_totals.csv", index=False)
    location_comparison.to_csv(output_dir / "h2_dri_location_comparison.csv", index=False)
    plot_cost_curves(merit, forced, years, output_dir / "h2_dri_marginal_cost_curves.png")
    plot_iron_cost_curves(merit, forced, years, output_dir / "h2_dri_iron_marginal_cost_curves.png")
    plot_support_totals(comparison, years, output_dir / "h2_dri_support_totals.png")
    plot_iron_support_totals(comparison, years, output_dir / "h2_dri_iron_support_totals.png")
    plot_total_steel_price_curves(
        merit,
        comparison,
        years,
        output_dir / "h2_dri_total_steel_price_curves.png",
    )
    plot_total_iron_price_curves(
        merit,
        comparison,
        years,
        output_dir / "h2_dri_total_iron_price_curves.png",
    )
    plot_method_gap_decomposition(
        comparison,
        years,
        output_dir / "h2_dri_method_gap_decomposition.png",
    )
    plot_actual_h2_dri_cost_curves(
        merit,
        comparison,
        years,
        output_dir / "h2_dri_actual_cost_vs_ex_post_curves.png",
    )
    plot_actual_h2_dri_cost_timeseries(
        comparison,
        years,
        output_dir / "h2_dri_actual_cost_vs_ex_post_timeseries.png",
    )
    plot_actual_cost_support_totals(
        comparison,
        years,
        output_dir / "h2_dri_actual_cost_support_totals.png",
    )
    if map_years and not location_comparison.empty:
        plot_h2_dri_location_maps(
            location_comparison,
            comparison,
            _resolve_shapefile(vanilla_run, None),
            map_years,
            output_dir,
        )
    write_findings(output_dir / "H2_DRI_QUOTA_FINDINGS.md", comparison, projects, merit, years)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path, default=Path("cluster/results_h2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument(
        "--vanilla-run",
        type=Path,
        default=None,
        help="Vanilla simulation directory (may be outside results_dir)",
    )
    parser.add_argument(
        "--candidate-scope",
        choices=("eu", "global"),
        default="eu",
        help="Candidate pool used for the vanilla ex-post merit order",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir / "analysis"
    run_analysis(
        args.results_dir,
        output_dir,
        tuple(args.years),
        vanilla_run=args.vanilla_run,
        candidate_scope=args.candidate_scope,
    )
    print(f"Wrote H2-DRI quota comparison to {output_dir}")


if __name__ == "__main__":
    main()
