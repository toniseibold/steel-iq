"""Post-run cost and destination reporting for the European H2-DRI quota."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import H2DRIQuotaConfig
from .policy import EU_EFTA_UK_ISO3, IRON_COMMODITIES

_YEAR = re.compile(r"steel_trade_allocations_(\d{4})\.csv$")
FLOW_COLUMNS = [
    "year",
    "source_iso3",
    "source_country",
    "source_plant_id",
    "source_furnace_group_id",
    "iron_commodity",
    "volume_t",
    "h2_dri_unit_production_cost_usd_t",
    "destination_iso3",
    "destination_country",
    "destination_plant_id",
    "destination_furnace_group_id",
    "destination_technology",
    "subsequent_steel_unit_production_cost_usd_t",
]
PROJECT_OUTCOME_COLUMNS = [
    "planning_year",
    "quota_year",
    "operation_start_year",
    "year",
    "plant_id",
    "furnace_group_id",
    "latitude",
    "longitude",
    "iso3",
    "country",
    "technology",
    "chosen_reductant",
    "candidate_scope",
    "capacity_t",
    "status",
    "production_t",
    "unit_production_cost_usd_t",
    "latest_npv_usd",
    "indicative_policy_support_npv_usd",
    "delivered_to_eu_eaf_t",
    "destination_countries",
]


def _weighted_average(data: pd.DataFrame, value: str, weight: str) -> float:
    clean = data[[value, weight]].apply(pd.to_numeric, errors="coerce").dropna()
    clean = clean[clean[weight] > 0]
    return float(np.average(clean[value], weights=clean[weight])) if not clean.empty else float("nan")


def _unique_furnaces(data: pd.DataFrame) -> pd.DataFrame:
    required = [
        "year",
        "iso3",
        "country",
        "plant_id",
        "furnace_group_id",
        "technology",
        "product",
        "chosen_reductant",
        "production",
        "unit_production_cost",
    ]
    optional = ["latitude", "longitude", "capacity", "status"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Quota reporting is missing post-processed columns: {', '.join(sorted(missing))}")
    selected = data.copy()
    for column in optional:
        if column not in selected:
            selected[column] = np.nan
    return selected[required + optional].drop_duplicates(["year", "furnace_group_id"])


def _write_forced_project_outcomes(*, output_dir: Path, furnaces: pd.DataFrame, flows: pd.DataFrame) -> Path:
    """Combine forced-project investment data with annual physical outcomes."""

    destination = output_dir / "forced_project_outcomes.csv"
    forced_path = output_dir / "forced_projects.csv"
    if not forced_path.exists():
        pd.DataFrame(columns=PROJECT_OUTCOME_COLUMNS).to_csv(destination, index=False)
        return destination

    forced = pd.read_csv(forced_path)
    records: list[dict[str, object]] = []
    for project in forced.to_dict("records"):
        fg_id = str(project["furnace_group_id"])
        annual_rows = furnaces[furnaces["furnace_group_id"].astype(str).eq(fg_id)].sort_values("year")
        if annual_rows.empty:
            annual_rows = pd.DataFrame([{"year": np.nan}])
        for annual in annual_rows.to_dict("records"):
            year = annual.get("year")
            annual_flows = flows[flows["source_furnace_group_id"].astype(str).eq(fg_id) & flows["year"].eq(year)]
            destination_totals = annual_flows.groupby("destination_iso3", as_index=False)["volume_t"].sum()
            destinations = [
                f"{row.destination_iso3} ({float(row.volume_t) / 1e6:.3f} Mt)"
                for row in destination_totals.sort_values("volume_t", ascending=False).itertuples(index=False)
            ]
            records.append(
                {
                    "planning_year": project.get("planning_year"),
                    "quota_year": project.get("quota_year"),
                    "operation_start_year": project.get("operation_start_year"),
                    "year": year,
                    "plant_id": project.get("plant_id", annual.get("plant_id")),
                    "furnace_group_id": fg_id,
                    "latitude": annual.get("latitude"),
                    "longitude": annual.get("longitude"),
                    "iso3": annual.get("iso3", project.get("iso3")),
                    "country": annual.get("country"),
                    "technology": annual.get("technology", project.get("technology")),
                    "chosen_reductant": annual.get("chosen_reductant", project.get("chosen_reductant")),
                    "candidate_scope": project.get("candidate_scope"),
                    "capacity_t": project.get("capacity_t", annual.get("capacity")),
                    "status": annual.get("status"),
                    "production_t": annual.get("production"),
                    "unit_production_cost_usd_t": annual.get("unit_production_cost"),
                    "latest_npv_usd": project.get("latest_npv_usd"),
                    "indicative_policy_support_npv_usd": project.get("indicative_policy_support_npv_usd"),
                    "delivered_to_eu_eaf_t": float(annual_flows["volume_t"].sum()),
                    "destination_countries": "; ".join(destinations),
                }
            )
    pd.DataFrame(records, columns=PROJECT_OUTCOME_COLUMNS).to_csv(destination, index=False)
    return destination


def generate_h2_dri_quota_report(
    *,
    post_processed_csv: Path,
    tm_dir: Path,
    output_dir: Path,
    config: H2DRIQuotaConfig,
) -> tuple[Path, Path, Path]:
    """Write quota compliance, H2-DRI destinations, and cost-comparison outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    furnaces = _unique_furnaces(pd.read_csv(post_processed_csv))
    furnaces["year"] = pd.to_numeric(furnaces["year"], errors="coerce").astype("Int64")
    furnaces["production"] = pd.to_numeric(furnaces["production"], errors="coerce").fillna(0.0)
    furnaces["unit_production_cost"] = pd.to_numeric(furnaces["unit_production_cost"], errors="coerce")
    h2 = furnaces["technology"].fillna("").str.casefold().eq("dri") & furnaces["chosen_reductant"].fillna(
        ""
    ).str.casefold().eq("hydrogen")
    if config.candidate_scope == "eu":
        h2 &= furnaces["iso3"].isin(EU_EFTA_UK_ISO3)
    h2_ids = {int(year): set(group["furnace_group_id"].astype(str)) for year, group in furnaces[h2].groupby("year")}
    steel_ids = {
        int(year): set(group["furnace_group_id"].astype(str))
        for year, group in furnaces[
            furnaces["product"].fillna("").str.casefold().eq("steel")
            & furnaces["technology"].fillna("").str.casefold().eq("eaf")
            & furnaces["iso3"].isin(EU_EFTA_UK_ISO3)
        ].groupby("year")
    }
    furnace_lookup = furnaces.set_index(["year", "furnace_group_id"], drop=False)

    flow_records: list[dict[str, object]] = []
    eu_demand: dict[int, float] = {}
    for path in sorted(tm_dir.glob("steel_trade_allocations_*.csv")):
        match = _YEAR.search(path.name)
        if not match:
            continue
        year = int(match.group(1))
        trade = pd.read_csv(path)
        # Final deliveries identify regional steel demand without requiring the
        # original workbook during post-processing.
        final_steel = trade[trade["commodity"].eq("steel") & trade["destination_type"].eq("DemandCenter")]
        destination_iso = (
            final_steel["destination_location"].astype(str).str.extract(r"iso3=['\"]([A-Z]{3})['\"]", expand=False)
        )
        eu_demand[year] = float(final_steel.loc[destination_iso.isin(EU_EFTA_UK_ISO3), "allocated_volume"].sum())

        selected = trade[
            trade["commodity"].isin(IRON_COMMODITIES)
            & trade["source_id"].astype(str).isin(h2_ids.get(year, set()))
            & trade["destination_id"].astype(str).isin(steel_ids.get(year, set()))
            & trade["allocated_volume"].gt(0)
        ]
        for row in selected.itertuples(index=False):
            source = furnace_lookup.loc[(year, str(row.source_id))]
            destination = furnace_lookup.loc[(year, str(row.destination_id))]
            flow_records.append(
                {
                    "year": year,
                    "source_iso3": source["iso3"],
                    "source_country": source["country"],
                    "source_plant_id": source["plant_id"],
                    "source_furnace_group_id": row.source_id,
                    "iron_commodity": row.commodity,
                    "volume_t": float(row.allocated_volume),
                    "h2_dri_unit_production_cost_usd_t": source["unit_production_cost"],
                    "destination_iso3": destination["iso3"],
                    "destination_country": destination["country"],
                    "destination_plant_id": destination["plant_id"],
                    "destination_furnace_group_id": row.destination_id,
                    "destination_technology": destination["technology"],
                    "subsequent_steel_unit_production_cost_usd_t": destination["unit_production_cost"],
                }
            )

    # Preserve a useful schema even if the configured start year lies beyond
    # the run or the solve produced no eligible flow.
    flows = pd.DataFrame(flow_records, columns=FLOW_COLUMNS)
    flow_path = output_dir / "eu_h2_dri_quota_flows.csv"
    flows.to_csv(flow_path, index=False)
    _write_forced_project_outcomes(output_dir=output_dir, furnaces=furnaces, flows=flows)

    market_files = sorted(output_dir.parent.glob("market_prices_*.csv"))
    market = pd.read_csv(market_files[-1]).set_index("year") if market_files else pd.DataFrame()
    summaries: list[dict[str, object]] = []
    for year in sorted(int(value) for value in furnaces["year"].dropna().unique()):
        annual = furnaces[furnaces["year"].eq(year)]
        annual_h2 = annual[h2.loc[annual.index]]
        iron = annual[annual["product"].fillna("").str.casefold().eq("iron")]
        steel = annual[annual["product"].fillna("").str.casefold().eq("steel")]
        eu_iron = iron[iron["iso3"].isin(EU_EFTA_UK_ISO3)]
        eu_steel = steel[steel["iso3"].isin(EU_EFTA_UK_ISO3)]
        annual_flows = flows[flows["year"].eq(year)] if not flows.empty else flows
        downstream_ids = set(annual_flows.get("destination_furnace_group_id", pd.Series(dtype=str)).astype(str))
        downstream = eu_steel[eu_steel["furnace_group_id"].astype(str).isin(downstream_ids)]
        other_eu_steel = eu_steel[~eu_steel["furnace_group_id"].astype(str).isin(downstream_ids)]
        demand = eu_demand.get(year, 0.0)
        target = demand * config.demand_share * config.iron_per_steel if year >= config.start_year else 0.0
        delivered = float(annual_flows["volume_t"].sum()) if not annual_flows.empty else 0.0
        summaries.append(
            {
                "year": year,
                "quota_enabled": True,
                "candidate_scope": config.candidate_scope,
                "quota_share": config.demand_share,
                "quota_iron_per_steel": config.iron_per_steel,
                "capacity_overbuild_margin": config.overbuild_margin,
                "eu_steel_demand_t": demand,
                "quota_production_target_t": target,
                "quota_scope_h2_dri_production_t": float(annual_h2["production"].sum()),
                "quota_h2_dri_delivered_to_eu_eaf_t": delivered,
                # Backward-compatible aliases used by the comparison script.
                "eu_h2_dri_production_t": float(annual_h2["production"].sum()),
                "eu_h2_dri_delivered_to_eu_steel_t": delivered,
                "quota_met": bool(year < config.start_year or delivered + 1e-3 >= target),
                "h2_dri_unit_production_cost_usd_t": _weighted_average(annual_h2, "unit_production_cost", "production"),
                "other_eu_iron_unit_production_cost_usd_t": _weighted_average(
                    eu_iron[~eu_iron["furnace_group_id"].isin(annual_h2["furnace_group_id"])],
                    "unit_production_cost",
                    "production",
                ),
                "world_iron_unit_production_cost_usd_t": _weighted_average(iron, "unit_production_cost", "production"),
                "h2_dri_downstream_steel_unit_production_cost_usd_t": _weighted_average(
                    downstream, "unit_production_cost", "production"
                ),
                "other_eu_steel_unit_production_cost_usd_t": _weighted_average(
                    other_eu_steel, "unit_production_cost", "production"
                ),
                "eu_steel_unit_production_cost_usd_t": _weighted_average(
                    eu_steel, "unit_production_cost", "production"
                ),
                "world_steel_unit_production_cost_usd_t": _weighted_average(
                    steel, "unit_production_cost", "production"
                ),
                "market_iron_price_usd_t": market.loc[year, "iron_price_usd_per_t"]
                if not market.empty and year in market.index
                else np.nan,
                "market_steel_price_usd_t": market.loc[year, "steel_price_usd_per_t"]
                if not market.empty and year in market.index
                else np.nan,
            }
        )

    summary = pd.DataFrame(summaries)
    summary_path = output_dir / "eu_h2_dri_quota_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    iron_lines = {
        "Quota-scope H₂-DRI": "h2_dri_unit_production_cost_usd_t",
        "Other European iron": "other_eu_iron_unit_production_cost_usd_t",
        "World iron": "world_iron_unit_production_cost_usd_t",
        "Iron market price": "market_iron_price_usd_t",
    }
    steel_lines = {
        "Steel receiving quota H₂-DRI": "h2_dri_downstream_steel_unit_production_cost_usd_t",
        "Other European steel": "other_eu_steel_unit_production_cost_usd_t",
        "World steel": "world_steel_unit_production_cost_usd_t",
        "Steel market price": "market_steel_price_usd_t",
    }
    for label, column in iron_lines.items():
        axes[0].plot(summary["year"], summary[column], marker="o", markersize=3, label=label)
    for label, column in steel_lines.items():
        axes[1].plot(summary["year"], summary[column], marker="o", markersize=3, label=label)
    axes[0].set_title("Iron production cost and market price")
    axes[1].set_title("Subsequent steel production cost and market price")
    for axis in axes:
        axis.set_xlabel("Year")
        axis.set_ylabel("USD/t")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        axis.legend(frameon=False, fontsize=8)
    source_label = "EU27 + EFTA + UK" if config.candidate_scope == "eu" else "global"
    fig.suptitle(f"H₂-DRI quota ({source_label} candidates → EU27 + EFTA + UK EAFs): cost comparison")
    fig.tight_layout()
    plot_path = output_dir / "eu_h2_dri_quota_cost_comparison.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary_path, flow_path, plot_path
