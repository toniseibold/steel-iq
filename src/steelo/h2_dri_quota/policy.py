"""Investment and trade enforcement for the European H2-DRI quota."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyomo.environ as pyo

from steelo.domain import Year
from steelo.domain.commands import UpdateFurnaceGroupStatus

if TYPE_CHECKING:
    from steelo.domain.trade_modelling.trade_lp_modelling import TradeLPModel
    from steelo.service_layer.message_bus import MessageBus


EU_EFTA_UK_ISO3 = frozenset(
    {
        "AUT",
        "BEL",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "SWE",
        "ISL",
        "LIE",
        "NOR",
        "CHE",
        "GBR",
    }
)
IRON_COMMODITIES = frozenset({"dri_low", "dri_mid", "dri_high", "hbi_low", "hbi_mid", "hbi_high"})

logger = logging.getLogger(__name__)


def is_h2_dri(technology: object, reductant: object) -> bool:
    """Return whether a technology/reductant pair is hydrogen-fired DRI."""

    return str(technology).strip().casefold() == "dri" and str(reductant).strip().casefold() == "hydrogen"


def _supports_hydrogen(furnace_group: object) -> bool:
    feedstocks = getattr(getattr(furnace_group, "technology", None), "dynamic_business_case", None) or []
    return any(str(getattr(feedstock, "reductant", "")).strip().casefold() == "hydrogen" for feedstock in feedstocks)


def regional_steel_demand(bus: "MessageBus", year: int) -> float:
    """Return EU27+EFTA+UK final steel demand in tonnes for ``year``."""

    target = Year(year)
    return float(
        sum(
            dc.demand_by_year.get(target, 0.0)
            for dc in bus.uow.repository.demand_centers.list()
            if dc.center_of_gravity.iso3 in EU_EFTA_UK_ISO3
        )
    )


def quota_production_target(bus: "MessageBus", year: int, config: object | None = None) -> float:
    """Physical H2-DRI tonnes required by the production quota."""

    if config is None:
        config = getattr(bus.env, "config", None)
    cfg = getattr(config, "h2_dri_quota", None)
    # Keeping this hook inert for legacy/test configurations is important: the
    # policy must not change ordinary trade models unless it is explicitly on.
    if cfg is None:
        return 0.0
    if not cfg.enabled or year < cfg.start_year:
        return 0.0
    return regional_steel_demand(bus, year) * cfg.demand_share * cfg.iron_per_steel


def _latest_npv(furnace_group: object) -> float:
    history = getattr(furnace_group, "historical_npv_business_opportunities", None) or {}
    finite = [float(value) for value in history.values() if value is not None and float(value) != float("-inf")]
    return finite[-1] if finite else float("-inf")


def _source_in_scope(iso3: str, cfg: object) -> bool:
    """Return whether an H2-DRI source is allowed by the configured merit order."""

    return getattr(cfg, "candidate_scope", "eu") == "global" or iso3 in EU_EFTA_UK_ISO3


def required_h2_dri_opportunity_count(bus: "MessageBus") -> int:
    """Return additional European DRI candidates needed for the next build decision.

    The ordinary opportunity selector keeps only the globally best location and
    technology pairs. A regional quota must therefore reserve enough European
    DRI candidates before that global truncation is applied.
    """

    cfg = getattr(getattr(bus.env, "config", None), "h2_dri_quota", None)
    if cfg is None or not cfg.enabled:
        return 0
    planning_year = int(bus.env.year)
    construction_time = bus.env.config.construction_time
    if planning_year + construction_time < cfg.start_year:
        return 0

    quota_year = max(cfg.start_year, planning_year + construction_time)
    required_production = quota_production_target(bus, quota_year)
    planned_nameplate = required_production / bus.env.config.capacity_limit * (1.0 + cfg.overbuild_margin)
    available_nameplate = 0.0
    for plant in bus.uow.plants.list():
        if not _source_in_scope(plant.location.iso3, cfg):
            continue
        for fg in plant.furnace_groups:
            if str(fg.technology.name).strip().casefold() != "dri":
                continue
            status = fg.status.casefold()
            if status in {"operating", "operating pre-retirement", "construction"}:
                if is_h2_dri(fg.technology.name, fg.chosen_reductant) and (
                    int(fg.lifetime.time_frame.start) <= quota_year < int(fg.lifetime.time_frame.end)
                ):
                    available_nameplate += float(fg.capacity)
            elif status in {"considered", "announced"} and (
                getattr(fg, "h2_dri_quota_candidate", False) or getattr(fg, "quota_reductant_lock", None) == "hydrogen"
            ):
                available_nameplate += float(fg.capacity)

    missing_nameplate = max(0.0, planned_nameplate - available_nameplate)
    candidate_capacity = float(bus.env.config.expanded_capacity)
    return math.ceil(missing_nameplate / candidate_capacity) if candidate_capacity > 0.0 else 0


def required_eu_eaf_opportunity_count(bus: "MessageBus") -> int:
    """Return the European EAF candidates needed to consume the quota iron."""

    cfg = getattr(getattr(bus.env, "config", None), "h2_dri_quota", None)
    if cfg is None or not cfg.enabled:
        return 0
    planning_year = int(bus.env.year)
    construction_time = bus.env.config.construction_time
    if planning_year + construction_time < cfg.start_year:
        return 0
    quota_year = max(cfg.start_year, planning_year + construction_time)
    required_steel = regional_steel_demand(bus, quota_year) * cfg.demand_share
    planned_nameplate = required_steel / bus.env.config.capacity_limit * (1.0 + cfg.overbuild_margin)
    available_nameplate = 0.0
    for plant in bus.uow.plants.list():
        if plant.location.iso3 not in EU_EFTA_UK_ISO3:
            continue
        for fg in plant.furnace_groups:
            if str(fg.technology.name).strip().casefold() != "eaf":
                continue
            status = fg.status.casefold()
            if status in {"operating", "operating pre-retirement", "construction"}:
                if int(fg.lifetime.time_frame.start) <= quota_year < int(fg.lifetime.time_frame.end):
                    available_nameplate += float(fg.capacity)
            elif status in {"considered", "announced"} and getattr(fg, "h2_dri_quota_eaf_candidate", False):
                available_nameplate += float(fg.capacity)
    missing_nameplate = max(0.0, planned_nameplate - available_nameplate)
    candidate_capacity = float(bus.env.config.expanded_capacity)
    return math.ceil(missing_nameplate / candidate_capacity) if candidate_capacity > 0.0 else 0


def force_eu_eaf_capacity(bus: "MessageBus") -> float:
    """Build and protect enough EU+EFTA+UK EAF nameplate for quota iron."""

    cfg = getattr(getattr(bus.env, "config", None), "h2_dri_quota", None)
    if cfg is None or not cfg.enabled:
        return 0.0
    planning_year = int(bus.env.year)
    quota_year = max(cfg.start_year, planning_year + bus.env.config.construction_time)
    if planning_year + bus.env.config.construction_time < cfg.start_year:
        return 0.0
    required_steel = regional_steel_demand(bus, quota_year) * cfg.demand_share
    planned_nameplate = required_steel / bus.env.config.capacity_limit * (1.0 + cfg.overbuild_margin)
    committed = 0.0
    operating: list[tuple[bool, object]] = []
    candidates: list[tuple[float, object, object]] = []
    for plant in bus.uow.plants.list():
        if plant.location.iso3 not in EU_EFTA_UK_ISO3:
            continue
        for fg in plant.furnace_groups:
            if str(fg.technology.name).strip().casefold() != "eaf":
                continue
            status = fg.status.casefold()
            if status in {"operating", "operating pre-retirement", "construction"}:
                if int(fg.lifetime.time_frame.start) <= quota_year < int(fg.lifetime.time_frame.end):
                    operating.append((bool(getattr(fg, "h2_dri_quota_protected", False)), fg))
            elif status in {"considered", "announced"} and getattr(fg, "h2_dri_quota_eaf_candidate", False):
                candidates.append((_latest_npv(fg), plant, fg))

    # Protect only the amount credited to this policy, preferring projects
    # already protected in prior planning years.
    for _already_protected, fg in sorted(operating, key=lambda item: item[0], reverse=True):
        if committed >= planned_nameplate:
            break
        fg.h2_dri_quota_protected = True
        committed += float(fg.capacity)

    gap = max(0.0, planned_nameplate - committed)
    forced = 0.0
    forced_rows: list[dict[str, object]] = []
    for npv, plant, fg in sorted(candidates, key=lambda item: item[0], reverse=True):
        if forced >= gap:
            break
        fg.h2_dri_quota_protected = True
        bus.handle(
            UpdateFurnaceGroupStatus(
                fg_id=fg.furnace_group_id,
                plant_id=plant.plant_id,
                new_status="construction",
            )
        )
        forced += float(fg.capacity)
        forced_rows.append(
            {
                "planning_year": planning_year,
                "quota_year": quota_year,
                "plant_id": plant.plant_id,
                "furnace_group_id": fg.furnace_group_id,
                "iso3": plant.location.iso3,
                "npv_construction_start_year": planning_year,
                "operation_start_year": planning_year + bus.env.config.construction_time,
                "latest_npv_usd": npv,
                "capacity_t": float(fg.capacity),
            }
        )
    remaining = max(0.0, gap - forced)
    logger.info(
        "operation=h2_dri_quota_eaf_capacity planning_year=%d quota_year=%d planned_nameplate_t=%.0f "
        "committed_nameplate_t=%.0f forced_nameplate_t=%.0f remaining_shortfall_t=%.0f",
        planning_year,
        quota_year,
        planned_nameplate,
        committed,
        forced,
        remaining,
    )
    if remaining > 1.0:
        logger.warning("[H2-DRI QUOTA] European EAF capacity plan remains %.0f t short", remaining)
    if forced_rows:
        policy_dir = bus.env.config.output_dir / "data" / "h2_dri_quota"
        policy_dir.mkdir(parents=True, exist_ok=True)
        _append_csv_rows(policy_dir / "forced_eaf_projects.csv", forced_rows)
    return remaining


@dataclass(frozen=True)
class ForcedCapacityResult:
    planning_year: int
    quota_year: int
    required_production_t: float
    planned_nameplate_t: float
    committed_nameplate_t: float
    forced_nameplate_t: float
    forced_projects: int
    forced_projects_npv_usd: float
    indicative_policy_support_npv_usd: float
    remaining_shortfall_t: float


def force_h2_dri_capacity(bus: "MessageBus") -> ForcedCapacityResult | None:
    """Force the best eligible opportunities into construction for a future quota year.

    The planning year is construction-time years before the first quota year and
    every year thereafter plans against the year in which newly started projects
    would commission. Existing operating/construction H2-DRI capacity is credited
    when its lifetime covers that quota year. Considered and announced European
    hydrogen-locked DRI opportunities are ranked by their hydrogen-specific NPV
    and forced into construction until the
    demand-equivalent target plus the configured capacity margin is covered.
    """

    cfg = getattr(getattr(bus.env, "config", None), "h2_dri_quota", None)
    if cfg is None or not cfg.enabled:
        return None

    planning_year = int(bus.env.year)
    quota_year = max(cfg.start_year, planning_year + bus.env.config.construction_time)
    if planning_year + bus.env.config.construction_time < cfg.start_year:
        return None

    required_production = quota_production_target(bus, quota_year)
    planned_nameplate = required_production / bus.env.config.capacity_limit * (1.0 + cfg.overbuild_margin)

    committed = 0.0
    operating: list[tuple[bool, object]] = []
    candidates: list[tuple[float, object, object]] = []
    for plant in bus.uow.plants.list():
        if not _source_in_scope(plant.location.iso3, cfg):
            continue
        for fg in plant.furnace_groups:
            if str(fg.technology.name).strip().casefold() != "dri":
                continue
            status = fg.status.casefold()
            if status in {"operating", "operating pre-retirement", "construction"}:
                if is_h2_dri(fg.technology.name, fg.chosen_reductant) and (
                    int(fg.lifetime.time_frame.start) <= quota_year < int(fg.lifetime.time_frame.end)
                ):
                    operating.append((bool(getattr(fg, "h2_dri_quota_protected", False)), fg))
            elif status in {"considered", "announced"} and (
                getattr(fg, "h2_dri_quota_candidate", False) or getattr(fg, "quota_reductant_lock", None) == "hydrogen"
            ):
                candidates.append((_latest_npv(fg), plant, fg))

    for _already_protected, fg in sorted(operating, key=lambda item: item[0], reverse=True):
        if committed >= planned_nameplate:
            break
        fg.h2_dri_quota_protected = True
        fg.quota_reductant_lock = "hydrogen"
        committed += float(fg.capacity)

    gap = max(0.0, planned_nameplate - committed)
    forced_capacity = 0.0
    forced_projects = 0
    forced_projects_npv = 0.0
    indicative_policy_support_npv = 0.0
    forced_rows: list[dict[str, object]] = []
    for npv, plant, fg in sorted(candidates, key=lambda item: item[0], reverse=True):
        if forced_capacity >= gap:
            break
        # This candidate was valued with hydrogen locked. Reassert the lock
        # before construction so annual cost refreshes cannot change route.
        fg.quota_reductant_lock = "hydrogen"
        fg.h2_dri_quota_protected = True
        fg.chosen_reductant = "hydrogen"
        refresh_energy_cost = getattr(fg, "generate_energy_vopex_by_reductant", None)
        if callable(refresh_energy_cost):
            refresh_energy_cost(forced_reductant="hydrogen")
        rebuild_bom = getattr(bus.env, "get_bom_from_avg_boms", None)
        if callable(rebuild_bom):
            bom, _utilization, reductant, output_shares = rebuild_bom(
                fg.energy_costs,
                fg.technology.name,
                float(fg.capacity),
                "hydrogen",
            )
            if bom is None or str(reductant).casefold() != "hydrogen":
                raise ValueError(f"Unable to rebuild hydrogen BOM for quota project {fg.furnace_group_id}")
            fg.bill_of_materials = bom
            fg.output_shares = output_shares
        bus.handle(
            UpdateFurnaceGroupStatus(
                fg_id=fg.furnace_group_id,
                plant_id=plant.plant_id,
                new_status="construction",
            )
        )
        forced_capacity += float(fg.capacity)
        forced_projects += 1
        forced_projects_npv += npv
        policy_support_npv = max(0.0, -npv)
        indicative_policy_support_npv += policy_support_npv
        forced_rows.append(
            {
                "planning_year": planning_year,
                "quota_year": quota_year,
                "plant_id": plant.plant_id,
                "furnace_group_id": fg.furnace_group_id,
                "iso3": plant.location.iso3,
                "technology": fg.technology.name,
                "chosen_reductant": fg.chosen_reductant,
                "candidate_scope": cfg.candidate_scope,
                "npv_route": "DRI hydrogen",
                "npv_construction_start_year": planning_year,
                "operation_start_year": planning_year + bus.env.config.construction_time,
                "latest_npv_usd": npv,
                "indicative_policy_support_npv_usd": policy_support_npv,
                "capacity_t": float(fg.capacity),
            }
        )
        logger.info(
            "[H2-DRI QUOTA] forced plant=%s fg=%s iso3=%s npv=%.2f capacity_t=%.0f quota_year=%d",
            plant.plant_id,
            fg.furnace_group_id,
            plant.location.iso3,
            npv,
            float(fg.capacity),
            quota_year,
        )

    remaining = max(0.0, gap - forced_capacity)
    result = ForcedCapacityResult(
        planning_year=planning_year,
        quota_year=quota_year,
        required_production_t=required_production,
        planned_nameplate_t=planned_nameplate,
        committed_nameplate_t=committed,
        forced_nameplate_t=forced_capacity,
        forced_projects=forced_projects,
        forced_projects_npv_usd=forced_projects_npv,
        indicative_policy_support_npv_usd=indicative_policy_support_npv,
        remaining_shortfall_t=remaining,
    )
    logger.info(
        "operation=h2_dri_quota_capacity planning_year=%d quota_year=%d required_production_t=%.0f "
        "planned_nameplate_t=%.0f committed_nameplate_t=%.0f forced_nameplate_t=%.0f "
        "forced_projects=%d forced_projects_npv_usd=%.0f indicative_policy_support_npv_usd=%.0f "
        "remaining_shortfall_t=%.0f",
        result.planning_year,
        result.quota_year,
        result.required_production_t,
        result.planned_nameplate_t,
        result.committed_nameplate_t,
        result.forced_nameplate_t,
        result.forced_projects,
        result.forced_projects_npv_usd,
        result.indicative_policy_support_npv_usd,
        result.remaining_shortfall_t,
    )
    if remaining > 1.0:
        logger.warning(
            "[H2-DRI QUOTA] Eligible European hydrogen-DRI opportunities are %.0f t short of the "
            "%d capacity plan; the %d production quota may be infeasible",
            remaining,
            quota_year,
            quota_year,
        )
    policy_dir = bus.env.config.output_dir / "data" / "h2_dri_quota"
    policy_dir.mkdir(parents=True, exist_ok=True)
    plan_path = policy_dir / "capacity_plan.csv"
    plan_row = {
        "planning_year": result.planning_year,
        "quota_year": result.quota_year,
        "required_production_t": result.required_production_t,
        "planned_nameplate_t": result.planned_nameplate_t,
        "committed_nameplate_t": result.committed_nameplate_t,
        "forced_nameplate_t": result.forced_nameplate_t,
        "forced_projects": result.forced_projects,
        "forced_projects_npv_usd": result.forced_projects_npv_usd,
        "indicative_policy_support_npv_usd": result.indicative_policy_support_npv_usd,
        "remaining_shortfall_t": result.remaining_shortfall_t,
        "candidate_scope": cfg.candidate_scope,
    }
    _append_csv_rows(plan_path, [plan_row])
    if forced_rows:
        _append_csv_rows(policy_dir / "forced_projects.csv", forced_rows)
    return result


def _append_csv_rows(path: object, rows: list[dict[str, object]]) -> None:
    """Append homogeneous policy records while writing the header once."""

    from pathlib import Path

    destination = Path(path)
    is_new = not destination.exists()
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def export_h2_dri_merit_order(output_dir: object, rows: list[dict[str, object]]) -> None:
    """Append one year's diagnostic H2-DRI merit order without affecting policy state."""

    if not rows:
        return
    from pathlib import Path

    destination = Path(output_dir) / "data" / "h2_dri_merit_order.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _append_csv_rows(destination, rows)


def add_h2_dri_production_quota(
    trade_lp: "TradeLPModel", bus: "MessageBus", year: int, config: object | None = None
) -> float:
    """Add the binding European H2-DRI-to-European-EAF lower bound to the LP."""

    target = quota_production_target(bus, year, config)
    if target <= 0.0:
        return 0.0

    eligible = []
    blocked_non_eaf = 0
    for from_pc, to_pc, commodity in trade_lp.legal_allocations:
        if commodity.name not in IRON_COMMODITIES:
            continue
        if from_pc.process.technology.casefold() != "dri" or "hydrogen" not in from_pc.process.name.casefold():
            continue
        if not any(product.name == "steel" for product in to_pc.process.products):
            continue
        key = (from_pc.name, to_pc.name, commodity.name)
        if to_pc.process.technology.casefold() != "eaf":
            # H2-DRI is a quota-backed primary iron route in this policy and may
            # only be converted to steel in an EAF. This blocks BOF and any
            # other steelmaking destinations, including outside Europe.
            trade_lp.lp_model.allocation_variables[key].fix(0.0)
            blocked_non_eaf += 1
            continue
        cfg = getattr(config or getattr(bus.env, "config", None), "h2_dri_quota", None)
        if cfg is None or not _source_in_scope(from_pc.location.iso3, cfg):
            continue
        if to_pc.location.iso3 not in EU_EFTA_UK_ISO3:
            continue
        eligible.append(key)

    if not eligible:
        raise ValueError(
            f"H2-DRI quota is active in {year}, but the trade LP has no eligible hydrogen-DRI to EU+EFTA+UK EAF arcs"
        )

    trade_lp.lp_model.h2_dri_quota_target_t = target
    trade_lp.lp_model.h2_dri_quota_arc_count = len(eligible)
    trade_lp.lp_model.h2_dri_non_eaf_arc_count = blocked_non_eaf
    trade_lp.lp_model.h2_dri_quota_constraint = pyo.Constraint(
        expr=sum(trade_lp.lp_model.allocation_variables[key] for key in eligible) >= target
    )
    logger.info(
        "operation=h2_dri_quota_lp year=%d target_t=%.0f eligible_eaf_arcs=%d blocked_non_eaf_arcs=%d",
        year,
        target,
        len(eligible),
        blocked_non_eaf,
    )
    return target
