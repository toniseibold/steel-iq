from pathlib import Path
from types import SimpleNamespace

import pyomo.environ as pyo
import pandas as pd
import pytest

from steelo.domain import Year
from steelo.h2_dri_quota.config import H2DRIQuotaConfig
from steelo.h2_dri_quota.policy import (
    add_h2_dri_production_quota,
    force_eu_eaf_capacity,
    force_h2_dri_capacity,
    required_eu_eaf_opportunity_count,
    required_h2_dri_opportunity_count,
)


def _location(iso3: str):
    return SimpleNamespace(iso3=iso3)


def test_config_defaults_and_validation():
    config = H2DRIQuotaConfig()
    assert config.enabled is False
    assert config.demand_share == 0.10
    assert config.start_year == 2030
    assert config.overbuild_margin == 0.10
    assert config.iron_per_steel == pytest.approx(1.170923)
    with pytest.raises(ValueError, match="demand_share"):
        H2DRIQuotaConfig(demand_share=1.1)


def test_lp_constraint_counts_hydrogen_dri_into_european_steel():
    source_process = SimpleNamespace(technology="DRI", name="DRI_hydrogen", products=[SimpleNamespace(name="dri_high")])
    steel_process = SimpleNamespace(technology="EAF", name="EAF", products=[SimpleNamespace(name="steel")])
    source = SimpleNamespace(name="source", location=_location("DEU"), process=source_process)
    destination = SimpleNamespace(name="steel", location=_location("FRA"), process=steel_process)
    commodity = SimpleNamespace(name="dri_high")
    model = pyo.ConcreteModel()
    model.allocation_variables = pyo.Var([("source", "steel", "dri_high")], domain=pyo.NonNegativeReals)
    trade_lp = SimpleNamespace(
        legal_allocations=[(source, destination, commodity)],
        lp_model=model,
    )
    demand_center = SimpleNamespace(center_of_gravity=_location("DEU"), demand_by_year={Year(2030): 1_000.0})
    repository = SimpleNamespace(demand_centers=SimpleNamespace(list=lambda: [demand_center]))
    bus = SimpleNamespace(
        env=SimpleNamespace(
            config=SimpleNamespace(h2_dri_quota=H2DRIQuotaConfig(enabled=True, demand_share=0.10, iron_per_steel=1.2))
        ),
        uow=SimpleNamespace(repository=repository),
    )

    target = add_h2_dri_production_quota(trade_lp, bus, 2030)

    assert target == pytest.approx(120.0)
    assert pyo.value(model.h2_dri_quota_constraint.lower) == pytest.approx(120.0)
    assert model.h2_dri_quota_arc_count == 1


def test_lp_constraint_blocks_hydrogen_dri_to_non_eaf_steelmaking():
    source_process = SimpleNamespace(technology="DRI", name="DRI_hydrogen", products=[])
    eaf_process = SimpleNamespace(technology="EAF", name="EAF", products=[SimpleNamespace(name="steel")])
    bof_process = SimpleNamespace(technology="BOF", name="BOF", products=[SimpleNamespace(name="steel")])
    source = SimpleNamespace(name="source", location=_location("DEU"), process=source_process)
    eaf = SimpleNamespace(name="eaf", location=_location("FRA"), process=eaf_process)
    bof = SimpleNamespace(name="bof", location=_location("FRA"), process=bof_process)
    commodity = SimpleNamespace(name="dri_high")
    keys = [("source", "eaf", "dri_high"), ("source", "bof", "dri_high")]
    model = pyo.ConcreteModel()
    model.allocation_variables = pyo.Var(keys, domain=pyo.NonNegativeReals)
    trade_lp = SimpleNamespace(
        legal_allocations=[(source, eaf, commodity), (source, bof, commodity)],
        lp_model=model,
    )
    demand_center = SimpleNamespace(center_of_gravity=_location("DEU"), demand_by_year={Year(2030): 1_000.0})
    repository = SimpleNamespace(demand_centers=SimpleNamespace(list=lambda: [demand_center]))
    bus = SimpleNamespace(
        env=SimpleNamespace(
            config=SimpleNamespace(h2_dri_quota=H2DRIQuotaConfig(enabled=True, demand_share=0.10, iron_per_steel=1.2))
        ),
        uow=SimpleNamespace(repository=repository),
    )

    add_h2_dri_production_quota(trade_lp, bus, 2030)

    assert model.h2_dri_quota_arc_count == 1
    assert model.h2_dri_non_eaf_arc_count == 1
    assert model.allocation_variables[("source", "bof", "dri_high")].fixed
    assert pyo.value(model.allocation_variables[("source", "bof", "dri_high")]) == pytest.approx(0.0)


def test_global_scope_counts_row_hydrogen_dri_feeding_european_eaf():
    source_process = SimpleNamespace(technology="DRI", name="DRI_hydrogen", products=[])
    eaf_process = SimpleNamespace(technology="EAF", name="EAF", products=[SimpleNamespace(name="steel")])
    source = SimpleNamespace(name="source", location=_location("CAN"), process=source_process)
    destination = SimpleNamespace(name="eaf", location=_location("DEU"), process=eaf_process)
    commodity = SimpleNamespace(name="dri_high")
    model = pyo.ConcreteModel()
    model.allocation_variables = pyo.Var([("source", "eaf", "dri_high")], domain=pyo.NonNegativeReals)
    trade_lp = SimpleNamespace(legal_allocations=[(source, destination, commodity)], lp_model=model)
    demand_center = SimpleNamespace(center_of_gravity=_location("DEU"), demand_by_year={Year(2030): 1_000.0})
    cfg = H2DRIQuotaConfig(enabled=True, demand_share=0.1, iron_per_steel=1.0, candidate_scope="global")
    bus = SimpleNamespace(
        env=SimpleNamespace(config=SimpleNamespace(h2_dri_quota=cfg)),
        uow=SimpleNamespace(repository=SimpleNamespace(demand_centers=SimpleNamespace(list=lambda: [demand_center]))),
    )

    add_h2_dri_production_quota(trade_lp, bus, 2030)

    assert model.h2_dri_quota_arc_count == 1


def test_capacity_forcing_ranks_candidates_by_npv_and_overbuilds(tmp_path: Path):
    def candidate(name: str, npv: float):
        return SimpleNamespace(
            furnace_group_id=name,
            capacity=70.0,
            status="considered",
            technology=SimpleNamespace(name="DRI", dynamic_business_case=[SimpleNamespace(reductant="hydrogen")]),
            chosen_reductant="hydrogen",
            quota_reductant_lock="hydrogen",
            h2_dri_quota_candidate=True,
            historical_npv_business_opportunities={2026: npv},
        )

    low = candidate("low", 10.0)
    high = candidate("high", 20.0)
    plants = [
        SimpleNamespace(plant_id="p-low", location=_location("DEU"), furnace_groups=[low]),
        SimpleNamespace(plant_id="p-high", location=_location("FRA"), furnace_groups=[high]),
    ]
    demand_center = SimpleNamespace(center_of_gravity=_location("DEU"), demand_by_year={Year(2030): 1_000.0})
    commands = []
    bus = SimpleNamespace(
        env=SimpleNamespace(
            year=Year(2026),
            config=SimpleNamespace(
                h2_dri_quota=H2DRIQuotaConfig(
                    enabled=True, demand_share=0.10, start_year=2030, overbuild_margin=0.10, iron_per_steel=1.0
                ),
                construction_time=4,
                capacity_limit=0.95,
                end_year=2040,
                output_dir=tmp_path,
            ),
        ),
        uow=SimpleNamespace(
            plants=SimpleNamespace(list=lambda: plants),
            repository=SimpleNamespace(demand_centers=SimpleNamespace(list=lambda: [demand_center])),
        ),
        handle=commands.append,
    )

    result = force_h2_dri_capacity(bus)

    assert result is not None
    assert result.planned_nameplate_t == pytest.approx(100.0 / 0.95 * 1.10)
    assert result.forced_nameplate_t == 140.0
    assert [command.fg_id for command in commands] == ["high", "low"]
    assert high.quota_reductant_lock == "hydrogen"
    assert high.h2_dri_quota_protected is True
    assert high.chosen_reductant == "hydrogen"
    assert result.forced_projects_npv_usd == pytest.approx(30.0)
    assert result.indicative_policy_support_npv_usd == pytest.approx(0.0)
    assert (tmp_path / "data" / "h2_dri_quota" / "capacity_plan.csv").exists()


def test_capacity_forcing_outputs_negative_npv_as_policy_support(tmp_path: Path):
    candidate = SimpleNamespace(
        furnace_group_id="negative",
        capacity=200.0,
        status="considered",
        technology=SimpleNamespace(name="DRI", dynamic_business_case=[SimpleNamespace(reductant="hydrogen")]),
        chosen_reductant="hydrogen",
        quota_reductant_lock="hydrogen",
        h2_dri_quota_candidate=True,
        historical_npv_business_opportunities={2026: -75.0},
    )
    plant = SimpleNamespace(plant_id="p-negative", location=_location("DEU"), furnace_groups=[candidate])
    demand_center = SimpleNamespace(center_of_gravity=_location("DEU"), demand_by_year={Year(2030): 1_000.0})
    bus = SimpleNamespace(
        env=SimpleNamespace(
            year=Year(2026),
            config=SimpleNamespace(
                h2_dri_quota=H2DRIQuotaConfig(
                    enabled=True, demand_share=0.10, start_year=2030, overbuild_margin=0.0, iron_per_steel=1.0
                ),
                construction_time=4,
                capacity_limit=0.95,
                output_dir=tmp_path,
            ),
        ),
        uow=SimpleNamespace(
            plants=SimpleNamespace(list=lambda: [plant]),
            repository=SimpleNamespace(demand_centers=SimpleNamespace(list=lambda: [demand_center])),
        ),
        handle=lambda _command: None,
    )

    result = force_h2_dri_capacity(bus)
    forced = pd.read_csv(tmp_path / "data" / "h2_dri_quota" / "forced_projects.csv")

    assert result is not None
    assert result.forced_projects_npv_usd == pytest.approx(-75.0)
    assert result.indicative_policy_support_npv_usd == pytest.approx(75.0)
    assert forced.loc[0, "latest_npv_usd"] == pytest.approx(-75.0)
    assert forced.loc[0, "indicative_policy_support_npv_usd"] == pytest.approx(75.0)


def test_opportunity_count_reserves_enough_regional_dri_candidates():
    demand_center = SimpleNamespace(center_of_gravity=_location("DEU"), demand_by_year={Year(2030): 1_000.0})
    bus = SimpleNamespace(
        env=SimpleNamespace(
            year=Year(2026),
            config=SimpleNamespace(
                h2_dri_quota=H2DRIQuotaConfig(
                    enabled=True,
                    demand_share=0.10,
                    start_year=2030,
                    overbuild_margin=0.10,
                    iron_per_steel=1.0,
                ),
                construction_time=4,
                capacity_limit=0.95,
                expanded_capacity=70.0,
            ),
        ),
        uow=SimpleNamespace(
            plants=SimpleNamespace(list=lambda: []),
            repository=SimpleNamespace(demand_centers=SimpleNamespace(list=lambda: [demand_center])),
        ),
    )

    assert required_h2_dri_opportunity_count(bus) == 2


def test_ordinary_gas_dri_is_not_treated_as_quota_candidate():
    gas = SimpleNamespace(
        capacity=200.0,
        status="considered",
        technology=SimpleNamespace(name="DRI", dynamic_business_case=[SimpleNamespace(reductant="hydrogen")]),
        chosen_reductant="natural_gas",
        quota_reductant_lock=None,
        h2_dri_quota_candidate=False,
    )
    demand_center = SimpleNamespace(center_of_gravity=_location("DEU"), demand_by_year={Year(2030): 1_000.0})
    bus = SimpleNamespace(
        env=SimpleNamespace(
            year=Year(2026),
            config=SimpleNamespace(
                h2_dri_quota=H2DRIQuotaConfig(enabled=True, demand_share=0.1, iron_per_steel=1.0),
                construction_time=4,
                capacity_limit=0.95,
                expanded_capacity=70.0,
            ),
        ),
        uow=SimpleNamespace(
            plants=SimpleNamespace(list=lambda: [SimpleNamespace(location=_location("DEU"), furnace_groups=[gas])]),
            repository=SimpleNamespace(demand_centers=SimpleNamespace(list=lambda: [demand_center])),
        ),
    )

    assert required_h2_dri_opportunity_count(bus) == 2


def test_forcing_ignores_higher_npv_ordinary_gas_candidate(tmp_path: Path):
    gas = SimpleNamespace(
        furnace_group_id="gas",
        capacity=200.0,
        status="considered",
        technology=SimpleNamespace(name="DRI", dynamic_business_case=[SimpleNamespace(reductant="hydrogen")]),
        chosen_reductant="natural_gas",
        quota_reductant_lock=None,
        h2_dri_quota_candidate=False,
        historical_npv_business_opportunities={2026: 1_000.0},
    )
    hydrogen = SimpleNamespace(
        furnace_group_id="hydrogen",
        capacity=200.0,
        status="considered",
        technology=SimpleNamespace(name="DRI", dynamic_business_case=[SimpleNamespace(reductant="hydrogen")]),
        chosen_reductant="hydrogen",
        quota_reductant_lock="hydrogen",
        h2_dri_quota_candidate=True,
        historical_npv_business_opportunities={2026: -100.0},
    )
    plants = [
        SimpleNamespace(plant_id="p-gas", location=_location("DEU"), furnace_groups=[gas]),
        SimpleNamespace(plant_id="p-hydrogen", location=_location("FRA"), furnace_groups=[hydrogen]),
    ]
    demand_center = SimpleNamespace(center_of_gravity=_location("DEU"), demand_by_year={Year(2030): 1_000.0})
    commands = []
    bus = SimpleNamespace(
        env=SimpleNamespace(
            year=Year(2026),
            config=SimpleNamespace(
                h2_dri_quota=H2DRIQuotaConfig(enabled=True, demand_share=0.1, iron_per_steel=1.0),
                construction_time=4,
                capacity_limit=0.95,
                output_dir=tmp_path,
            ),
        ),
        uow=SimpleNamespace(
            plants=SimpleNamespace(list=lambda: plants),
            repository=SimpleNamespace(demand_centers=SimpleNamespace(list=lambda: [demand_center])),
        ),
        handle=commands.append,
    )

    force_h2_dri_capacity(bus)

    assert [command.fg_id for command in commands] == ["hydrogen"]


def test_eaf_planner_forces_and_protects_best_european_candidate(tmp_path: Path):
    candidate = SimpleNamespace(
        furnace_group_id="eaf-best",
        capacity=200.0,
        status="considered",
        technology=SimpleNamespace(name="EAF"),
        historical_npv_business_opportunities={2026: 25.0},
        h2_dri_quota_protected=False,
        h2_dri_quota_eaf_candidate=True,
    )
    plant = SimpleNamespace(plant_id="p-eaf", location=_location("DEU"), furnace_groups=[candidate])
    demand_center = SimpleNamespace(center_of_gravity=_location("DEU"), demand_by_year={Year(2030): 1_000.0})
    commands = []
    bus = SimpleNamespace(
        env=SimpleNamespace(
            year=Year(2026),
            config=SimpleNamespace(
                h2_dri_quota=H2DRIQuotaConfig(enabled=True, demand_share=0.1, iron_per_steel=1.0),
                construction_time=4,
                capacity_limit=0.95,
                expanded_capacity=200.0,
                output_dir=tmp_path,
            ),
        ),
        uow=SimpleNamespace(
            plants=SimpleNamespace(list=lambda: [plant]),
            repository=SimpleNamespace(demand_centers=SimpleNamespace(list=lambda: [demand_center])),
        ),
        handle=commands.append,
    )

    assert required_eu_eaf_opportunity_count(bus) == 0
    remaining = force_eu_eaf_capacity(bus)

    assert remaining == 0.0
    assert candidate.h2_dri_quota_protected is True
    assert [command.fg_id for command in commands] == ["eaf-best"]
