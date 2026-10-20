"""Configuration for the European hydrogen-DRI production quota."""

from dataclasses import dataclass


@dataclass
class H2DRIQuotaConfig:
    """Runtime settings for the EU+EFTA+UK H2-DRI quota.

    ``demand_share`` is the share of regional crude-steel demand whose iron
    requirement must be produced by hydrogen-fired DRI plants in the region.
    ``iron_per_steel`` converts steel-equivalent demand into physical DRI.
    ``overbuild_margin`` applies only to the capacity-planning target; the LP
    production floor itself is not inflated.
    ``candidate_scope`` is ``"eu"`` for domestic H2-DRI or ``"global"`` to
    allow ROW H2-DRI projects that supply EU+EFTA+UK EAFs.
    """

    enabled: bool = False
    demand_share: float = 0.10
    start_year: int = 2030
    overbuild_margin: float = 0.10
    iron_per_steel: float = 1.170923
    candidate_scope: str = "eu"

    def __post_init__(self) -> None:
        if not 0.0 <= self.demand_share <= 1.0:
            raise ValueError(f"demand_share must be within [0, 1], got {self.demand_share}")
        if self.start_year < 0:
            raise ValueError(f"start_year must be non-negative, got {self.start_year}")
        if self.overbuild_margin < 0.0:
            raise ValueError(f"overbuild_margin must be non-negative, got {self.overbuild_margin}")
        if self.iron_per_steel <= 0.0:
            raise ValueError(f"iron_per_steel must be positive, got {self.iron_per_steel}")
        if self.candidate_scope not in {"eu", "global"}:
            raise ValueError(f"candidate_scope must be 'eu' or 'global', got {self.candidate_scope!r}")
