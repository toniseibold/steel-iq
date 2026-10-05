"""Shared post-processing labels for reductants and low-carbon production routes."""

from __future__ import annotations

import pandas as pd


DRI_HYDROGEN = "DRI (hydrogen)"
DRI_GAS = "DRI (natural gas)"
DRI_COAL = "DRI (coal)"
DRI_UNKNOWN = "DRI (unknown reductant)"

TECHNOLOGY_ROUTE_COLOURS = {
    DRI_HYDROGEN: "#1f78b4",  # requested blue
    DRI_GAS: "#f2c14e",  # requested yellow
    DRI_COAL: "#8c510a",
    DRI_UNKNOWN: "#bdbdbd",
}

CARBON_ROUTE_HATCHES = {
    "green": "///",
    "blue": "xxx",
    "grey": "...",
    "unknown": "",
    "not_applicable": "",
}
CARBON_ROUTE_LABELS = {
    "green": "Green: hydrogen",
    "blue": "Blue: fossil + CCS",
    "grey": "Grey: fossil without CCS",
    "unknown": "Unknown route",
}
CARBON_ROUTE_COMMODITIES = {
    "pig_iron",
    "hbi_low",
    "hbi_mid",
    "hbi_high",
    "dri_low",
    "dri_mid",
    "dri_high",
    "hot_metal",
    "liquid_iron",
}


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def technology_route_label(technology: object, chosen_reductant: object) -> str:
    """Split DRI into hydrogen, natural-gas, coal, and unknown routes."""
    technology_text = _text(technology)
    reductant = _text(chosen_reductant).casefold().replace("-", "_").replace(" ", "_")
    if not technology_text.upper().startswith("DRI"):
        return technology_text or "Unknown"
    if reductant in {"hydrogen", "h2", "green_hydrogen"}:
        return DRI_HYDROGEN
    if reductant in {"natural_gas", "gas", "methane"}:
        return DRI_GAS
    if reductant in {"coal", "pci", "coke", "coke+pci"}:
        return DRI_COAL
    return DRI_UNKNOWN


def carbon_route(technology: object, chosen_reductant: object) -> str:
    """Classify iron provenance as green, blue, grey, or unknown.

    CCS is taken from the technology name because CCS variants are explicit
    model technologies. A route is green only when hydrogen is the reductant,
    rather than merely a small co-injection in a fossil route.
    """
    technology_text = _text(technology).casefold()
    reductant = _text(chosen_reductant).casefold().replace("-", "_").replace(" ", "_")
    if reductant in {"hydrogen", "h2", "green_hydrogen"}:
        return "green"
    is_fossil = any(token in reductant for token in ("natural_gas", "gas", "coal", "coke", "pci"))
    if is_fossil and "ccs" in technology_text:
        return "blue"
    if is_fossil:
        return "grey"
    return "unknown"


def source_route_lookup(plants: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return one route record per producing furnace group in ``year``."""
    columns = ["furnace_group_id", "technology", "chosen_reductant"]
    if plants.empty or not set(columns[:2]).issubset(plants.columns):
        return pd.DataFrame(columns=[*columns, "technology_route", "carbon_route"])
    selected = plants
    if "year" in selected:
        selected = selected[pd.to_numeric(selected["year"], errors="coerce").eq(year)]
    selected = selected.drop_duplicates("furnace_group_id").copy()
    if "chosen_reductant" not in selected:
        selected["chosen_reductant"] = ""
    selected = selected[columns]
    selected["furnace_group_id"] = selected["furnace_group_id"].astype(str)
    selected["technology_route"] = [
        technology_route_label(technology, reductant)
        for technology, reductant in zip(selected["technology"], selected["chosen_reductant"])
    ]
    selected["carbon_route"] = [
        carbon_route(technology, reductant)
        for technology, reductant in zip(selected["technology"], selected["chosen_reductant"])
    ]
    return selected
