"""Compare EU-only and global sourcing for the EU H2-DRI quota."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from steelo.h2_dri_quota.policy import EU_EFTA_UK_ISO3

from .h2_dri_quota_comparison import DEFAULT_YEARS, run_analysis


PRICE_COLUMNS = {
    "actual_h2_dri_unit_production_cost_usd_per_t": "H₂-DRI production cost",
    "actual_downstream_steel_unit_production_cost_usd_per_t": "Downstream steel production cost",
}


def _weighted_average(data: pd.DataFrame, value: str, weight: str = "production") -> float:
    selected = data[[value, weight]].apply(pd.to_numeric, errors="coerce").dropna()
    selected = selected[selected[weight].gt(0)]
    if selected.empty:
        return np.nan
    return float(np.average(selected[value], weights=selected[weight]))


def load_vanilla_price_reference(vanilla_run: Path, years: tuple[int, ...]) -> pd.DataFrame:
    """Load observed vanilla production costs and market prices."""

    post_processed = sorted(
        vanilla_run.glob("post_processed_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not post_processed:
        raise FileNotFoundError(f"No post_processed_*.csv found in {vanilla_run}")
    furnaces = pd.read_csv(post_processed[-1], low_memory=False)
    furnaces = furnaces.drop_duplicates(["year", "furnace_group_id"])
    market_files = sorted((vanilla_run / "data").glob("market_prices_*.csv"))
    if not market_files:
        raise FileNotFoundError(f"No market_prices_*.csv found in {vanilla_run / 'data'}")
    market = pd.read_csv(market_files[-1]).set_index("year")
    rows = []
    for year in years:
        annual = furnaces[furnaces["year"].eq(year)]
        record: dict[str, float | int] = {"year": year}
        for product in ("iron", "steel"):
            product_data = annual[annual["product"].astype(str).str.casefold().eq(product)]
            eu_data = product_data[product_data["iso3"].isin(EU_EFTA_UK_ISO3)]
            record[f"vanilla_eu_{product}_production_cost_usd_per_t"] = _weighted_average(
                eu_data,
                "unit_production_cost",
            )
            record[f"vanilla_world_{product}_production_cost_usd_per_t"] = _weighted_average(
                product_data,
                "unit_production_cost",
            )
        record["vanilla_market_iron_price_usd_per_t"] = float(market.loc[year, "iron_price_usd_per_t"])
        record["vanilla_market_steel_price_usd_per_t"] = float(market.loc[year, "steel_price_usd_per_t"])
        rows.append(record)
    return pd.DataFrame(rows)


def load_actual_built_merit_order(
    results_root: Path,
    vanilla_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load realised forced H2-DRI plants and aggregate them into country blocks."""

    frames = []
    for scope_dir, scope in (("EU", "eu"), ("global", "global")):
        for container in sorted((results_root / scope_dir).glob("h2dri_*")):
            simulations = sorted(container.glob("sim_*"), key=lambda path: path.stat().st_mtime)
            if not simulations:
                continue
            simulation = simulations[-1]
            outcomes_path = simulation / "data/h2_dri_quota/forced_project_outcomes.csv"
            summary_path = simulation / "data/h2_dri_quota/eu_h2_dri_quota_summary.csv"
            if not outcomes_path.is_file() or not summary_path.is_file():
                continue
            quota_share = float(pd.read_csv(summary_path, nrows=1)["quota_share"].iloc[0])
            outcomes = pd.read_csv(outcomes_path)
            outcomes["production_t"] = pd.to_numeric(outcomes["production_t"], errors="coerce")
            outcomes["unit_production_cost_usd_t"] = pd.to_numeric(
                outcomes["unit_production_cost_usd_t"],
                errors="coerce",
            )
            outcomes = outcomes[outcomes["production_t"].gt(0) & outcomes["unit_production_cost_usd_t"].notna()].copy()
            outcomes["candidate_scope"] = scope
            outcomes["scenario"] = container.name
            outcomes["quota_share"] = quota_share
            frames.append(outcomes)
    if not frames:
        raise FileNotFoundError(f"No realised forced-project outcomes found below {results_root}")
    projects = pd.concat(frames, ignore_index=True)
    projects = projects.merge(
        vanilla_prices[["year", "vanilla_market_iron_price_usd_per_t"]],
        on="year",
        how="left",
        validate="many_to_one",
    )
    projects["premium_vs_vanilla_iron_market_usd_per_t"] = (
        projects["unit_production_cost_usd_t"] - projects["vanilla_market_iron_price_usd_per_t"]
    )
    projects["price_increase_vs_vanilla_iron_market_pct"] = (
        projects["premium_vs_vanilla_iron_market_usd_per_t"] / projects["vanilla_market_iron_price_usd_per_t"] * 100.0
    )
    projects = projects.sort_values(["candidate_scope", "quota_share", "year", "unit_production_cost_usd_t", "iso3"])
    projects["plant_merit_order_rank"] = projects.groupby(["candidate_scope", "quota_share", "year"]).cumcount() + 1
    projects["cumulative_production_t"] = projects.groupby(["candidate_scope", "quota_share", "year"])[
        "production_t"
    ].cumsum()

    group_columns = ["candidate_scope", "scenario", "quota_share", "year", "iso3", "country"]
    country = (
        projects.groupby(group_columns, as_index=False)
        .apply(
            lambda data: pd.Series(
                {
                    "plant_count": data["furnace_group_id"].nunique(),
                    "nameplate_capacity_t": data["capacity_t"].sum(),
                    "production_t": data["production_t"].sum(),
                    "delivered_to_eu_eaf_t": data["delivered_to_eu_eaf_t"].sum(),
                    "unit_production_cost_usd_t": _weighted_average(
                        data,
                        "unit_production_cost_usd_t",
                        "production_t",
                    ),
                    "vanilla_market_iron_price_usd_per_t": data["vanilla_market_iron_price_usd_per_t"].iloc[0],
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    country["premium_vs_vanilla_iron_market_usd_per_t"] = (
        country["unit_production_cost_usd_t"] - country["vanilla_market_iron_price_usd_per_t"]
    )
    country["price_increase_vs_vanilla_iron_market_pct"] = (
        country["premium_vs_vanilla_iron_market_usd_per_t"] / country["vanilla_market_iron_price_usd_per_t"] * 100.0
    )
    country = country.sort_values(["candidate_scope", "quota_share", "year", "unit_production_cost_usd_t", "iso3"])
    country["country_merit_order_rank"] = country.groupby(["candidate_scope", "quota_share", "year"]).cumcount() + 1
    country["cumulative_production_t"] = country.groupby(["candidate_scope", "quota_share", "year"])[
        "production_t"
    ].cumsum()
    return projects, country


def build_scope_comparison(eu: pd.DataFrame, global_: pd.DataFrame) -> pd.DataFrame:
    """Return aligned EU-only versus global quota-run prices and premiums."""

    keys = ["year", "quota_share"]
    columns = [
        *keys,
        "scenario",
        "forced_run_valid",
        "actual_h2_dri_unit_production_cost_usd_per_t",
        "actual_downstream_steel_unit_production_cost_usd_per_t",
        "actual_eu_steel_unit_production_cost_usd_per_t",
        "actual_world_steel_unit_production_cost_usd_per_t",
        "actual_iron_market_price_usd_per_t",
        "actual_steel_market_price_usd_per_t",
        "actual_h2_dri_production_t",
        "actual_h2_dri_delivered_to_eu_steel_t",
        "actual_cost_support_gap_usd_per_t",
        "actual_cost_support_total_production_usd_per_year",
        "ex_post_break_even_price_at_actual_h2_dri_production_usd_per_t",
        "actual_cost_minus_ex_post_break_even_usd_per_t",
        "recorded_minus_ex_post_usd_per_t_iron",
    ]
    combined = eu[columns].merge(
        global_[columns],
        on=keys,
        how="outer",
        suffixes=("_eu_scope", "_global_scope"),
        validate="one_to_one",
    )
    for column in PRICE_COLUMNS:
        eu_column = f"{column}_eu_scope"
        global_column = f"{column}_global_scope"
        stem = column.removesuffix("_usd_per_t")
        combined[f"eu_minus_global_{stem}_premium_usd_per_t"] = combined[eu_column] - combined[global_column]
        combined[f"eu_minus_global_{stem}_premium_pct"] = (
            np.divide(
                combined[eu_column] - combined[global_column],
                combined[global_column],
            )
            * 100.0
        )

    for scope in ("eu_scope", "global_scope"):
        combined[f"h2_dri_premium_to_iron_market_usd_per_t_{scope}"] = (
            combined[f"actual_h2_dri_unit_production_cost_usd_per_t_{scope}"]
            - combined[f"actual_iron_market_price_usd_per_t_{scope}"]
        )
        combined[f"downstream_steel_premium_to_market_usd_per_t_{scope}"] = (
            combined[f"actual_downstream_steel_unit_production_cost_usd_per_t_{scope}"]
            - combined[f"actual_steel_market_price_usd_per_t_{scope}"]
        )
        combined[f"eu_minus_world_average_steel_cost_usd_per_t_{scope}"] = (
            combined[f"actual_eu_steel_unit_production_cost_usd_per_t_{scope}"]
            - combined[f"actual_world_steel_unit_production_cost_usd_per_t_{scope}"]
        )
    return combined.sort_values(keys).reset_index(drop=True)


def _annotated_matrix(
    ax: plt.Axes,
    data: pd.DataFrame,
    column: str,
    years: tuple[int, ...],
    shares: list[float],
    title: str,
    *,
    premium: bool = False,
) -> None:
    matrix = data.pivot(index="quota_share", columns="year", values=column).reindex(
        index=shares,
        columns=years,
    )
    values = matrix.to_numpy(dtype=float)
    if premium:
        limit = max(1.0, float(np.nanmax(np.abs(values))))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    else:
        image = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(years)), labels=years)
    ax.set_yticks(range(len(shares)), labels=[f"{share:.0%}" for share in shares])
    ax.set_xlabel("Year")
    ax.set_ylabel("Quota share")
    span = float(np.nanmax(values) - np.nanmin(values)) if np.isfinite(values).any() else 1.0
    midpoint = float(np.nanmin(values) + span / 2.0) if np.isfinite(values).any() else 0.0
    for row in range(len(shares)):
        for col in range(len(years)):
            value = values[row, col]
            if np.isfinite(value):
                colour = (
                    "white"
                    if (premium and abs(value) > limit * 0.55) or (not premium and value < midpoint)
                    else "black"
                )
                ax.text(
                    col,
                    row,
                    f"{value:+.0f}" if premium else f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=colour,
                )
    colourbar = ax.figure.colorbar(image, ax=ax, shrink=0.82)
    colourbar.set_label("USD/t")


def plot_price_matrices(data: pd.DataFrame, years: tuple[int, ...], output: Path) -> None:
    shares = sorted(data["quota_share"].dropna().unique(), reverse=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    rows = (
        ("actual_h2_dri_unit_production_cost_usd_per_t", "H₂-DRI"),
        ("actual_downstream_steel_unit_production_cost_usd_per_t", "Downstream steel"),
    )
    for row, (column, label) in enumerate(rows):
        _annotated_matrix(axes[row, 0], data, f"{column}_eu_scope", years, shares, f"EU-only sourcing: {label}")
        _annotated_matrix(axes[row, 1], data, f"{column}_global_scope", years, shares, f"Global sourcing: {label}")
        stem = column.removesuffix("_usd_per_t")
        _annotated_matrix(
            axes[row, 2],
            data,
            f"eu_minus_global_{stem}_premium_usd_per_t",
            years,
            shares,
            f"EU-only minus global: {label}",
            premium=True,
        )
    fig.suptitle(
        "EU-only versus global H₂-DRI quota sourcing\nPositive premium means EU-only sourcing is more expensive",
        fontsize=16,
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_2050_prices(data: pd.DataFrame, output: Path) -> None:
    year = 2050 if 2050 in set(data["year"]) else int(data["year"].max())
    selected = data[data["year"].eq(year)].sort_values("quota_share")
    x = selected["quota_share"] * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    panels = (
        ("actual_h2_dri_unit_production_cost_usd_per_t", "Iron", "H₂-DRI cost [USD/t iron]"),
        (
            "actual_downstream_steel_unit_production_cost_usd_per_t",
            "Steel",
            "Downstream steel cost [USD/t steel]",
        ),
    )
    for ax, (column, title, ylabel) in zip(axes, panels, strict=True):
        ax.plot(x, selected[f"{column}_eu_scope"], marker="o", linewidth=2, label="EU-only H₂-DRI")
        ax.plot(x, selected[f"{column}_global_scope"], marker="o", linewidth=2, label="Global H₂-DRI")
        market = "actual_iron_market_price_usd_per_t" if title == "Iron" else "actual_steel_market_price_usd_per_t"
        ax.plot(
            x,
            selected[f"{market}_eu_scope"],
            linestyle="--",
            color="#666666",
            label="Market price (EU-scope run)",
        )
        ax.set_title(f"{title} – {year}")
        ax.set_xlabel("Quota [% of EU+EFTA+UK steel demand]")
        ax.set_ylabel(ylabel)
        ax.grid(linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prices_against_vanilla(
    data: pd.DataFrame,
    vanilla: pd.DataFrame,
    output: Path,
    year: int = 2050,
) -> None:
    """Compare realised quota-run iron and downstream steel costs with vanilla."""

    selected = data[data["year"].eq(year)].sort_values("quota_share")
    reference = vanilla.set_index("year").loc[year]
    x = selected["quota_share"] * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    panels = (
        (
            axes[0],
            "actual_h2_dri_unit_production_cost_usd_per_t",
            "Iron",
            "Realised H₂-DRI production cost [USD/t iron]",
            (
                ("vanilla_world_iron_production_cost_usd_per_t", "Vanilla world iron production cost", "#238b45"),
                ("vanilla_market_iron_price_usd_per_t", "Vanilla iron market price", "#525252"),
            ),
        ),
        (
            axes[1],
            "actual_downstream_steel_unit_production_cost_usd_per_t",
            "Steel",
            "Receiving-EAF production cost [USD/t steel]",
            (
                ("vanilla_eu_steel_production_cost_usd_per_t", "Vanilla EU steel production cost", "#238b45"),
                ("vanilla_world_steel_production_cost_usd_per_t", "Vanilla world steel production cost", "#756bb1"),
                ("vanilla_market_steel_price_usd_per_t", "Vanilla steel market price", "#525252"),
            ),
        ),
    )
    for ax, column, title, ylabel, reference_lines in panels:
        ax.plot(
            x,
            selected[f"{column}_eu_scope"],
            marker="o",
            linewidth=2,
            label="EU-only quota run",
        )
        ax.plot(
            x,
            selected[f"{column}_global_scope"],
            marker="o",
            linewidth=2,
            label="Global quota run",
        )
        for reference_column, label, colour in reference_lines:
            value = float(reference[reference_column])
            if np.isfinite(value):
                ax.axhline(value, linestyle="--", color=colour, linewidth=1.5, label=f"{label} ({value:.0f})")
        ax.set_title(f"{title} – {year}")
        ax.set_xlabel("Quota [% of EU+EFTA+UK steel demand]")
        ax.set_ylabel(ylabel)
        ax.grid(linestyle="--", alpha=0.35)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Forced H₂-DRI quota runs compared with the specified vanilla simulation", fontsize=15)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_merit_order(
    merit: pd.DataFrame,
    candidate_scope: str,
    output_csv: Path,
    output_png: Path,
    years: tuple[int, ...],
) -> None:
    """Write and plot a country/capacity H2-DRI merit order."""

    columns = [
        "target_year",
        "merit_order_rank",
        "iso3",
        "country",
        "capacity_t",
        "capacity_counted_t",
        "annual_iron_output_t",
        "cumulative_iron_output_t",
        "reference_iron_price_usd_per_t",
        "levelized_support_usd_per_t_iron",
        "required_iron_price_usd_per_t",
        "iron_price_increase_pct",
        "indicative_policy_support_npv_usd",
    ]
    exported = merit[columns].copy()
    exported.insert(0, "candidate_scope", candidate_scope)
    exported.to_csv(output_csv, index=False)

    fig, axes = plt.subplots(4, 2, figsize=(19, 25))
    for ax, year in zip(axes.flat, years, strict=False):
        annual = exported[exported["target_year"].eq(year)]
        if annual.empty:
            ax.axis("off")
            continue
        left = 0.0
        positions = []
        labels = []
        for row in annual.itertuples(index=False):
            width = float(row.annual_iron_output_t) / 1e6
            price = float(row.required_iron_price_usd_per_t)
            premium = float(row.levelized_support_usd_per_t_iron)
            ax.bar(
                left,
                price,
                width=width,
                align="edge",
                color="#9ecae1" if candidate_scope == "EU-only" else "#fdae6b",
                edgecolor="white",
                linewidth=0.6,
            )
            ax.text(
                left + width / 2,
                price,
                f"+{premium:.0f}",
                rotation=90,
                ha="center",
                va="top",
                fontsize=5.5,
            )
            positions.append(left + width / 2)
            labels.append(f"{row.iso3}\n{float(row.capacity_t) / 1e6:.1f} Mt cap")
            left += width
        reference = float(annual.iloc[0]["reference_iron_price_usd_per_t"])
        ax.axhline(reference, color="black", linestyle="--", linewidth=1.2, label=f"Vanilla market: {reference:.0f}")
        ax.set_xticks(positions, labels=labels, rotation=90, fontsize=5.5)
        ax.set_title(str(year))
        ax.set_xlabel("Country and nameplate plant capacity")
        ax.set_ylabel("Break-even iron price [USD/t]")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    for ax in list(axes.flat)[len(years) :]:
        ax.axis("off")
    fig.suptitle(
        f"Vanilla H₂-DRI merit order – {candidate_scope}\n"
        "Bar height: market price + levelized NPV premium; number in bar: premium [USD/t iron]",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.5)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_actual_built_merit_order(
    projects: pd.DataFrame,
    scope: str,
    output: Path,
    year: int = 2050,
) -> None:
    """Plot realised plant portfolios for every quota share at one year."""

    selected = projects[projects["candidate_scope"].eq(scope) & projects["year"].eq(year)]
    shares = sorted(selected["quota_share"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(19, 25))
    colour = "#3182bd" if scope == "eu" else "#e6550d"
    for ax, share in zip(axes.flat, shares, strict=False):
        annual = selected[selected["quota_share"].eq(share)].sort_values("unit_production_cost_usd_t")
        left = 0.0
        positions = []
        labels = []
        for row in annual.itertuples(index=False):
            width = float(row.production_t) / 1e6
            price = float(row.unit_production_cost_usd_t)
            premium = float(row.premium_vs_vanilla_iron_market_usd_per_t)
            ax.bar(
                left,
                price,
                width=width,
                align="edge",
                color=colour,
                edgecolor="white",
                linewidth=0.7,
            )
            ax.text(
                left + width / 2,
                price,
                f"{premium:+.0f}",
                rotation=90,
                ha="center",
                va="top",
                fontsize=7,
                color="white",
            )
            positions.append(left + width / 2)
            labels.append(f"{row.iso3}\n{row.capacity_t / 1e6:.1f} Mt cap")
            left += width
        if not annual.empty:
            reference = float(annual.iloc[0]["vanilla_market_iron_price_usd_per_t"])
            ax.axhline(
                reference,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=f"Vanilla iron market price: {reference:.0f}",
            )
        ax.set_xticks(positions, labels=labels, rotation=90, fontsize=7)
        ax.set_title(f"{share:.0%} quota")
        ax.set_xlabel("Country and plant nameplate capacity")
        ax.set_ylabel("Realised H₂-DRI cost [USD/t iron]")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    for ax in list(axes.flat)[len(shares) :]:
        ax.axis("off")
    scope_label = "EU+EFTA+UK-only" if scope == "eu" else "Global"
    fig.suptitle(
        f"Actually built forced H₂-DRI portfolio – {scope_label}, {year}\n"
        "Plant blocks are sorted by realised cost; number in bar is premium versus vanilla market [USD/t]",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4.0)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_findings(data: pd.DataFrame, output: Path) -> None:
    lines = [
        "# EU-only versus global H₂-DRI quota comparison",
        "",
        "The EU-only runs may force H₂-DRI projects only in EU27+EFTA+UK. The global runs may force "
        "projects anywhere, while the resulting iron must still be delivered to EU27+EFTA+UK EAFs.",
        "",
        "All prices below are production-weighted realised annual costs from the quota reporter. They are "
        "not NPV-derived break-even prices.",
        "",
        "## 2050",
        "",
        "| Quota | EU H₂-DRI | Global H₂-DRI | EU iron premium | "
        "EU downstream steel | Global downstream steel | EU steel premium |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    year = 2050 if 2050 in set(data["year"]) else int(data["year"].max())
    selected = data[data["year"].eq(year)].sort_values("quota_share")
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.quota_share:.0%} | "
            f"{row.actual_h2_dri_unit_production_cost_usd_per_t_eu_scope:.1f} | "
            f"{row.actual_h2_dri_unit_production_cost_usd_per_t_global_scope:.1f} | "
            f"{row.eu_minus_global_actual_h2_dri_unit_production_cost_premium_usd_per_t:+.1f} | "
            f"{row.actual_downstream_steel_unit_production_cost_usd_per_t_eu_scope:.1f} | "
            f"{row.actual_downstream_steel_unit_production_cost_usd_per_t_global_scope:.1f} | "
            f"{row.eu_minus_global_actual_downstream_steel_unit_production_cost_premium_usd_per_t:+.1f} |"
        )
    iron_premium = selected["eu_minus_global_actual_h2_dri_unit_production_cost_premium_usd_per_t"]
    steel_premium = selected["eu_minus_global_actual_downstream_steel_unit_production_cost_premium_usd_per_t"]
    delivery_ratios = []
    for scope in ("eu_scope", "global_scope"):
        delivery_ratios.append(
            data[f"actual_h2_dri_delivered_to_eu_steel_t_{scope}"] / data[f"actual_h2_dri_production_t_{scope}"]
        )
    delivery_ratio = pd.concat(delivery_ratios).dropna()
    vanilla_gaps = {
        scope: selected[f"actual_cost_minus_ex_post_break_even_usd_per_t_{scope}"].dropna()
        for scope in ("eu_scope", "global_scope")
    }
    lines.extend(
        [
            "",
            f"In {year}, EU-only H₂-DRI costs "
            f"**{iron_premium.min():.0f}–{iron_premium.max():.0f} USD/t iron more** "
            "than the globally sourced portfolio across the tested quota shares.",
            "",
            "The downstream-steel difference is "
            f"**{steel_premium.min():+.0f} to {steel_premium.max():+.0f} USD/t steel**. "
            "It is not a mechanical iron-cost pass-through: the selected H₂-DRI locations, transport, destination "
            "EAF mix, EAF utilisation and endogenous market response also change between the two simulations.",
            "",
            f"Across the analysed observations, **{delivery_ratio.min():.0%}–{delivery_ratio.max():.0%}** of "
            "reported quota-scope H₂-DRI production is recorded as delivered to EU27+EFTA+UK EAFs. The quota "
            "report's `quota_met` field follows the production target, so it should not be read as proof that "
            "every forced tonne reaches a European EAF.",
            "",
            "## Forced runs versus vanilla merit order",
            "",
            f"In {year}, realised EU-only H₂-DRI cost is "
            f"**{vanilla_gaps['eu_scope'].min():.0f}–{vanilla_gaps['eu_scope'].max():.0f} USD/t** above the "
            "vanilla EU ex-post portfolio's NPV break-even price at the same volume. For global candidates, "
            f"the corresponding gap is **{vanilla_gaps['global_scope'].min():.0f}–"
            f"{vanilla_gaps['global_scope'].max():.0f} USD/t**.",
            "",
            "This comparison mixes a realised one-year production cost with a lifetime NPV break-even price; "
            "it is useful as a scenario diagnostic, not as a like-for-like accounting identity.",
            "",
            "## Files",
            "",
            "- `eu_vs_global_h2_dri_price_premiums.csv`: all aligned prices, volumes and premiums.",
            "- `eu_vs_global_h2_dri_price_matrices.png`: every quota and five-year checkpoint.",
            "- `eu_vs_global_h2_dri_prices_2050.png`: quota curves for realised iron and steel costs.",
            "- `eu_vs_global_h2_dri_country_mix.csv`: realised forced-project output by country.",
            "- `EU/analysis` and `global/analysis`: full forced-versus-vanilla analyses for each candidate scope.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scope_analysis(
    results_root: Path,
    vanilla_run: Path,
    output_dir: Path,
    years: tuple[int, ...],
) -> None:
    eu_output = results_root / "EU" / "analysis"
    global_output = results_root / "global" / "analysis"
    run_analysis(results_root / "EU", eu_output, years, vanilla_run=vanilla_run, candidate_scope="eu")
    run_analysis(
        results_root / "global",
        global_output,
        years,
        vanilla_run=vanilla_run,
        candidate_scope="global",
    )
    eu = pd.read_csv(eu_output / "h2_dri_quota_comparison.csv")
    global_ = pd.read_csv(global_output / "h2_dri_quota_comparison.csv")
    comparison = build_scope_comparison(eu, global_)
    output_dir.mkdir(parents=True, exist_ok=True)
    vanilla_prices = load_vanilla_price_reference(vanilla_run, years)
    vanilla_prices.to_csv(output_dir / "vanilla_price_reference.csv", index=False)
    actual_projects, actual_country_merit = load_actual_built_merit_order(
        results_root,
        vanilla_prices,
    )
    actual_projects.to_csv(output_dir / "actual_built_h2_dri_projects_merit_order.csv", index=False)
    actual_country_merit.to_csv(
        output_dir / "actual_built_h2_dri_country_merit_order.csv",
        index=False,
    )
    plot_actual_built_merit_order(
        actual_projects,
        "eu",
        output_dir / "actual_built_h2_dri_merit_order_eu_2050.png",
    )
    plot_actual_built_merit_order(
        actual_projects,
        "global",
        output_dir / "actual_built_h2_dri_merit_order_global_2050.png",
    )
    prices_vs_vanilla = comparison.merge(vanilla_prices, on="year", validate="many_to_one")
    for scope in ("eu_scope", "global_scope"):
        prices_vs_vanilla[f"h2_dri_premium_vs_vanilla_iron_market_usd_per_t_{scope}"] = (
            prices_vs_vanilla[f"actual_h2_dri_unit_production_cost_usd_per_t_{scope}"]
            - prices_vs_vanilla["vanilla_market_iron_price_usd_per_t"]
        )
        prices_vs_vanilla[f"downstream_steel_premium_vs_vanilla_eu_cost_usd_per_t_{scope}"] = (
            prices_vs_vanilla[f"actual_downstream_steel_unit_production_cost_usd_per_t_{scope}"]
            - prices_vs_vanilla["vanilla_eu_steel_production_cost_usd_per_t"]
        )
        prices_vs_vanilla[f"downstream_steel_premium_vs_vanilla_market_usd_per_t_{scope}"] = (
            prices_vs_vanilla[f"actual_downstream_steel_unit_production_cost_usd_per_t_{scope}"]
            - prices_vs_vanilla["vanilla_market_steel_price_usd_per_t"]
        )
    comparison.to_csv(output_dir / "eu_vs_global_h2_dri_price_premiums.csv", index=False)
    prices_vs_vanilla.to_csv(output_dir / "h2_dri_prices_vs_vanilla.csv", index=False)
    plot_prices_against_vanilla(
        comparison,
        vanilla_prices,
        output_dir / "h2_dri_prices_against_vanilla_2050.png",
    )
    for scope, label, scope_output in (
        ("eu", "EU-only", eu_output),
        ("global", "Global", global_output),
    ):
        merit = pd.read_csv(scope_output / "ex_post_h2_dri_merit_curve.csv")
        export_merit_order(
            merit,
            label,
            output_dir / f"vanilla_h2_dri_merit_order_{scope}.csv",
            output_dir / f"vanilla_h2_dri_merit_order_{scope}.png",
            years,
        )
    country_frames = []
    for scope, scope_output in (("eu", eu_output), ("global", global_output)):
        locations = pd.read_csv(scope_output / "h2_dri_location_comparison.csv")
        locations = locations[locations["method"].eq("Forced")].copy()
        locations["candidate_scope"] = scope
        country_frames.append(locations)
    country_mix = (
        pd.concat(country_frames, ignore_index=True)
        .groupby(["candidate_scope", "year", "quota_share", "iso3"], as_index=False)["represented_output_t"]
        .sum()
    )
    country_mix.to_csv(output_dir / "eu_vs_global_h2_dri_country_mix.csv", index=False)
    route_frames = []
    for scope in ("EU", "global"):
        containers = sorted((results_root / scope).glob("h2dri_*"))
        two_percent = next(
            container
            for container in containers
            if np.isclose(
                float(
                    pd.read_csv(
                        next(container.glob("sim_*")) / "data/h2_dri_quota/eu_h2_dri_quota_summary.csv",
                        nrows=1,
                    )["quota_share"].iloc[0]
                ),
                0.02,
            )
        )
        simulation = next(two_percent.glob("sim_*"))
        flows = pd.read_csv(simulation / "data/h2_dri_quota/eu_h2_dri_quota_flows.csv")
        flows = flows[flows["year"].eq(2050)].copy()
        flows.insert(0, "candidate_scope", scope.lower())
        route_frames.append(flows)
    pd.concat(route_frames, ignore_index=True).to_csv(
        output_dir / "h2_dri_2pct_2050_route_trace.csv",
        index=False,
    )
    plot_price_matrices(comparison, years, output_dir / "eu_vs_global_h2_dri_price_matrices.png")
    plot_2050_prices(comparison, output_dir / "eu_vs_global_h2_dri_prices_2050.png")
    write_findings(comparison, output_dir / "EU_VS_GLOBAL_H2_DRI_FINDINGS.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--vanilla-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_root / "analysis"
    run_scope_analysis(args.results_root, args.vanilla_run, output_dir, tuple(args.years))
    print(f"Wrote EU-versus-global H2-DRI comparison to {output_dir}")


if __name__ == "__main__":
    main()
