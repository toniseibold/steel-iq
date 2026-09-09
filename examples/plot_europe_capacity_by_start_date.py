"""Plot European iron and steel capacity additions by plant start year.

The equipment expansion and capacity mapping follow ``MasterExcelReader.read_plants``:

* semicolon-delimited equipment strings are expanded into separate technologies;
* BF and DRI use their technology-specific iron capacity (falling back to total
  nominal iron capacity);
* BOF uses its steel capacity (falling back to total crude-steel capacity);
* OHF is folded into EAF and its capacity is added to EAF capacity.

Colors represent processes and hatches represent capacity operating statuses.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


DEFAULT_INPUT = Path("/home/toni-seibold/.steelo/data_cache/master-input-v2.0.0/master_input_0.xlsx")
DEFAULT_OUTPUT = Path("plots/europe_capacity_by_start_date.png")
SHEET_NAME = "Iron and steel plants"

PROCESS_ORDER = ["BF", "DRI", "BOF", "EAF"]
PROCESS_COLORS = {
    "BF": "#4C566A",
    "DRI": "#2A9D8F",
    "BOF": "#E76F51",
    "EAF": "#457B9D",
}
STATUS_ORDER = [
    "operating",
    "operating pre-retirement",
    "construction",
    "announced",
    "mothballed",
    "mothballed pre-retirement",
    "retired",
    "cancelled",
]
STATUS_HATCHES = {
    "operating": "",
    "operating pre-retirement": "///",
    "construction": "xxx",
    "announced": "...",
    "mothballed": "\\\\\\",
    "mothballed pre-retirement": "ooo",
    "retired": "---",
    "cancelled": "+++",
}


def _number(value: object) -> float:
    """Match the main reader's treatment of blank and non-numeric capacity."""
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(parsed) else float(parsed)


def _capacity(row: pd.Series, process: str) -> float:
    """Return capacity in ktpa using the same process mapping as the main flow."""
    if process == "BF":
        return _number(row.get("Nominal BF capacity (ttpa)")) or _number(row.get("Nominal iron capacity (ttpa)"))
    if process == "DRI":
        return _number(row.get("Nominal DRI capacity (ttpa)")) or _number(row.get("Nominal iron capacity (ttpa)"))
    if process == "BOF":
        return _number(row.get("Nominal BOF steel capacity (ttpa)")) or _number(
            row.get("Nominal crude steel capacity (ttpa)")
        )
    if process == "EAF":
        return _number(row.get("Nominal EAF steel capacity (ttpa)")) + _number(
            row.get("Nominal OHF steel capacity (ttpa)")
        )
    raise ValueError(f"Unsupported process: {process}")


def _start_year(value: object) -> int | None:
    """Extract the authored start year, preserving legitimate pre-1900 dates."""
    if pd.isna(value):
        return None
    match = re.match(r"^\s*(\d{4})", str(value))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1700 <= year <= 2100 else None


def expand_equipment_rows(plants: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Expand each plant row to one row per supported process."""
    records: list[dict[str, object]] = []
    missing_start_date_count = 0

    for _, row in plants.iterrows():
        year = _start_year(row.get("Start date"))
        if year is None:
            missing_start_date_count += 1
            continue

        equipment_text = str(row.get("Main production equipment", ""))
        equipment = {item.strip() for item in equipment_text.split(";")}
        if "OHF" in equipment:
            equipment.remove("OHF")
            equipment.add("EAF")

        status = str(row.get("Capacity operating status", "unknown")).strip().lower()
        for process in PROCESS_ORDER:
            if process not in equipment:
                continue
            capacity = _capacity(row, process)
            if capacity <= 0:
                continue
            records.append(
                {
                    "start_year": year,
                    "process": process,
                    "status": status,
                    "capacity_ktpa": capacity,
                }
            )

    return pd.DataFrame.from_records(records), missing_start_date_count


def make_plot(data: pd.DataFrame, output_path: Path, missing_dates: int) -> None:
    totals = data.groupby(["start_year", "process", "status"])["capacity_ktpa"].sum()
    years = range(int(data["start_year"].min()), int(data["start_year"].max()) + 1)

    fig, ax = plt.subplots(figsize=(22, 10), constrained_layout=True)
    bottoms = pd.Series(0.0, index=list(years))
    statuses = [s for s in STATUS_ORDER if s in data["status"].unique()]
    statuses += sorted(set(data["status"]) - set(statuses))

    for process in PROCESS_ORDER:
        for status in statuses:
            values = pd.Series(
                [totals.get((year, process, status), 0.0) for year in years],
                index=list(years),
            )
            ax.bar(
                list(years),
                values,
                bottom=bottoms,
                width=0.86,
                color=PROCESS_COLORS[process],
                edgecolor="#202020",
                linewidth=0.25,
                hatch=STATUS_HATCHES.get(status, "***"),
            )
            bottoms += values

    ax.set_title("European iron and steel capacity by plant start year", fontsize=17, pad=16)
    ax.set_xlabel("Start date (year)")
    ax.set_ylabel("Nominal capacity (ktpa)")
    ax.set_xlim(min(years) - 1, max(years) + 1)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)

    start_tick = ((min(years) + 9) // 10) * 10
    ax.set_xticks(list(range(start_tick, max(years) + 1, 10)))
    ax.tick_params(axis="x", rotation=45)

    process_legend = [Patch(facecolor=PROCESS_COLORS[p], edgecolor="#202020", label=p) for p in PROCESS_ORDER]
    status_legend = [
        Patch(
            facecolor="white",
            edgecolor="#202020",
            hatch=STATUS_HATCHES.get(s, "***"),
            label=s,
        )
        for s in statuses
    ]
    first_legend = ax.legend(
        handles=process_legend,
        title="Main production equipment",
        loc="upper left",
        ncol=len(process_legend),
        frameon=True,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=status_legend,
        title="Capacity operating status",
        loc="upper right",
        ncol=2,
        frameon=True,
    )

    ax.text(
        0.0,
        -0.12,
        f"Source: master_input_0.xlsx, Region = Europe. "
        f"{missing_dates} plant rows with unknown/unparseable start dates excluded. "
        "OHF is included in EAF, matching the Steel-IQ main flow.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    plants = pd.read_excel(args.input, sheet_name=SHEET_NAME)
    european_plants = plants.loc[plants["Region"].eq("Europe")].copy()
    expanded, missing_dates = expand_equipment_rows(european_plants)
    if expanded.empty:
        raise RuntimeError("No European capacity records with valid start dates were found")

    make_plot(expanded, args.output, missing_dates)
    csv_path = args.output.with_suffix(".csv")
    expanded.sort_values(["start_year", "process", "status"]).to_csv(csv_path, index=False)

    print(f"Read {len(european_plants)} European plant rows")
    print(f"Plotted {len(expanded)} process records; excluded {missing_dates} rows without a start year")
    print(f"Plot: {args.output.resolve()}")
    print(f"Expanded data: {csv_path.resolve()}")
    print("Capacity totals (ktpa):")
    print(expanded.groupby("process")["capacity_ktpa"].sum().reindex(PROCESS_ORDER).to_string())


if __name__ == "__main__":
    main()
